#!/usr/bin/env python3
"""Benchmark ctxpack's token estimator and packer, and catch regressions.

Two jobs:

1. **Measure.** Substantiate the accuracy claims in the README with numbers
   this machine can reproduce, rather than numbers someone typed once.
2. **Guard.** ``--check`` compares against a stored baseline and exits non-zero
   when something regressed, so a change that silently makes the estimator
   worse cannot land.

    python3 scripts/bench.py --root ~/code --root ~/repos
    python3 scripts/bench.py --root ~/code --baseline bench.json --write
    python3 scripts/bench.py --root ~/code --baseline bench.json --check
    python3 scripts/bench.py --synthetic          # no corpus needed

The synthetic shapes matter as much as the real corpus: "200 identical files",
"one 5MB file", "5000 tiny files" and "depth 40" are the inputs that break
greedy packers, and none of them show up in a small sample of real repos.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ctxpack.pack import Budget, Packer
from ctxpack.render import render
from ctxpack.tokens import (
    PARAMS,
    Tokenizer,
    calibration_factor,
    mean_abs_pct_error,
)
from ctxpack.walk import discover

TEXT_EXTS = {
    ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".go", ".rs", ".java",
    ".md", ".rst", ".json", ".yaml", ".yml", ".toml", ".sql", ".sh", ".css",
    ".html", ".php", ".rb", ".kt", ".swift", ".c", ".h", ".cpp",
}

SKIP_DIRS = {
    "node_modules", ".git", "dist", "build", "target", "__pycache__",
    ".venv", "venv", ".next", ".cache", "vendor", ".tox", "site-packages",
}

#: Per-extension sample size. Capped so a Python-heavy checkout cannot define
#: the numbers for every other language.
PER_EXT = 220

#: Files larger than this are sampled rather than fully read.
MAX_BYTES = 300_000

#: MAPE regression that fails --check. Deliberately loose: this is a canary for
#: "something got structurally worse", not a gate on the third decimal place.
MAPE_TOLERANCE = 0.03


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------


def collect(roots: list[Path], seed: int) -> dict[str, list[Path]]:
    """Sample text files per extension across ``roots``."""
    buckets: dict[str, list[Path]] = defaultdict(list)
    seen: set[Path] = set()

    for root in roots:
        if not root.exists():
            print(f"  ! skipping missing root {root}", file=sys.stderr)
            continue
        if root.is_file():
            buckets[root.suffix.lower()].append(root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext not in TEXT_EXTS:
                    continue
                path = Path(dirpath) / name
                if path in seen:
                    continue
                seen.add(path)
                buckets[ext].append(path)

    rng = random.Random(seed)
    for ext in buckets:
        rng.shuffle(buckets[ext])
        buckets[ext] = buckets[ext][:PER_EXT]
    return dict(buckets)


def _stats(pairs: list[tuple[int, int]]) -> dict:
    """Accuracy metrics for a set of ``(estimate, truth)`` pairs.

    Two different means are reported because they answer different questions.

    MAPE treats every file equally, which over-weights tiny ones: a 5-token
    ``FUNDING.yml`` counted as 4 instead of 6 is 20% wrong but costs nothing,
    and a sample containing several of them reports a scary number that has no
    bearing on packing. The token-weighted figure -- total miscounted tokens over
    total tokens -- is what actually decides whether a bundle fits its budget.
    A corpus of real files routinely shows the two disagreeing by 2x.
    """
    errors = sorted(abs(h - e) / e for h, e in pairs)
    absolute = sum(abs(h - e) for h, e in pairs)
    total = sum(e for _, e in pairs)
    return {
        "files": len(pairs),
        "mape": round(mean_abs_pct_error(pairs), 5),
        "weighted_mape": round(absolute / total, 5) if total else 0.0,
        "p90": round(errors[int(len(errors) * 0.9)], 5),
        "max": round(errors[-1], 5),
        "over_count_rate": round(
            sum(1 for h, e in pairs if h > e) / len(pairs), 4
        ),
        "median_tokens": round(statistics.median(e for _, e in pairs)),
    }


def measure_accuracy(buckets: dict[str, list[Path]], exact: Tokenizer) -> dict:
    """Per-extension accuracy of the shipped estimator against tiktoken."""
    estimator = Tokenizer("o200k_base", mode="never", calibration=1.0)
    per_ext: dict[str, dict] = {}
    all_pairs: list[tuple[int, int]] = []

    for ext, paths in sorted(buckets.items()):
        pairs: list[tuple[int, int]] = []
        for path in paths:
            try:
                raw = path.read_bytes()[:MAX_BYTES]
            except OSError:
                continue
            if b"\x00" in raw[:4096]:
                continue
            text = raw.decode("utf-8", "replace")
            if not text.strip():
                continue
            truth = exact.count(text).tokens
            if not truth:
                continue
            pairs.append((estimator.count(text).tokens, truth))
        if len(pairs) < 5:
            continue
        all_pairs.extend(pairs)
        per_ext[ext] = _stats(pairs)

    overall = _stats(all_pairs) if all_pairs else _stats([(1, 1)])
    return {
        "per_ext": per_ext,
        **{f"overall_{k}": v for k, v in overall.items() if k != "files"},
        "samples": len(all_pairs),
        "least_squares_factor": round(calibration_factor(all_pairs), 5)
        if all_pairs
        else 1.0,
    }


# ---------------------------------------------------------------------------
# synthetic shapes
# ---------------------------------------------------------------------------


def _write(root: Path, rel: str, body: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def build_shapes(base: Path) -> dict[str, Path]:
    """Build adversarial repo shapes and return their roots."""
    shapes: dict[str, Path] = {}

    # 200 identical files: pure duplicate waste.
    dup = base / "duplicates"
    body = "\n".join(
        f"    exported_name_{i} = compute_value({i}, other_arg)"
        for i in range(40)
    )
    for i in range(200):
        _write(dup, f"locale_{i:03d}/messages.py", f"MESSAGES = {{\n{body}\n}}\n")
    _write(dup, "README.md", "# Duplicates\n" + body + "\n")
    shapes["200 identical files"] = dup

    # One enormous file that would eat any budget.
    big = base / "monolith"
    _write(
        big,
        "huge.py",
        "\n".join(
            f"def generated_function_{i}(a, b, c):\n    return a + b + c + {i}\n"
            for i in range(20_000)
        ),
    )
    _write(big, "README.md", "# Monolith\n\nOne very large file.\n")
    shapes["1 x 20k-line file"] = big

    # Many tiny files: header overhead should dominate if the packer is naive.
    tiny = base / "tiny"
    for i in range(1500):
        _write(tiny, f"pkg{i % 40}/mod_{i}.py", f"VALUE_{i} = {i}\n")
    shapes["1500 tiny files"] = tiny

    # Pathological nesting.
    deep = base / "deep"
    _write(deep, "a" * 40 + "/file.py", "X = 1\n")
    shapes["40-deep nesting"] = deep

    # Mixed realistic repo.
    mixed = base / "mixed"
    _write(mixed, "README.md", "# Mixed\n\nA realistic small project.\n")
    _write(mixed, "CHANGELOG.md", "# Changelog\n" + "\n".join(f"- {i}" for i in range(500)))
    _write(mixed, "src/app/main.py", "def main():\n    print('hi')\n")
    _write(mixed, "src/app/util.py", "X = 1\n" * 100)
    _write(mixed, "tests/test_main.py", "def test_main():\n    assert True\n")
    _write(mixed, "package-lock.json", '{"lockfileVersion": 3}\n')
    _write(mixed, "logo.png", "")
    (mixed / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")
    shapes["mixed realistic"] = mixed

    return shapes


def measure_packing(shapes: dict[str, Path]) -> list[dict]:
    """For each shape and budget, record fill rate and any overrun."""
    tokenizer = Tokenizer(mode="never")
    rows: list[dict] = []

    for label, root in shapes.items():
        discovery = discover(root)
        for budget in (2_000, 10_000, 60_000):
            for mode in ("balanced", "coverage", "depth"):
                start = time.perf_counter()
                result = Packer(
                    tokenizer, Budget(total=budget), mode=mode
                ).pack(discovery)
                elapsed = time.perf_counter() - start
                rows.append(
                    {
                        "shape": label,
                        "budget": budget,
                        "mode": mode,
                        "files": len(result.documents),
                        "discovered": result.discovered,
                        "fill": round(result.accounted / budget, 4),
                        "over": result.accounted - budget,
                        "truncated": len(result.truncated),
                        "duplicates": len(result.duplicates),
                        "seconds": round(elapsed, 3),
                    }
                )
    return rows


def measure_integrity(shapes: dict[str, Path]) -> list[str]:
    """Confirm every rendered format is actually parseable, including XML."""
    tokenizer = Tokenizer(mode="never")
    problems: list[str] = []

    for label, root in shapes.items():
        result = Packer(tokenizer, Budget(total=8_000)).pack(discover(root))
        try:
            ElementTree.fromstring(render(result, "xml"))
        except ElementTree.ParseError as exc:
            problems.append(f"{label}: xml does not parse: {exc}")
        try:
            json.loads(render(result, "json"))
        except ValueError as exc:
            problems.append(f"{label}: json does not parse: {exc}")
        if not render(result, "markdown").strip():
            problems.append(f"{label}: markdown is empty")

    # Hostile content, which is where renderers usually break.
    #
    # This runs in its own mkdtemp and is removed with rmtree. An earlier
    # version wrote these fixtures into a shared scratch directory and unlinked
    # whatever it found there, which would have deleted other processes' files.
    # Never clean up anything this script did not create.
    scratch = Path(tempfile.mkdtemp(prefix="ctxpack-hostile-"))
    try:
        payloads = {
            "fence.py": "X = 1\n```\nnot the end\n```\n",
            "fence4.md": "````\n```\n````\n",
            "cdata.py": 'X = "]]> breaks CDATA"\n',
            "cdata2.py": "X = ']]]]><![CDATA[>'\n",
            "amp.py": 'X = "a & b < c > d"\n',
            "ctrl.py": "X = 1\n" + "".join(chr(c) for c in range(1, 9)) + "\n",
            "quote.py": 'X = \'he said "hi" and \\ escaped\'\n',
            "unicode.py": "X = '日本語 🎉 Ünïcödé'\n",
            "trailing.py": "X = 1",
            "empty.py": "",
            "blank.py": "\n\n\n",
            "onlyfence.py": "```",
        }
        for name, body in payloads.items():
            (scratch / name).write_text(body, encoding="utf-8")

        result = Packer(tokenizer, Budget(total=40_000)).pack(discover(scratch))
        for fmt, checker in (
            ("xml", lambda t: ElementTree.fromstring(t)),
            ("json", json.loads),
        ):
            try:
                checker(render(result, fmt))
            except Exception as exc:
                problems.append(f"hostile content: {fmt} failed: {exc}")

        # A markdown fence must be able to contain every payload intact.
        markdown = render(result, "markdown")
        if "not the end" not in markdown:
            problems.append("markdown: fenced content was lost")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    return problems


def measure_import_graph() -> list[str]:
    """Canary for ``ctxpack.deps``: the graph must actually resolve.

    A resolver that trusts import specifiers verbatim produces a graph that
    *looks* built -- correct shape, no errors -- while resolving almost nothing.
    That happened on a TypeScript repo using ``NodeNext``, where
    ``from './x.js'`` means ``./x.ts``: ~670 resolvable imports collapsed to 2,
    and nothing errored. So this asserts the specific expected edges rather than
    counting them, because a count alone cannot tell "complete" from "empty".
    """
    from ctxpack.deps import build_graph, find_entrypoints

    problems: list[str] = []

    def check(
        label: str,
        files: dict[str, str],
        expected: dict[str, set[str]],
    ) -> None:
        paths = list(files)
        graph = build_graph(paths, lambda p: files.get(p))
        entry = find_entrypoints(paths)
        for source, targets in expected.items():
            missing = targets - graph.get(source, set())
            if missing:
                problems.append(
                    f"import graph {label}: {source} did not resolve "
                    f"{sorted(missing)}"
                )
        edges = sum(len(v) for v in graph.values())
        print(
            f"  import graph {label}: {edges} edges, "
            f"{len(entry)} entrypoints",
            file=sys.stderr,
        )

    py = {
        "src/app/__init__.py": "from .server import serve\n",
        "src/app/main.py": "from .server import serve\nfrom .config import load\n",
        "src/app/server.py": "from .store import Store\nimport os\n",
        "src/app/config.py": "import pathlib\n",
        "src/app/store.py": "import json\n",
    }
    check(
        "python package",
        py,
        {
            "src/app/__init__.py": {"src/app/server.py"},
            "src/app/main.py": {"src/app/server.py", "src/app/config.py"},
            "src/app/server.py": {"src/app/store.py"},
        },
    )

    # NodeNext-style ``.js`` specifiers naming ``.ts`` sources: every one of
    # these is resolvable only via extension substitution.
    ts = {
        "src/index.ts": "export { run } from './server.js';\n",
        "src/server.ts": "import { store } from './store.js';\n"
        "import { cfg } from './config.js';\n",
        "src/store.ts": "import { log } from './util/logger.js';\n"
        "export const store = 1;\n",
        "src/config.ts": "export const cfg = {};\n",
        "src/util/logger.ts": "export const log = () => {};\n",
    }
    check(
        "typescript nodenext",
        ts,
        {
            "src/index.ts": {"src/server.ts"},
            "src/server.ts": {"src/store.ts", "src/config.ts"},
            "src/store.ts": {"src/util/logger.ts"},
        },
    )

    return problems


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def accuracy_table(accuracy: dict) -> str:
    rows = [
        "| ext | files | median tokens | mean error | token-weighted | p90 | over-counts |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ext, stats in sorted(accuracy["per_ext"].items()):
        rows.append(
            f"| `{ext}` | {stats['files']} | {stats['median_tokens']:,} | "
            f"{stats['mape']:.1%} | {stats['weighted_mape']:.1%} | "
            f"{stats['p90']:.1%} | {stats['over_count_rate']:.0%} |"
        )
    overall = accuracy
    rows.append(
        f"| **all** | {accuracy['samples']:,} | — | "
        f"{overall['overall_mape']:.2%} | {overall['overall_weighted_mape']:.2%} | "
        f"{overall['overall_p90']:.1%} | — |"
    )
    return "\n".join(rows)


def packing_table(rows: list[dict]) -> str:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["shape"]].append(row)
    out = ["| shape | files found | budget | fill | worst overrun | slowest |",
           "|---|---:|---:|---:|---:|---:|"]
    for shape, group in grouped.items():
        fill = statistics.mean(r["fill"] for r in group)
        worst = max(r["over"] for r in group)
        slowest = max(r["seconds"] for r in group)
        out.append(
            f"| {shape} | {group[0]['discovered']:,} | "
            f"{min(r['budget'] for r in group):,}-{max(r['budget'] for r in group):,} | "
            f"{fill:.0%} | {worst:+,} | {slowest:.2f}s |"
        )
    return "\n".join(out)


def compare(baseline: dict, current: dict) -> list[str]:
    """Report regressions. Empty list means nothing got worse."""
    problems: list[str] = []

    base_acc = baseline.get("accuracy", {})
    cur_acc = current.get("accuracy", {})
    for key, label in (
        ("overall_mape", "mean error"),
        ("overall_weighted_mape", "token-weighted error"),
    ):
        was, now = base_acc.get(key), cur_acc.get(key)
        if was and now and now > was + MAPE_TOLERANCE:
            problems.append(
                f"{label} regressed: {was:.2%} -> {now:.2%} "
                f"(tolerance {MAPE_TOLERANCE:.0%})"
            )

    base_fill = statistics.mean(r["fill"] for r in baseline.get("packing", [])) if baseline.get("packing") else 0
    cur_fill = statistics.mean(r["fill"] for r in current.get("packing", [])) if current.get("packing") else 0
    if base_fill and cur_fill < base_fill - 0.10:
        problems.append(
            f"budget fill rate dropped: {base_fill:.0%} -> {cur_fill:.0%}"
        )

    problems.extend(
        f"budget overrun: {row['shape']} / {row['mode']} / "
        f"b={row['budget']} over by {row['over']} tokens"
        for row in current.get("packing", [])
        if row["over"] > 0
    )
    problems.extend(f"integrity: {p}" for p in current.get("integrity", []))
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", action="append", default=[], help="dir to sample")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="deprecated; kept for compat")
    ap.add_argument("--synthetic", action="store_true", help="skip the real corpus")
    ap.add_argument("--baseline", default=None, help="baseline JSON for --check/--write")
    ap.add_argument("--write", action="store_true", help="write the baseline")
    ap.add_argument("--check", action="store_true", help="fail on regression")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    args = ap.parse_args()

    # Everything this script writes lives under a private mkdtemp, removed on
    # exit. It never touches a directory it did not create.
    scratch = Path(tempfile.mkdtemp(prefix="ctxpack-bench-"))
    report: dict = {"params": {k: list(v) for k, v in PARAMS.items()}}

    try:
        shapes = build_shapes(scratch)
        print(f"synthetic shapes: {', '.join(shapes)}", file=sys.stderr)

        if not args.synthetic:
            roots = [Path(r).expanduser() for r in args.root] or [Path.cwd()]
            print(
                f"sampling corpus from {', '.join(str(r) for r in roots)} ...",
                file=sys.stderr,
            )
            try:
                import tiktoken  # noqa: F401
            except ImportError:
                print(
                    "accuracy measurement needs tiktoken; use --synthetic or "
                    "pip install 'ctxpack[exact]'",
                    file=sys.stderr,
                )
                return 1
            buckets = collect(roots, args.seed)
            accuracy = measure_accuracy(buckets, Tokenizer("o200k_base", mode="exact"))
            report["accuracy"] = accuracy
            print(
                f"accuracy: {accuracy['samples']:,} files, MAPE "
                f"{accuracy['overall_mape']:.2%}, token-weighted "
                f"{accuracy['overall_weighted_mape']:.2%}",
                file=sys.stderr,
            )

        report["packing"] = measure_packing(shapes)
        report["integrity"] = measure_integrity(shapes) + measure_import_graph()
        for problem in report["integrity"]:
            print(f"INTEGRITY: {problem}", file=sys.stderr)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    baseline: dict = {}
    if args.baseline:
        baseline_path = Path(args.baseline)
        if baseline_path.exists():
            try:
                baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            except ValueError as exc:
                print(f"baseline {baseline_path} is not valid JSON: {exc}", file=sys.stderr)
                return 1
        elif args.check:
            # --check against nothing would pass vacuously, which is worse than
            # failing: it would silently approve a regression forever.
            print(
                f"no baseline at {baseline_path}; run with --write first",
                file=sys.stderr,
            )
            return 1
        else:
            print(f"no baseline at {baseline_path}; starting a fresh one", file=sys.stderr)

    if args.write and args.baseline:
        Path(args.baseline).write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.baseline}", file=sys.stderr)

    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print("## estimator accuracy\n")
        print(accuracy_table(report["accuracy"]) if "accuracy" in report else "_skipped_")
        print("\n## packing\n")
        print(packing_table(report["packing"]))
        print("\n## integrity\n")
        print(
            "all formats parse"
            if not report["integrity"]
            else "\n".join(f"- {p}" for p in report["integrity"])
        )

    if args.check:
        problems = compare(baseline, report)
        if problems:
            print("\nREGRESSIONS:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        print("\nno regressions vs baseline", file=sys.stderr)

    return 0 if not report["integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
