#!/usr/bin/env python3
"""Fit ctxpack's cost-model constants against tiktoken ground truth.

This is the script that produced ``PARAMS`` in ``ctxpack/tokens.py``. It is
checked in so the constants are reproducible rather than magic numbers, and so
anyone who disagrees with the fit can re-run it on a corpus that looks more
like their own code.

    python3 scripts/fit_params.py --root ~/code --root ~/src
    python3 scripts/fit_params.py --root ~/code --write

Method: split each sample into pre-token pieces once (the expensive part),
hold out 30% of the corpus, then optimise the ``chars_per_token`` /
``free_prefix`` pairs on the training half by coordinate descent with random
restarts. The number that matters is the holdout MAPE, not the training one.

MAPE rather than least squares, because packing cares about relative error on
every file rather than total error dominated by whichever file happens to be
largest. Stratified by extension, because a Python-heavy checkout must not be
allowed to quietly define the constants for every other language.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ctxpack.tokens import _CLASS_KEYS, PARAMS, piece_costs

TEXT_EXTS = {
    ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".go", ".rs",
    ".java", ".kt", ".rb", ".php", ".c", ".h", ".cc", ".cpp", ".hpp", ".cs",
    ".swift", ".scala", ".sql", ".graphql", ".proto", ".md", ".mdx", ".rst",
    ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".css", ".scss",
    ".html", ".sh", ".zsh", ".bash", ".lua", ".ex", ".exs", ".vue", ".svelte",
}

SKIP_DIRS = {
    "node_modules", ".git", "dist", "build", "target", "__pycache__",
    ".venv", "venv", ".next", ".nuxt", ".cache", "vendor", ".tox",
    ".mypy_cache", ".pytest_cache", "coverage", ".gradle", "Pods", "out",
}

MAX_BYTES = 400_000

#: A sample is a per-class length histogram plus the true token count.
Sample = tuple[dict[tuple[int, int], int], int]

#: Search bounds, per class: ``(chars_per_token_lo, chars_per_token_hi,
#: free_prefix_hi)``.
BOUNDS: dict[str, tuple[float, float, float]] = {
    "letters": (3.0, 8.0, 4.0),
    "digits": (1.2, 4.0, 3.0),
    "punct": (1.2, 4.0, 2.0),
    "ws": (2.0, 12.0, 4.0),
    "mixed": (1.5, 8.0, 3.0),
    "underscore": (1.2, 6.0, 6.0),
    # Accented text, CJK and emoji. Wide bounds: a CJK ideograph is often one
    # token, while a stray "é" inside a Swedish word is often two.
    "nonascii": (0.4, 4.0, 2.0),
}

#: Classes the optimiser is allowed to move.
#:
#: Only ``letters``, ``punct``, ``ws`` and ``nonascii`` are actually
#: *identifiable* from a source-code corpus. Every piece gets a free prefix that
#: absorbs short pieces, and the digit / underscore / mixed classes are
#: overwhelmingly made of 1-3 character pieces that fall entirely inside that
#: prefix -- so their ratios change nothing and the optimiser pins them to
#: whatever bound it was given. Fitting them anyway produced constants like
#: "1.2 chars per token" for ``__``, which is meaningless and misleading to
#: anyone reading the source.
#:
#: The fixed classes keep principled values: BPE merges runs of up to three
#: digits, which is exactly what ``digits`` encodes below.
FIT_KEYS = ("letters", "punct", "ws", "nonascii")
FIXED_PARAMS: dict[str, tuple[float, float]] = {
    "digits": (3.0, 3.0),  # "2024" is two tokens; "42" is one
    "mixed": (3.5, 0.0),
    "underscore": (3.0, 0.0),
}


def _clamp(key: str, axis: int, value: float) -> float:
    lo, hi, prefix_hi = BOUNDS[key]
    if axis == 0:
        return round(min(max(value, lo), hi), 4)
    return round(min(max(value, 0.0), prefix_hi), 4)


def collect(roots: list[Path], limit: int, seed: int) -> list[Path]:
    """Walk ``roots`` for text files, stratified down to ``limit``."""
    buckets: dict[str, list[Path]] = {}
    seen: set[Path] = set()

    for root in roots:
        if not root.exists():
            print(f"  ! skipping missing root {root}", file=sys.stderr)
            continue
        if root.is_file():
            buckets.setdefault(root.suffix.lower(), []).append(root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext in TEXT_EXTS:
                    path = Path(dirpath) / name
                    if path not in seen:
                        seen.add(path)
                        buckets.setdefault(ext, []).append(path)

    rng = random.Random(seed)
    for key in buckets:
        buckets[key] = sorted(buckets[key])
        rng.shuffle(buckets[key])

    if limit and sum(len(v) for v in buckets.values()) > limit:
        # Round-robin across extensions so rare languages are not drowned out.
        picked: list[Path] = []
        while len(picked) < limit and any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key] and len(picked) < limit:
                    picked.append(buckets[key].pop())
        return picked

    picked = [p for key in sorted(buckets) for p in buckets[key]]
    return picked


def histogram(costs: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    """Collapse ``(class, length)`` pairs into counts keyed by the same pair.

    A typical source file has ~2000 pre-token pieces but only ~90 distinct
    ``(class, length)`` combinations. Collapsing once up front turns each
    candidate-model evaluation from thousands of operations into dozens, which
    is the difference between a fit that finishes in seconds and one that does
    not finish at all.
    """
    hist: dict[tuple[int, int], int] = {}
    for key in costs:
        hist[key] = hist.get(key, 0) + 1
    return hist


def load_samples(paths: list[Path], enc) -> list[Sample]:
    samples: list[Sample] = []
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
        truth = len(enc.encode(text, disallowed_special=()))
        samples.append((histogram(piece_costs(text)), truth))
    return samples


def evaluate(params: dict[str, tuple[float, float]], samples: list[Sample]) -> float:
    """Mean absolute percentage error of a candidate cost model."""
    ratios = [params[k][0] for k in _CLASS_KEYS]
    prefixes = [params[k][1] for k in _CLASS_KEYS]
    error = 0.0
    seen = 0
    for hist, truth in samples:
        total = 0.0
        for (cls, n), count in hist.items():
            # Must mirror ctxpack.tokens.count_from_costs exactly: a free
            # prefix per class, then a characters-per-token ratio above it.
            span = n - prefixes[cls]
            per = 1.0 if span <= 0 else span / ratios[cls]
            total += per * count
        if truth:
            error += abs(max(1, math.ceil(total)) - truth) / truth
        seen += 1
    return error / seen if seen else 0.0


def perturb(params: dict[str, tuple[float, float]], rng) -> dict[str, tuple[float, float]]:
    """Random restart point."""
    out = dict(params)
    for key in FIT_KEYS:
        ratio, prefix = params[key]
        out[key] = (
            _clamp(key, 0, ratio * rng.uniform(0.7, 1.5)),
            _clamp(key, 1, prefix + rng.uniform(-1.5, 1.5)),
        )
    return out


def fit(
    params: dict[str, tuple[float, float]],
    samples: list[Sample],
    rng,
    restarts: int = 8,
) -> tuple[dict[str, tuple[float, float]], float]:
    """Coordinate descent with a shrinking window, over several restarts.

    Coordinate descent is greedy and the objective is lumpy (``ceil`` plus
    per-file relative error), so a single run lands in whatever basin the
    parameter order happened to walk into. Two runs of the same code with
    slightly different bounds disagreed by 0.8 points of holdout MAPE, which is
    exactly the kind of difference you do not want in a shipped constant.
    Restarts are scored on the *training* half; only the winner is reported
    against the holdout, so the holdout stays honest.
    """
    best_params = dict(params)
    best_score = evaluate(params, samples)

    for attempt in range(restarts):
        start = dict(params) if attempt == 0 else perturb(params, rng)
        candidate, score = _descend(start, samples)
        if score < best_score:
            best_params, best_score = candidate, score

    return best_params, best_score


def _descend(
    params: dict[str, tuple[float, float]], samples: list[Sample]
) -> tuple[dict[str, tuple[float, float]], float]:
    best = evaluate(params, samples)

    for window in (3.0, 1.5, 0.6, 0.25, 0.1, 0.04, 0.015):
        for _ in range(60):
            improved = False
            for key in FIT_KEYS:
                for axis in (0, 1):
                    ratio, prefix = params[key]
                    current = ratio if axis == 0 else prefix
                    for step in (-6, -3, -2, -1, 1, 2, 3, 6):
                        cand = _clamp(key, axis, current + step * window)
                        if cand == current:
                            continue
                        params[key] = (cand, prefix) if axis == 0 else (ratio, cand)
                        score = evaluate(params, samples)
                        if score < best - 1e-9:
                            best, current, improved = score, cand, True
                        else:
                            params[key] = (ratio, prefix) if axis == 0 else (ratio, cand)
                    params[key] = (current, prefix) if axis == 0 else (ratio, current)
            if not improved:
                break

    return params, best


def patch_tokens(params: dict[str, tuple[float, float]], target: Path) -> None:
    """Rewrite the ``PARAMS`` literal in tokens.py with the fitted values.

    Matches by regex rather than by ``repr`` so it survives someone reformatting
    ``4.30`` to ``4.3`` or reflowing the literal.
    """
    text = target.read_text(encoding="utf-8")
    for key in _CLASS_KEYS:
        ratio, prefix = params[key]
        pattern = re.compile(rf'("{re.escape(key)}":\s*)\([^)]*\)')
        replacement = rf"\g<1>({round(ratio, 3)!r}, {round(prefix, 3)!r})"
        text, n = pattern.subn(replacement, text, count=1)
        if n != 1:
            raise SystemExit(
                f"could not locate PARAMS[{key!r}] in {target}; update it by hand"
            )
    target.write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit ctxpack cost-model constants.")
    ap.add_argument("--root", action="append", default=[], help="dir or file to sample")
    ap.add_argument("--limit", type=int, default=900, help="max files (0 = all)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--encoding", default="o200k_base")
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--write", action="store_true", help="patch tokens.py in place")
    args = ap.parse_args()

    roots = [Path(r).expanduser() for r in args.root] or [Path.cwd()]
    try:
        import tiktoken
    except ImportError:
        print("fit_params needs tiktoken: pip install 'ctxpack[exact]'", file=sys.stderr)
        return 1

    paths = collect(roots, args.limit, args.seed)
    if not paths:
        print("no text files found", file=sys.stderr)
        return 1

    print(f"tokenising {len(paths)} files with {args.encoding} ...")
    samples = load_samples(paths, tiktoken.get_encoding(args.encoding))
    if len(samples) < 40:
        print(f"only {len(samples)} usable samples; refusing to fit", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * (1 - args.holdout)))
    train, test = shuffled[:cut], shuffled[cut:]

    baseline = dict(PARAMS)
    baseline.update(FIXED_PARAMS)
    print(
        f"train {len(train)} files / holdout {len(test)} files\n"
        f"baseline MAPE: train {evaluate(baseline, train):.2%}"
        f"  holdout {evaluate(baseline, test):.2%}"
    )
    print("optimising ...", file=sys.stderr)

    params, train_mape = fit(baseline, train, random.Random(args.seed))

    print("\nfitted constants (chars/token, free_prefix):")
    for key in _CLASS_KEYS:
        ratio, prefix = params[key]
        tag = "fitted" if key in FIT_KEYS else "fixed "
        print(
            f"  {key:<11} {ratio:7.3f}, {prefix:6.3f}  [{tag}]"
            f"    was {PARAMS[key][0]:.3f}, {PARAMS[key][1]:.3f}"
        )
    print(
        f"\ntrain MAPE {train_mape:.2%}   "
        f"HOLDOUT MAPE {evaluate(params, test):.2%}"
    )

    if args.write:
        target = Path(__file__).resolve().parent.parent / "src" / "ctxpack" / "tokens.py"
        patch_tokens(params, target)
        print(f"patched {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
