"""Command line interface.

    ctxpack pack ~/code/myapp -b 60000 -f xml > context.xml
    ctxpack count . --limit 30
    ctxpack calibrate --write
    ctxpack explain . -p 'src/**'

``pack`` is the default subcommand, so ``ctxpack . -b 40000`` and
``ctxpack pack . -b 40000`` are the same thing.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from contextlib import suppress
from pathlib import Path

from . import __version__
from .errors import CtxpackError
from .pack import MODES, Budget, Packer
from .rank import rank_all
from .render import FORMATS, render
from .tokens import (
    KNOWN_ENCODINGS,
    Tokenizer,
    calibration_factor,
    mean_abs_pct_error,
    save_calibration,
)
from .walk import discover, read_text

COMMANDS = ("pack", "count", "calibrate", "explain")
TOP_LEVEL_FLAGS = {"-h", "--help", "--version"}

_SIZE_SUFFIXES = {
    "k": 1_000, "kb": 1_000, "m": 1_000_000, "mb": 1_000_000,
    "g": 1_000_000_000, "gb": 1_000_000_000,
}


def _parse_size(value: str) -> int:
    """Accept ``500000``, ``500k``, ``5M``."""
    text = value.strip().lower().rstrip("b")
    for suffix in sorted(_SIZE_SUFFIXES, key=len, reverse=True):
        if text.endswith(suffix):
            head = text[: -len(suffix)]
            if head:
                try:
                    return int(float(head) * _SIZE_SUFFIXES[suffix])
                except ValueError:
                    break
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"cannot read size {value!r}; try 500000, 500k or 5M"
        ) from None


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "path", nargs="?", default=".", help="file or directory to pack (default: .)"
    )
    parser.add_argument("-C", "--root", default=None, help="alias for the path argument")
    parser.add_argument(
        "-x", "--exclude", action="append", default=[], metavar="GLOB",
        help="skip paths matching GLOB (repeatable)",
    )
    parser.add_argument(
        "-i", "--include", action="append", default=[], metavar="GLOB",
        help="only include paths matching GLOB (repeatable)",
    )
    parser.add_argument(
        "-e", "--encoding", default="o200k_base", choices=KNOWN_ENCODINGS,
        help="tokenizer encoding (default: o200k_base)",
    )
    parser.add_argument(
        "--exact", default="auto", choices=("auto", "exact", "never"),
        help="use tiktoken for ground truth, or the bundled estimator",
    )
    parser.add_argument(
        "--calibration", type=float, default=None, metavar="F",
        help="override the estimator calibration factor",
    )
    parser.add_argument(
        "--max-size", type=_parse_size, default=None, metavar="SIZE",
        help="ignore files larger than SIZE (e.g. 2M)",
    )
    parser.add_argument(
        "--no-default-ignores", action="store_true",
        help="do not apply ctxpack's built-in noise ignores",
    )
    parser.add_argument(
        "--no-gitignore", action="store_true",
        help="do not read .gitignore / .ctxpackignore",
    )


def _resolve_root(args: argparse.Namespace) -> str:
    root = getattr(args, "root", None) or args.path
    if not root:
        raise CtxpackError("no path given")
    return root


def _tokenizer(args: argparse.Namespace) -> Tokenizer:
    return Tokenizer(
        args.encoding, mode=args.exact, calibration=args.calibration
    )


def _discover(args: argparse.Namespace, root: str | None = None):
    return discover(
        root or _resolve_root(args),
        exclude=tuple(args.exclude),
        include=tuple(args.include),
        max_size=args.max_size,
        use_default_ignores=not args.no_default_ignores,
        respect_gitignore=not args.no_gitignore,
    )


def _budget(args: argparse.Namespace) -> Budget:
    return Budget(
        total=args.budget,
        per_file_frac=args.per_file_frac,
        per_ext_frac=args.per_ext_frac,
        per_dir_frac=args.per_dir_frac,
        max_per_dir=args.max_per_dir,
        max_per_ext=args.max_per_ext,
        manifest=args.manifest,
        dedupe=not args.no_dedupe,
        dedupe_threshold=args.dedupe_threshold,
        truncate=not args.no_truncate,
        max_scan=args.max_scan,
    )


# ---------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------


def cmd_pack(args: argparse.Namespace) -> int:
    tokenizer = _tokenizer(args)
    budget = _budget(args)
    progress = None if args.quiet else _progress
    discovery = _discover(args)

    if not discovery.files:
        raise CtxpackError(
            f"no text files found under {discovery.root}"
            + (" (try --no-default-ignores or --include)" if args.include or args.no_default_ignores else "")
        )

    packer = Packer(tokenizer, budget, mode=args.mode, progress=progress)
    result = packer.pack(discovery)
    text = render(result, args.format)

    if args.out:
        Path(args.out).expanduser().write_text(text, encoding="utf-8")
        if not args.quiet:
            actual = tokenizer.count(text)
            print(
                f"ctxpack: wrote {args.out} "
                f"({actual.tokens:,} tokens {actual.method}, "
                f"budget {budget.total:,})",
                file=sys.stderr,
            )
    else:
        if args.stats and not args.quiet:
            _print_stats(result, tokenizer, file=sys.stderr)
        if not args.quiet and sys.stdout.isatty():
            print(
                "ctxpack: writing bundle to a terminal; pipe it or use --out",
                file=sys.stderr,
            )
        sys.stdout.write(text)

    if args.stats and args.out and not args.quiet:
        _print_stats(result, tokenizer, file=sys.stderr)
    return 0


def _progress(message: str) -> None:
    print(message, file=sys.stderr)


def _print_stats(result, tokenizer, *, file) -> None:
    lines = [
        "",
        f"root              {result.root}",
        f"budget            {result.budget:,} tokens",
        f"index             {len(result.manifest):,} entries, "
        f"{result.manifest_tokens:,} tokens",
        f"files packed      {len(result.documents):,} of {result.discovered:,} "
        f"discovered ({result.ignored:,} ignored)",
        f"content           {result.content_tokens:,} tokens",
        f"accounted         {result.accounted:,} tokens "
        f"({result.over_budget:+,} vs budget)",
        f"truncated         {len(result.truncated):,}",
        f"duplicates        {len(result.duplicates):,}",
        f"counted with      {result.encoding} ({result.method})",
    ]
    lines.extend(f"note              {note}" for note in result.notes)
    print("\n".join(lines), file=file)


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


def cmd_count(args: argparse.Namespace) -> int:
    tokenizer = _tokenizer(args)
    discovery = _discover(args)
    ranked = rank_all(discovery.files)

    rows = []
    for item in ranked[: args.limit] if args.limit else ranked:
        text = read_text(item.candidate)
        if text is None:
            continue
        count = tokenizer.count(text)
        rows.append(
            {
                "path": item.candidate.path,
                "tokens": count.tokens,
                "bytes": item.candidate.size,
                "lines": text.count("\n") + 1,
                "chars_per_token": round(count.chars_per_token, 2),
                "score": item.score,
            }
        )

    if args.json:
        payload = {
            "root": str(discovery.root),
            "encoding": tokenizer.encoding,
            "method": tokenizer.method,
            "discovered": len(discovery.files),
            "ignored": discovery.ignored,
            "total_bytes": discovery.total_bytes,
            "total_tokens": sum(r["tokens"] for r in rows),
            "files": rows,
        }
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    shown = len(rows)
    print(
        f"{'':>10} {'tokens':>10} {'bytes':>11} {'c/t':>6} "
        f"{'score':>7}  path",
        file=sys.stderr,
    )
    print("-" * 78, file=sys.stderr)
    for row in rows:
        print(
            f"{row['tokens']:>10,} {row['bytes']:>11,} "
            f"{row['chars_per_token']:>6.2f} {row['score']:>7.2f}  {row['path']}",
            file=sys.stderr,
        )
    total = sum(r["tokens"] for r in rows)
    print("-" * 78, file=sys.stderr)
    print(
        f"{total:>10,} tokens for {shown} files "
        f"({tokenizer.method}, {tokenizer.encoding})",
        file=sys.stderr,
    )
    if args.limit and len(ranked) > shown:
        print(
            f"{len(ranked) - shown:,} more files not shown (--limit 0 for all)",
            file=sys.stderr,
        )
    return 0


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------


def cmd_calibrate(args: argparse.Namespace) -> int:
    roots = [Path(p).expanduser() for p in args.path] or [Path(".")]
    exact = Tokenizer(args.encoding, mode="exact")
    heuristic = Tokenizer(args.encoding, mode="never", calibration=1.0)

    pairs: list[tuple[int, int]] = []
    total_bytes = 0
    for root in roots:
        found = discover(
            root,
            exclude=(),
            include=(),
            use_default_ignores=not args.no_default_ignores,
            respect_gitignore=not args.no_gitignore,
        )
        for candidate in found.files:
            text = read_text(candidate)
            if not text or not text.strip():
                continue
            pairs.append((heuristic.count(text).tokens, exact.count(text).tokens))
            total_bytes += candidate.size

    if len(pairs) < 5:
        raise CtxpackError(
            f"only {len(pairs)} readable text files found; need at least 5 to fit"
        )

    if args.limit and len(pairs) > args.limit:
        # Deterministic sample: calibration should not change because the file
        # listing order did.
        pairs = random.Random(0).sample(pairs, args.limit)

    factor = calibration_factor(pairs)
    scaled = [(max(1, round(h * factor)), e) for h, e in pairs]
    over = sum(1 for h, e in scaled if h > e)

    print(f"sampled {len(pairs):,} files ({total_bytes / 1e6:.1f} MB)")
    print(f"encoding          {args.encoding} (exact)")
    print(f"raw estimator MAPE {mean_abs_pct_error(pairs):.2%}")
    print(f"least-squares factor {factor:.4f}")
    print(f"MAPE after scaling {mean_abs_pct_error(scaled):.2%}")
    print(
        f"over-estimates {over / len(scaled):.0%} of files "
        f"(over-counting is the safe direction: the bundle stays in budget)"
    )

    if args.write:
        path = save_calibration(
            factor,
            samples=len(pairs),
            mean_abs_pct=mean_abs_pct_error(scaled),
            encoding=args.encoding,
        )
        print(f"wrote {path}")
    else:
        print("\nre-run with --write to persist this factor")
    return 0


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------


def cmd_explain(args: argparse.Namespace) -> int:
    from fnmatch import fnmatch

    discovery = _discover(args)
    ranked = rank_all(discovery.files)
    if args.pattern:
        ranked = [s for s in ranked if fnmatch(s.path, args.pattern)]
    if args.limit:
        ranked = ranked[: args.limit]

    if args.json:
        json.dump(
            {
                "root": str(discovery.root),
                "files": [
                    {
                        "path": s.path,
                        "score": s.score,
                        "signals": [
                            {"key": g.key, "weight": g.weight, "detail": g.detail}
                            for g in s.signals
                        ],
                    }
                    for s in ranked
                ],
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 0

    for item in ranked:
        print(item.explain())
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ctxpack",
        description="Pack any codebase into a token-budgeted context bundle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  ctxpack ~/code/app -b 60000 -f xml -o ctx.xml\n"
            "  ctxpack . -b 40000 -m coverage -o bundle.md\n"
            "  ctxpack count . --limit 25\n"
            "  ctxpack calibrate --write\n"
            "  ctxpack explain . --filter 'src/*'\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"ctxpack {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- pack --------------------------------------------------------------
    pack = subparsers.add_parser(
        "pack", help="build a context bundle (default)"
    )
    _common(pack)
    pack.add_argument(
        "-b", "--budget", type=int, default=32000, metavar="TOKENS",
        help="total token budget (default: 32000)",
    )
    pack.add_argument(
        "-f", "--format", default="markdown", choices=FORMATS, help="output format"
    )
    pack.add_argument(
        "-m", "--mode", default="balanced", choices=MODES,
        help="balanced (default), coverage (breadth first) or depth (rank order)",
    )
    pack.add_argument(
        "-o", "--out", default=None, metavar="FILE", help="write to FILE"
    )
    pack.add_argument(
        "--manifest", default="full", choices=("full", "paths", "none"),
        help="how much index to include (default: full)",
    )
    pack.add_argument(
        "--per-file-frac", type=float, default=0.12,
        help="max share of the budget for one file (default: 0.12)",
    )
    pack.add_argument(
        "--per-dir-frac", type=float, default=0.40,
        help="max share of the budget for one directory (default: 0.40)",
    )
    pack.add_argument(
        "--per-ext-frac", type=float, default=0.55,
        help="max share of the budget for one extension (default: 0.55)",
    )
    pack.add_argument(
        "--max-per-dir", type=int, default=8, help="max files from one directory"
    )
    pack.add_argument(
        "--max-per-ext", type=int, default=0, help="max files of one extension (0: no cap)"
    )
    pack.add_argument(
        "--dedupe-threshold", type=float, default=0.85,
        help="collapse files this similar or more (default: 0.85)",
    )
    pack.add_argument("--no-dedupe", action="store_true", help="disable duplicate collapsing")
    pack.add_argument("--no-truncate", action="store_true", help="drop oversized files instead of cutting them")
    pack.add_argument(
        "--max-scan", type=int, default=3000,
        help="only read the top N files by rank (default: 3000)",
    )
    pack.add_argument("--stats", action="store_true", help="print a summary to stderr")
    pack.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    pack.set_defaults(func=cmd_pack)

    # -- count -------------------------------------------------------------
    count = subparsers.add_parser(
        "count", help="show per-file token counts and rank scores"
    )
    _common(count)
    count.add_argument(
        "-n", "--limit", type=int, default=40, metavar="N",
        help="rows to show, 0 for all (default: 40)",
    )
    count.add_argument("--json", action="store_true", help="machine-readable output")
    count.set_defaults(func=cmd_count)

    # -- calibrate ---------------------------------------------------------
    calibrate = subparsers.add_parser(
        "calibrate", help="fit the token estimator against tiktoken ground truth"
    )
    calibrate.add_argument(
        "path", nargs="*", default=None, help="files or directories to sample"
    )
    calibrate.add_argument(
        "-e", "--encoding", default="o200k_base", choices=KNOWN_ENCODINGS
    )
    calibrate.add_argument(
        "-n", "--limit", type=int, default=600, metavar="N",
        help="max files to sample (default: 600)",
    )
    calibrate.add_argument(
        "--no-default-ignores", action="store_true", help="include ignored files"
    )
    calibrate.add_argument("--no-gitignore", action="store_true")
    calibrate.add_argument(
        "--write", action="store_true", help="persist the factor to the config file"
    )
    calibrate.set_defaults(func=cmd_calibrate)

    # -- explain -----------------------------------------------------------
    explain = subparsers.add_parser(
        "explain", help="show why files rank where they do"
    )
    _common(explain)
    explain.add_argument(
        "-p", "--filter", dest="pattern", default=None, metavar="GLOB",
        help="only explain paths matching GLOB (e.g. 'src/**')",
    )
    explain.add_argument("-n", "--limit", type=int, default=30)
    explain.add_argument("--json", action="store_true")
    explain.set_defaults(func=cmd_explain)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0] not in COMMANDS and argv[0] not in TOP_LEVEL_FLAGS):
        argv.insert(0, "pack")

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CtxpackError as exc:
        print(f"ctxpack: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        # `ctxpack ... | head` should not print a traceback. Closing stdout can
        # itself raise if the interpreter has already torn it down.
        with suppress(OSError):
            sys.stdout.close()
        return 0
    except KeyboardInterrupt:
        print("ctxpack: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
