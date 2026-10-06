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
from .pack import MODES, Budget, Packer, SharedBlock
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

#: Never sourced from a config file. `path` is a positional, and the mechanism
#: that detects which flags were typed only knows about option strings -- so
#: merging it would replace an explicitly given path with its default of ".".
_CLI_ONLY_KEYS = frozenset({"command", "path", "root", "help", "func", "_parser", "_argv"})

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
    parser.add_argument(
        "--no-config", action="store_true",
        help="ignore ctxpack.toml / .ctxpack.toml in this directory or above",
    )
    parser.add_argument(
        "--show-config", action="store_true",
        help="print the effective settings and where they came from, then exit",
    )


def _diff_flags(parser: argparse.ArgumentParser) -> None:
    """Flags for restricting the pack to what a git range changed."""
    parser.add_argument(
        "--diff", default=None, metavar="SPEC",
        help="only pack files changed in this git range, e.g. 'main...HEAD', "
        "'HEAD~3' or just 'main'",
    )
    parser.add_argument(
        "--staged", action="store_true",
        help="with --diff: compare the index against HEAD",
    )
    parser.add_argument(
        "--uncommitted", action="store_true",
        help="with --diff: compare the worktree against HEAD (the default)",
    )
    parser.add_argument(
        "--untracked", action="store_true",
        help="with --diff: also include files git has never seen",
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


def _restrict_to_diff(args: argparse.Namespace, found) -> None:
    """Narrow ``found`` to the files a git range touched, in place.

    Reviewing a change is the case where the whole repository is least useful,
    so ``--diff`` is the difference between 200 files of context and the 6 that
    actually moved.
    """
    from .gitdiff import changed_files, is_git_repo, parse_refs, relativise, repo_root

    root = found.root
    if not is_git_repo(root):
        raise CtxpackError(
            f"{root} is not inside a git work tree, so --diff has nothing to "
            "compare against"
        )

    base, head = parse_refs(args.diff)
    paths = changed_files(
        root,
        base,
        head,
        staged=args.staged,
        uncommitted=args.uncommitted,
        include_untracked=args.untracked,
    )
    if not paths:
        raise CtxpackError(
            f"no files changed for --diff {args.diff!r}"
            + (" (staged)" if args.staged else "")
            + (" (uncommitted)" if args.uncommitted else "")
        )

    repo = repo_root(root)
    wanted = set(relativise(paths, repo, root))

    # Compare in repo-relative terms rather than discovery-relative ones. A
    # single-file root is named by its *basename* in the discovery while
    # `relativise` speaks repo-relative paths, so the two conventions never met
    # and `--diff` on one file matched nothing -- then suggested remedies that
    # could not possibly help. Deriving each candidate's repo-relative path
    # bridges that for file and directory roots alike.
    matched = []
    for candidate in found.files:
        try:
            relative = str(candidate.abs_path.resolve().relative_to(repo))
        except ValueError:
            relative = candidate.path
        if relative in wanted:
            matched.append(candidate)
    before = len(found.files)
    found.files = matched
    if not found.files:
        raise CtxpackError(
            f"--diff {args.diff!r} matched {len(wanted)} changed path(s), but none "
            f"are readable text inside {root}; try --no-gitignore or widen --root"
        )
    print(
        f"ctxpack: --diff {args.diff} -> {len(found.files):,} of {before:,} "
        f"discovered files ({len(wanted):,} changed paths)",
        file=sys.stderr,
    )


def _rerank_with_graph(args: argparse.Namespace, found):
    """Return a ``rerank`` callable that boosts files reachable from entrypoints.

    Returns ``None`` when the user did not ask for it, so the common path pays
    nothing for a feature only some callers want.
    """
    weight = getattr(args, "reach_weight", 0.0)
    if weight <= 0:
        return None

    from .deps import analyse
    from .rank import rank_all

    texts = {c.path: (read_text(c) or "") for c in found.files}

    def rerank(candidates):
        scored = rank_all(candidates)
        analysis = analyse(
            [c.path for c in candidates],
            lambda p: texts.get(p),
        )
        if not analysis.entrypoints:
            print(
                "ctxpack: no entrypoints detected; ranking unchanged "
                "(a repo with no recognisable main/index entry cannot be walked)",
                file=sys.stderr,
            )
            return scored
        reachable = len(analysis.reachable())
        print(
            f"ctxpack: import graph {reachable:,}/{len(candidates):,} files "
            f"reachable from {len(analysis.entrypoints)} entrypoint(s)",
            file=sys.stderr,
        )
        return analysis.boost(scored, weight)

    return rerank


def _apply_config(args: argparse.Namespace) -> None:
    """Merge a discovered config file under the flags the user actually typed.

    Precedence is explicit flag > config file > built-in default. Getting the
    middle case right needs care: argparse fills in a default for every flag the
    user did *not* pass, so after parsing a config value is indistinguishable
    from a default. The config module recovers the set of flags really typed by
    walking ``argv`` against the parser's own option strings, so ``-b 32000``
    typed by hand still beats a config saying 60000.

    ``args._parser`` and ``args._argv`` are attached in :func:`main`.
    """
    from .config import discover_config, load_or_empty, merge, passed_dests

    if getattr(args, "no_config", False):
        return

    parser = getattr(args, "_parser", None)
    argv = getattr(args, "_argv", [])
    if parser is None:  # pragma: no cover - only when called directly
        return

    root = _config_search_root(args)
    path = discover_config(root)
    config = load_or_empty(path)
    if path is None and not config.unknown_keys:
        return

    if config.unknown_keys:
        from .config import unknown_keys_warning

        warning = unknown_keys_warning(config)
        if warning and not args.quiet:
            print(f"ctxpack: {warning}", file=sys.stderr)

    if path is not None and not args.quiet:
        print(f"ctxpack: using settings from {path}", file=sys.stderr)

    defaults = _defaults_for(parser, args)
    # Keys that only ever come from the command line. `passed_dests` walks
    # option strings, so it cannot see that a *positional* was typed -- and the
    # merge would then treat `path` as unspecified and replace it with its
    # default of ".". That silently packed the current working directory instead
    # of the target, which is the worst possible way to be wrong: no error, just
    # a bundle of entirely the wrong repository.
    for key in _CLI_ONLY_KEYS:
        defaults.pop(key, None)
    merged = merge(
        vars(args),
        config,
        defaults,
        passed=passed_dests(parser, argv),
        root=root,
    )
    for key, value in merged.items():
        if key in vars(args):
            setattr(args, key, value)

    if getattr(args, "show_config", False):
        from .config import describe

        print(describe(config), file=sys.stderr)


def _config_search_root(args: argparse.Namespace) -> Path:
    """Where to look for a config file, tolerating the commands that differ.

    ``pack`` and ``count`` take a single path; ``calibrate`` takes zero or more,
    so its ``path`` is a list and cannot go straight into ``Path``.
    """
    raw = getattr(args, "path", None)
    if isinstance(raw, (list, tuple)):
        first = raw[0] if raw else None
        raw = first or "."
    root = getattr(args, "root", None) or raw or "."
    candidate = Path(root).expanduser()
    return candidate if candidate.is_dir() else candidate.parent


def _defaults_for(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> dict:
    """argparse defaults for the *subcommand* that is actually running.

    The top-level parser only knows about the subcommands themselves, so
    collecting defaults from it yields no real settings and the config merge
    silently produces nothing -- which is exactly what happened first time.
    """
    action = next(
        (a for a in parser._actions if isinstance(a, argparse._SubParsersAction)),
        None,
    )
    sub = action.choices.get(getattr(args, "command", ""), None) if action else None
    source = sub if sub is not None else parser
    return {a.dest: a.default for a in source._actions if a.dest != "help"}


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


def _factor_shared(args: argparse.Namespace, result, tokenizer: Tokenizer) -> None:
    """Hoist repeated blocks out of the packed files, in place.

    Opt-in because the measured payoff on real code is small -- under 1% on
    every corpus tried, including a vendored crates registry that does still
    carry repeated licence headers. Modern code consolidated those into
    single-line SPDX identifiers, so the premise this feature rests on is
    largely historical. It is kept because it is free when off, it genuinely
    helps older and template-generated repositories, and "we measured it and it
    saves 0.4%" is more useful to a reader than silence.
    """
    from .boiler import factor, find_blocks

    texts = {doc.path: doc.text for doc in result.documents}
    if len(texts) < 2:
        return

    # Captured before any mutation. ``PackResult.content_tokens`` is a property
    # over the documents, so reading it after rewriting them always yields the
    # post-factoring total and the saving computes as exactly zero.
    before = sum(doc.tokens for doc in result.documents)

    blocks = find_blocks(texts, min_lines=args.shared_min_lines)
    if not blocks:
        return

    factored, used = factor(texts, blocks, min_occurrences=args.shared_min_occurrences)
    if not used:
        return

    from .boiler import assign_ids

    ids = assign_ids(used)
    by_path = {doc.path: doc for doc in result.documents}
    for path, text in factored.items():
        doc = by_path.get(path)
        if doc is not None:
            doc.text = text
            doc.tokens = tokenizer.count(text).tokens

    result.shared_blocks = sorted(
        (
            SharedBlock(
                id=ids[block],
                lines=block.lines,
                occurrences=block.occurrences,
                text=block.text,
            )
            for block in used
        ),
        key=lambda b: b.id,
    )

    after = sum(doc.tokens for doc in result.documents)
    saved = before - after
    added = sum(tokenizer.count(b.text).tokens for b in used)
    net = saved - added
    if net <= 0:
        # Factoring cost more than it saved. Say so and keep the original files.
        for doc in result.documents:
            if doc.path in by_path:
                doc.text = texts[doc.path]
                doc.tokens = tokenizer.count(doc.text).tokens
        result.shared_blocks = []
        result.notes.append(
            f"skipped shared-block factoring: it would have cost "
            f"{-net:,} tokens net"
        )
        return

    result.notes.append(
        f"factored {len(used)} shared block(s), saving ~{net:,} tokens net "
        f"({len(used)} hoisted out of {len(texts)} files)"
    )


def _outline_transform(args: argparse.Namespace):
    """Build a ``transform`` callable that replaces bodies with outlines.

    Applied *before* counting rather than after selection, which is the whole
    point: if the packer only learns a file is cheap once it is already in the
    bundle, the budget that freeing up cannot buy any more files. Transforming
    first means selection sees the outline's real size, so ``--outline`` packs
    more of the repository instead of the same files more cheaply.
    """
    from .outline import OUTLINE_EXTS, estimate_ratio, outline_text

    stats = {"outlined": 0, "left_alone": 0, "ratios": []}

    def transform(candidate, text: str) -> str:
        if candidate.ext not in OUTLINE_EXTS:
            stats["left_alone"] += 1
            return text
        ratio = estimate_ratio(text, path=candidate.path)
        if ratio < args.outline_min_ratio:
            # estimate_ratio is a compression factor: original / outline. At 1.0
            # the outline costs exactly as much as the body, so a low ratio means
            # outlining this file would not pay for itself. Leave it whole.
            stats["left_alone"] += 1
            return text
        stats["outlined"] += 1
        stats["ratios"].append(ratio)
        return outline_text(candidate.path, text)

    transform.stats = stats
    return transform


def _note_outline(result, stats: dict) -> None:
    """Record what outline mode did, including when it did nothing."""
    if not stats["outlined"]:
        result.notes.append(
            f"outline mode: no files were worth summarising "
            f"({stats['left_alone']:,} left as full text)"
        )
        return
    ratios = sorted(stats["ratios"])
    median = ratios[len(ratios) // 2]
    result.notes.append(
        f"outline mode: {stats['outlined']:,} file(s) reduced to structure "
        f"(median {median:.1f}x cheaper than the full body); "
        f"{stats['left_alone']:,} left as full text. "
        "This bundle shows signatures, not implementations."
    )


def _build_bundle(args: argparse.Namespace):
    """Pack once and return ``(text, result, tokenizer, budget)``.

    Split out of :func:`cmd_pack` so ``--watch`` can rebuild through exactly the
    same code path. A watcher that reimplemented the pipeline would drift from
    the one-shot path, and the drift would only show up as a stale bundle.
    """
    tokenizer = _tokenizer(args)
    budget = _budget(args)
    progress = None if args.quiet else _progress
    discovery = _discover(args)

    if getattr(args, "diff", None):
        _restrict_to_diff(args, discovery)

    if not discovery.files:
        raise CtxpackError(
            f"no text files found under {discovery.root}"
            + (" (try --no-default-ignores or --include)" if args.include or args.no_default_ignores else "")
        )

    # Outlining happens during reading, not after selection, so the freed
    # budget buys more files.
    transform = _outline_transform(args) if args.outline else None

    packer = Packer(
        tokenizer,
        budget,
        mode=args.mode,
        progress=progress,
        rerank=_rerank_with_graph(args, discovery),
        transform=transform,
    )
    result = packer.pack(discovery)
    if transform is not None:
        _note_outline(result, transform.stats)
    # Factoring cannot share the transform hook: it needs the whole selected set
    # before it can know what repeats, and an outline has already discarded the
    # bodies it would be looking at.
    if args.factor_shared and not args.outline:
        _factor_shared(args, result, tokenizer)
    elif args.factor_shared:
        # Say so rather than quietly doing nothing, which reads as "no repeated
        # blocks found" when the truth is "you asked for two things that cannot
        # both happen".
        result.notes.append(
            "skipped --factor-shared: --outline already replaced the bodies it "
            "would have looked for shared text in"
        )
    return render(result, args.format), result, tokenizer, budget, discovery


def _emit(
    args: argparse.Namespace,
    text: str,
    result,
    tokenizer,
    budget,
) -> None:
    """Write the bundle wherever the flags say, and report on it."""
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


def cmd_pack(args: argparse.Namespace) -> int:
    if getattr(args, "watch", False):
        return _cmd_watch(args)

    text, result, tokenizer, budget, _ = _build_bundle(args)
    _emit(args, text, result, tokenizer, budget)
    return 0


def _cmd_watch(args: argparse.Namespace) -> int:
    """Repack on change until interrupted.

    The first build runs before the watch starts, so a fresh bundle exists even
    if nothing is ever edited -- otherwise ``--watch`` would block before writing
    anything at all, which looks like a hang.
    """
    from ctxpack.config import passed_dests
    from ctxpack.watch import env_interval, run_watch

    text, result, tokenizer, budget, discovery = _build_bundle(args)
    _emit(args, text, result, tokenizer, budget)

    paths = [c.path for c in discovery.files]
    # Same precedence as everything else in this CLI: a typed flag beats the
    # environment beats the built-in default. `env_interval` is the last two.
    typed = passed_dests(getattr(args, "_parser", None), getattr(args, "_argv", []))
    interval = (
        args.watch_interval
        if "watch_interval" in typed
        else env_interval(default=args.watch_interval)
    )

    if not args.quiet:
        print(
            f"ctxpack: watching {len(paths)} files, rebuilding on change "
            f"(every {interval:g}s, ctrl-C to stop)",
            file=sys.stderr,
        )

    def repack() -> None:
        text, result, tokenizer, budget, _ = _build_bundle(args)
        _emit(args, text, result, tokenizer, budget)

    def on_change(changes: list) -> None:
        if args.quiet:
            return
        shown = ", ".join(str(c) for c in changes[:5])
        more = f" (+{len(changes) - 5} more)" if len(changes) > 5 else ""
        print(f"ctxpack: {shown}{more}", file=sys.stderr)

    def on_error(exc: BaseException) -> None:
        # A half-typed file is the normal state of a repo being worked on. Report
        # it and keep waiting; the previous bundle on disk stays valid.
        print(f"ctxpack: rebuild failed: {exc}", file=sys.stderr)

    try:
        run_watch(
            paths,
            repack,
            root=Path(discovery.root),
            interval=interval,
            settle=args.watch_settle,
            on_change=on_change,
            on_error=on_error,
        )
    except KeyboardInterrupt:
        if not args.quiet:
            print("", file=sys.stderr)
        return 0
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
    if getattr(args, "diff", None):
        _restrict_to_diff(args, discovery)
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
    pack.add_argument(
        "--reach-weight", type=float, default=0.0, metavar="W",
        help="boost files reachable from detected entrypoints, by import-graph "
        "distance (0 disables; try 4)",
    )
    pack.add_argument(
        "--factor-shared", action="store_true",
        help="hoist blocks repeated across files into one shared section "
        "(opt-in: measured at well under 1%% on modern code)",
    )
    pack.add_argument(
        "--shared-min-lines", type=int, default=4, metavar="N",
        help="only factor shared blocks of at least N lines (default: 4)",
    )
    pack.add_argument(
        "--shared-min-occurrences", type=int, default=3, metavar="N",
        help="only factor blocks appearing at least N times (default: 3)",
    )
    pack.add_argument(
        "--outline", action="store_true",
        help="replace file bodies with structural outlines (signatures, no "
        "implementations) so many more files fit the same budget",
    )
    pack.add_argument(
        "--outline-min-ratio", type=float, default=2.0, metavar="R",
        help="only outline a file whose structure is at least R times cheaper "
        "than its body (default: 2.0; 1.0 is break-even)",
    )
    pack.add_argument(
        "--watch", action="store_true",
        help="repack on change until interrupted (writes the first bundle "
        "immediately)",
    )
    pack.add_argument(
        "--watch-interval", type=float, default=0.5, metavar="S",
        help="seconds between polls in --watch (default: 0.5; also settable "
        "via CTXPACK_WATCH_INTERVAL)",
    )
    pack.add_argument(
        "--watch-settle", type=float, default=0.4, metavar="S",
        help="seconds of quiet required before a rebuild in --watch "
        "(default: 0.4)",
    )
    _diff_flags(pack)
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
    _diff_flags(count)
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
    if argv and argv[0] in COMMANDS and len(argv) == 1 and Path(argv[0]).exists():
        # `ctxpack count` inside a checkout that happens to contain a directory
        # named `count` is genuinely ambiguous, and guessing wrong is silent: the
        # subcommand quietly reports on the whole current directory instead of
        # packing the one the user named. Only the bare, argument-less form is
        # rejected -- `ctxpack count --json` is unambiguously the subcommand,
        # because an option cannot be a path.
        print(
            f"ctxpack: '{argv[0]}' is both a subcommand and a path in this "
            f"directory. Use './{argv[0]}' to pack that directory, or add an "
            f"option (e.g. `ctxpack {argv[0]} --json`) to use the subcommand.",
            file=sys.stderr,
        )
        return 2
    if not argv or (argv[0] not in COMMANDS and argv[0] not in TOP_LEVEL_FLAGS):
        argv.insert(0, "pack")

    parser = build_parser()
    args = parser.parse_args(argv)
    # Kept for config merging, which has to know which flags were really typed --
    # argparse cannot tell a default from a user value after the fact.
    args._parser = parser
    args._argv = argv
    try:
        _apply_config(args)
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
