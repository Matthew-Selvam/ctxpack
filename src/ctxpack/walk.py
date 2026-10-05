"""Deciding which files on disk are worth packing.

Two problems, in order of how much they ruin a naive implementation:

1. **Noise.** A checkout of any real project contains more build output,
   vendored code and lockfiles than source. The defaults here cover the
   offenders that appear in nearly every repository.

2. **gitignore.** Users have already written down what they do not care about.
   Re-implementing that badly is worse than ignoring it, so this module
   implements the subset of gitignore(5) that actually shows up in real
   ``.gitignore`` files -- anchoring, trailing slashes, ``*`` / ``**`` / ``?``,
   character classes, escapes, and ``!`` negation with last-match-wins -- and
   says plainly where it deviates.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path

from .errors import CtxpackError

__all__ = [
    "DEFAULT_IGNORES",
    "Candidate",
    "Discovery",
    "IgnoreRule",
    "IgnoreStack",
    "compile_pattern",
    "discover",
]

#: Filenames that suppress noise in a repo even when nothing ignores them.
IGNORE_FILENAMES = (".gitignore", ".ctxpackignore")

#: Directories never worth descending into, regardless of ignore files.
HARD_SKIP_DIRS = frozenset(
    {".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", ".tox"}
)

#: Applied unless ``--no-default-ignores``. Trailing ``/`` means directory-only.
DEFAULT_IGNORES: tuple[str, ...] = (
    # vcs + editor
    ".git/",
    ".hg/",
    ".svn/",
    ".idea/",
    ".DS_Store",
    "Thumbs.db",
    # dependencies
    "node_modules/",
    "bower_components/",
    "vendor/",
    ".venv/",
    "venv/",
    "env/",
    ".bundle/",
    # build output
    "dist/",
    "build/",
    "out/",
    "target/",
    ".next/",
    ".nuxt/",
    ".svelte-kit/",
    ".turbo/",
    ".parcel-cache/",
    "*.egg-info/",
    # caches + coverage
    "__pycache__/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".cache/",
    "htmlcov/",
    ".coverage",
    "coverage/",
    # generated + minified
    "*.min.js",
    "*.min.css",
    "*.min.mjs",
    "*.map",
    "*.pb.go",
    "*_pb2.py",
    "*_pb2_grpc.py",
    "*.generated.*",
    # lockfiles: enormous, and near-zero signal per token
    "*.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "Cargo.lock",
    "poetry.lock",
    "composer.lock",
    "Gemfile.lock",
    # test fixtures and snapshots
    "*.snap",
    "__snapshots__/",
    "testdata/",
    "fixtures/",
    # binaries and media
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.webp",
    "*.ico",
    "*.pdf",
    "*.zip",
    "*.gz",
    "*.tar",
    "*.7z",
    "*.wasm",
    "*.mp4",
    "*.mov",
    "*.mp3",
    "*.wav",
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.otf",
    "*.eot",
    "*.bin",
    "*.db",
    "*.sqlite",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.exe",
)

#: Sniff length for the NUL-byte binary test.
_SNIFF = 8192

#: Rough characters-per-token used when pricing a file we did not read.
_SIZE_CPT = 3.4

#: Most unbounded quantifiers permitted in one compiled pattern.
#:
#: Two unbounded quantifiers in the same path segment, separated only by
#: literals the wildcard can also match, gives Python's engine an exponential
#: number of ways to split the input: `*a*a*...*ab` measured 14ms with two, 1.1s
#: with three and never finished with four, against a 200-character path. That
#: is reachable from ordinary user input -- one .gitignore line -- and it turns
#: `ctxpack .` into a hang.
#:
#: Patterns with more than this are pathological by construction and no real
#: .gitignore uses them, so they degrade to matching nothing. Failing *open* is
#: the safe direction: a pattern ctxpack cannot evaluate leaves files in the
#: bundle that arguably should have been filtered out, rather than silently
#: hiding files that were going to be included.
#:
#: Per-segment counts stay far below this in practice: `*.log`, `**/build`,
#: `*_pb2.py` and every entry in DEFAULT_IGNORES use one.
MAX_UNBOUNDED_QUANTIFIERS = 2

#: Matches a bare unbounded quantifier, i.e. one with nothing between it and
#: its neighbour that would prevent splitting the input.
_UNBOUNDED_RE = re.compile(r"(?:\[\^/\]\*|\.\*)")

#: Used for patterns ctxpack refuses to evaluate. Shared, because it is immutable
#: and compiled once.
_NEVER_MATCHES = re.compile(r"(?!)")


class _TooManyWildcards(Exception):
    """Internal signal: this pattern would backtrack catastrophically."""


@dataclass(frozen=True)
class IgnoreRule:
    """One compiled gitignore pattern."""

    source: str
    regex: re.Pattern[str]
    negated: bool
    dir_only: bool

    def matches(self, rel_path: str) -> bool:
        return self.regex.match(rel_path) is not None


def _translate(pattern: str) -> str:
    """Translate one gitignore glob body into a regex body."""
    out: list[str] = []
    i = 0
    n = len(pattern)
    while i < n:
        char = pattern[i]
        if char == "*":
            # Consume the whole run of stars at once. A run of `*` is
            # semantically identical to a single `*` (`[^/]*[^/]*` matches
            # exactly what `[^/]*` does), but emitting one quantifier per star
            # puts adjacent unbounded quantifiers in the regex, and Python's
            # engine then explores every way of splitting the input between
            # them. A `.gitignore` line of sixteen stars made `discover()` run
            # for minutes on a 240-character path -- exponential in the star
            # count, and reachable from ordinary user input.
            run_end = i
            while run_end < n and pattern[run_end] == "*":
                run_end += 1
            run = run_end - i
            if run >= 2 and pattern[run_end : run_end + 1] == "/":
                # git treats `a/**/b` as matching `a/b` as well as `a/x/b`:
                # `/**/` spans zero or more directory levels. The preceding `/`
                # is already in the output, so this only covers the extra ones.
                out.append("(?:.*/)?")
                i = run_end + 1
                continue
            if run >= 2:
                out.append(".*")
                i = run_end
                continue
            out.append("[^/]*")
            i = run_end
        elif char == "?":
            out.append("[^/]")
            i += 1
        elif char == "[":
            end = i + 1
            if end < n and pattern[end] in "!^":
                end += 1
            if end < n and pattern[end] == "]":
                end += 1
            while end < n and pattern[end] != "]":
                end += 1
            if end >= n:
                # Unterminated class: git treats '[' as a literal.
                out.append(re.escape("["))
                i += 1
            else:
                body = pattern[i + 1 : end]
                if body.startswith("!"):
                    body = "^" + body[1:]
                # Re-escape backslashes. gitignore treats `\` as an escape, but
                # the regex engine treats a lone one as swallowing the closing
                # bracket: `[a\]` compiled to an unterminated character set and
                # raised out of `discover()` as a raw `re.PatternError`, killing
                # `ctxpack .` with a traceback over four characters of
                # .gitignore. Doubling the backslash keeps it a literal.
                body = body.replace("\\", "\\\\")
                # A literal '[' inside a class is legal but warned about, and the
                # warning surfaces to the user of a tool that is supposed to be
                # silent about other people's text. Escaping it is equivalent.
                body = body.replace("[", "\\[")
                candidate = f"[{body}]"
                # Validated rather than trusted. A class body can still be
                # invalid after escaping -- `[z-a]` is a reversed range, which
                # raises "bad character range" -- and any such raise escapes as a
                # traceback rather than a clean message. So compile it here and
                # fall back to git's behaviour for a malformed class, which is to
                # treat the bracket as an ordinary character.
                try:
                    # Compiled under warnings-as-errors on purpose. Several
                    # class bodies are legal-but-suspicious enough that Python
                    # emits a FutureWarning -- "possible nested set" for `[[`,
                    # "possible set difference" for `[a--z]` -- and a tool that
                    # reads other people's repositories should not print those
                    # at the user. A body that warns is treated as malformed,
                    # which is what git does with a bracket it cannot parse.
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", FutureWarning)
                        re.compile(candidate)
                except (re.error, FutureWarning):
                    out.append(re.escape("["))
                    i = i + 1
                else:
                    out.append(candidate)
                    i = end + 1
        elif char == "\\" and i + 1 < n:
            out.append(re.escape(pattern[i + 1]))
            i += 2
        else:
            out.append(re.escape(char))
            i += 1
    return "".join(out)


def compile_pattern(pattern: str) -> IgnoreRule | None:
    """Compile a single gitignore line. Returns ``None`` for comments/blanks.

    Deviations from git, both deliberate and both rare in practice:

    * A ``dir/`` pattern also matches a *file* literally named ``dir``.
    * A leading ``!`` re-includes a path, but -- as in git -- it cannot
      re-include something whose parent directory was already excluded, since
      ctxpack stops descending at the excluded directory.
    """
    line = pattern.rstrip()
    if not line.strip() or line.lstrip().startswith("#"):
        return None

    negated = line.startswith("!")
    if negated:
        line = line[1:]

    dir_only = line.endswith("/")
    anchored = "/" in line.rstrip("/")
    if line.startswith("/"):
        anchored = True
        line = line.lstrip("/")
    line = line.rstrip("/")
    if not line:
        return None

    body = _translate(line)
    prefix = "" if anchored else "(?:.*/)?"
    # Only the body is counted. The `(?:.*/)?` prefix and `(?:/.*)?` suffix both
    # begin with `/`, which `[^/]*` can never match, so each has exactly one way
    # to match and neither can multiply the search. Counting them degraded real
    # patterns -- `src/**/*.ts` and ctxpack's own `*.generated.*`.
    unbounded = len(_UNBOUNDED_RE.findall(body))
    try:
        if unbounded > MAX_UNBOUNDED_QUANTIFIERS:
            raise _TooManyWildcards
        # The trailing group makes a matched directory swallow its contents,
        # which is how git behaves and saves threading dir/file state around.
        regex = re.compile(f"^{prefix}{body}(?:/.*)?$")
    except (_TooManyWildcards, re.error):
        # Belt and braces. Fuzzing kept finding fresh ways to make a translated
        # pattern either invalid or explosive -- unterminated classes, reversed
        # ranges, stray backslashes, adjacent wildcards -- and each escaped
        # `discover()` as a raw exception or a multi-minute hang, killing
        # `ctxpack .` over a single .gitignore line. A pattern ctxpack cannot
        # evaluate safely must match nothing, never crash or stall the run.
        regex = _NEVER_MATCHES
    return IgnoreRule(source=pattern, negated=negated, regex=regex, dir_only=dir_only)


def parse_ignore_text(text: str) -> list[IgnoreRule]:
    rules = []
    for line in text.splitlines():
        rule = compile_pattern(line)
        if rule is not None:
            rules.append(rule)
    return rules


class IgnoreStack:
    """Nested ignore rules, one layer per directory.

    Git gives the deepest matching pattern the last word, including over an
    ancestor's ``!`` re-include, so layers are consulted outermost-first and
    later matches override earlier ones.
    """

    def __init__(self, base_rules: list[IgnoreRule] | None = None) -> None:
        self._layers: list[tuple[str, list[IgnoreRule]]] = [("", base_rules or [])]

    def push(self, rel_dir: str, rules: list[IgnoreRule]) -> None:
        self._layers.append((rel_dir, rules))

    def pop(self) -> None:
        if len(self._layers) > 1:
            self._layers.pop()

    def ignored(self, rel_path: str, is_dir: bool) -> bool:
        verdict = False
        for base, rules in self._layers:
            if base:
                prefix = base + "/"
                if not rel_path.startswith(prefix):
                    continue
                local = rel_path[len(prefix) :]
            else:
                local = rel_path
            for rule in rules:
                if rule.matches(local):
                    verdict = not rule.negated
        return verdict

    def count(self) -> int:
        return sum(len(rules) for _, rules in self._layers)


@dataclass(frozen=True)
class Candidate:
    """A file worth considering. Text is *not* held here; see :func:`read_text`."""

    path: str  # posix, relative to the root
    abs_path: Path
    size: int
    is_binary: bool

    @property
    def ext(self) -> str:
        return os.path.splitext(self.path)[1].lower()

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def depth(self) -> int:
        return self.path.count("/")


@dataclass
class Discovery:
    root: Path
    files: list[Candidate] = field(default_factory=list)
    ignored: int = 0
    directories: int = 0
    total_bytes: int = 0
    too_large: int = 0
    symlinks: int = 0
    binary: int = 0
    ignore_rules: int = 0

    def __len__(self) -> int:
        return len(self.files)

    @property
    def text_files(self) -> int:
        return sum(1 for c in self.files if not c.is_binary)


def is_binary_bytes(chunk: bytes) -> bool:
    """NUL-byte sniff, the same heuristic git uses for "is this text?"."""
    return b"\x00" in chunk


def read_text(candidate: Candidate) -> str | None:
    """Read a candidate as UTF-8, or ``None`` if it is not usable text.

    Resolves through ``abs_path`` rather than the root-relative path, so it is
    correct regardless of the process working directory.
    """
    if candidate.is_binary:
        return None
    try:
        raw = candidate.abs_path.read_bytes()
    except OSError:
        return None
    if is_binary_bytes(raw[:_SNIFF]):
        return None
    return raw.decode("utf-8", "replace")


def _match_glob(pattern: str, rel: str) -> bool:
    """Glob match for ``--include`` / ``--exclude``, where ``*`` may cross ``/``.

    This is intentionally *not* gitignore semantics: CLI filters are a
    different job, and shell-like behaviour is what people expect there.
    """
    pat = pattern.strip().lstrip("./")
    if not pat:
        return True
    if fnmatchcase(rel, pat):
        return True
    if "/" not in pat and fnmatchcase(os.path.basename(rel), pat):
        return True
    return fnmatchcase(rel, "*/" + pat) or fnmatchcase(rel, "**/" + pat)


def discover(
    root: str | os.PathLike[str],
    *,
    exclude: tuple[str, ...] = (),
    include: tuple[str, ...] = (),
    max_size: int | None = None,
    use_default_ignores: bool = True,
    respect_gitignore: bool = True,
) -> Discovery:
    """Walk ``root`` and return the text files worth packing."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise CtxpackError(f"no such path: {root}")

    base_rules = (
        parse_ignore_text("\n".join(DEFAULT_IGNORES)) if use_default_ignores else []
    )
    found = Discovery(root=root_path)

    if root_path.is_file():
        candidate = Candidate(
            path=root_path.name,
            abs_path=root_path,
            size=root_path.stat().st_size,
            is_binary=False,
        )
        found.files = [candidate]
        found.total_bytes = candidate.size
        found.ignore_rules = len(base_rules)
        return found

    stack = IgnoreStack(base_rules)
    if respect_gitignore:
        # The root's own ignore file is not picked up by the recursive scan,
        # because the scan only reads ignore files for directories it descends
        # *into*. Without this, `ctxpack .` at a repo root ignores its
        # .gitignore entirely.
        stack.push("", _read_ignore_file(root_path))
    _scan(root_path, "", stack, found, exclude, include, max_size, respect_gitignore)
    found.files.sort(key=lambda c: c.path)
    found.ignore_rules = stack.count()
    return found


def _scan(
    abs_dir: Path,
    rel_dir: str,
    stack: IgnoreStack,
    found: Discovery,
    exclude: tuple[str, ...],
    include: tuple[str, ...],
    max_size: int | None,
    respect_gitignore: bool,
) -> None:
    found.directories += 1
    try:
        entries = sorted(os.scandir(abs_dir), key=lambda e: e.name)
    except OSError:
        return

    for entry in entries:
        name = entry.name
        rel = f"{rel_dir}/{name}" if rel_dir else name

        try:
            is_symlink = entry.is_symlink()
            # is_dir(follow_symlinks=False) is False for a symlink to a
            # directory, so ask again before deciding what this entry is.
            is_dir = entry.is_dir(follow_symlinks=is_symlink)
        except OSError:
            continue

        if is_dir:
            if name in HARD_SKIP_DIRS:
                found.ignored += 1
                continue
            if stack.ignored(rel, True):
                found.ignored += 1
                continue
            if is_symlink:
                # Descending into symlinked directories invites cycles.
                found.symlinks += 1
                continue

            rules: list[IgnoreRule] = []
            if respect_gitignore:
                rules.extend(_read_ignore_file(Path(entry.path)))
            stack.push(rel, rules)
            _scan(
                Path(entry.path),
                rel,
                stack,
                found,
                exclude,
                include,
                max_size,
                respect_gitignore,
            )
            stack.pop()
            continue

        if stack.ignored(rel, False):
            found.ignored += 1
            continue
        if exclude and any(_match_glob(pat, rel) for pat in exclude):
            found.ignored += 1
            continue
        if include and not any(_match_glob(pat, rel) for pat in include):
            found.ignored += 1
            continue

        try:
            size = entry.stat(follow_symlinks=False).st_size
        except OSError:
            continue
        if max_size is not None and size > max_size:
            found.too_large += 1
            continue

        try:
            with open(entry.path, "rb") as handle:
                head = handle.read(_SNIFF)
        except OSError:
            continue

        binary = is_binary_bytes(head)
        if binary:
            found.binary += 1
            continue

        found.files.append(
            Candidate(path=rel, abs_path=Path(entry.path), size=size, is_binary=False)
        )
        found.total_bytes += size


def _read_ignore_file(directory: Path) -> list[IgnoreRule]:
    rules: list[IgnoreRule] = []
    for name in IGNORE_FILENAMES:
        path = directory / name
        try:
            if path.is_file():
                rules.extend(parse_ignore_text(path.read_text(encoding="utf-8")))
        except OSError:
            continue
    return rules
