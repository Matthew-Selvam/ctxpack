"""Repeated *blocks* of text, factored out into a shared section.

ctxpack already collapses near-duplicate *files* (see :mod:`ctxpack.pack`). This
module handles the other half: repeated *runs of lines*, within one file and
across thousands of them. A 14-line Apache licence header in 200 source files
spends 2,800 lines of context window saying one sentence 200 times. So does the
``__all__`` re-export block every package repeats, the scaffolding table in
every spec file, the identical dict literal in every config. On real
repositories, hoisting these is frequently a double-digit percentage of the
bundle -- more than near-duplicate file collapsing, which is the thing people
notice first because it shows up as "39 identical files dropped".

**A factored bundle is a reading aid, not source.** This is the single most
important thing to know about this module, and it is repeated at the top of
every :func:`render_shared` section because a model that has only ever seen
factored files will happily try to run them.

Three consequences shape everything below:

* Substitution is only ever done with a **comment**, chosen per language. So
  replacing a run of comments is parse-neutral; replacing a run of *code* is
  only allowed where the removal cannot join two statements together.
* Runs are only substituted at **unit boundaries** -- see
  :func:`_replaceable` for the three rules and their rationale.
* Nothing is deleted that cannot be resolved. Every substitution points at a
  block printed once in the shared section, with an id, its line count, and how
  many times it occurred.

The known false negatives are deliberate, and they are expensive:

* A run touching a triple-quoted string is never substituted. That rules out
  the module docstring, which is one of the most repeated blocks in any
  package -- and rightly, because deleting a docstring deletes
  ``__doc__``.
* A run embedded in the middle of a function body is never substituted, even
  when it is textually identical everywhere it appears. A duplicated five-line
  SQL query block inside a handler is left inline.
* Runs shorter than :data:`DEFAULT_MIN_LINES`, or mostly punctuation, are not
  even candidates. ``}``, ``pass``, ``return None`` and closing brackets are
  everywhere; hoisting them saves a handful of tokens and destroys the shape of
  the code.

Cost of being wrong in the other direction is much higher than the cost of
missing a block: a broken-looking function costs the reader far more than a
duplicated licence header does.

The false negative worth knowing about, because it is the surprising one:
:func:`find_blocks` reports the *maximal* run that every occurrence agrees on,
and the conservatism rules are applied to that whole run. So a run that starts
at column zero of one file and runs into a function body that every file also
happens to share is rejected *entirely*, even though the inner few lines would
have been safe on their own. Trimming rejected runs and retrying would recover
those, at the cost of a second search and a second set of ways to be wrong;
conservatism wins.

Matching strategy, in one paragraph, because it is the interesting part: every
window of :data:`DEFAULT_MIN_LINES` consecutive lines is hashed, identical
hashes are grouped, and each group is then **verified by comparing the actual
lines** -- hashes collide, and a 128-bit digest over a few million windows is
not an assumption worth making -- before being greedily extended one line at a
time. That is O(total lines) with a small constant, where pairwise file
comparison is O(files^2 * lines^2) and a repository with two thousand files
never finishes. Windows are hashed on ``str.rstrip()``-normalised lines so that
a reformatted file still matches its siblings, but the extracted text keeps the
original indentation and trailing whitespace, because the block is meant to be
read as real source.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

from .errors import CtxpackError
from .tokens import estimate_tokens

__all__ = [
    "DEFAULT_MAX_LINES",
    "DEFAULT_MIN_FILES",
    "DEFAULT_MIN_LINES",
    "DEFAULT_MIN_OCCURRENCES",
    "MARKER_STYLES",
    "MIN_LINE_CHARS",
    "MIN_SUBSTANTIVE_RATIO",
    "SHARED_SECTION_TITLE",
    "Block",
    "SavingsReport",
    "assign_ids",
    "factor",
    "find_blocks",
    "marker_for",
    "render_shared",
    "savings",
    "savings_report",
]

# --------------------------------------------------------------------------
# Tuning constants
# --------------------------------------------------------------------------

#: Shortest run considered. Four is where "boilerplate" and "two lines that
#: happen to match" part company: below this, hash groups fill with ``}``,
#: ``pass`` and ``return None``, and every file in a repository matches every
#: other file.
DEFAULT_MIN_LINES = 4

#: Longest run considered. A 200-line match is a vendored file, a lockfile or a
#: generated client, not boilerplate. Hoisting one turns the only copy into a
#: pointer to a pointer, and a reader cannot use what they must first
#: reassemble.
DEFAULT_MAX_LINES = 200

#: Distinct files a run must appear in to count as shared. One occurrence is a
#: local convention -- a project uses tabs, or one file names things oddly --
#: and hoisting a convention to a shared block teaches the reader nothing they
#: could not see in the file itself.
DEFAULT_MIN_FILES = 2

#: Occurrences before a block is worth substituting. The marker is not free, so
#: factoring a block seen twice or three times is a real calculation rather than
#: an automatic win, and ``min_occurrences=3`` is where it reliably comes out
#: ahead. Overridable, because a caller that knows its bundle format may price
#: markers differently.
DEFAULT_MIN_OCCURRENCES = 3

#: Shortest line (after stripping) that carries information. Four characters --
#: ``pass``, ``end``, ``};``, ``None`` -- is structure, not content, and
#: structure repeats by coincidence.
MIN_LINE_CHARS = 5

#: Fraction of a window's lines that must be substantive. Half, not most: a
#: licence header or a re-export block is mostly comments but not *entirely*,
#: and requiring the whole window to be prose would reject every block with a
#: blank separator in it.
MIN_SUBSTANTIVE_RATIO = 0.5

#: Digest width for window keys. 128 bits over the few million windows a large
#: monorepo produces puts the birthday probability below 1e-20 -- which is
#: insurance, not a correctness argument: every candidate group is verified by
#: comparing real lines before anything is extracted.
_DIGEST_BYTES = 16

#: Window width used to *find* anchor positions again inside :func:`factor`.
#: Two lines of substantive source is narrow enough that candidate lists stay
#: short, and a block shorter than this falls back to a linear scan.
_ANCHOR_WIDTH = 2

# --------------------------------------------------------------------------
# Substitution syntax
# --------------------------------------------------------------------------

#: Heading that markers point at. Markers quote this string verbatim, so the two
#: must not be changed independently.
SHARED_SECTION_TITLE = "Shared blocks"

#: Comment openers by file extension, as ``(open, close)``. The close is empty
#: for line comments and ``" -->"`` for the XML/HTML form, which cannot be
#: closed by nothing.
#:
#: Extensions are matched case-insensitively against the final ``.suffix``; a
#: path with no recognised suffix falls back to ``#``, which is wrong for C and
#: merely ugly for JavaScript. That is the deliberate default: ``#`` is the
#: comment form for the largest share of what ctxpack packs, and a bundle that
#: has already hoisted code into a shared section is a reading aid where one
#: stray ``#`` in a ``.js`` file costs a few tokens of confusion.
MARKER_STYLES: dict[str, tuple[str, str]] = {
    "--": ("--", ""),
    "//": ("//", ""),
    "#": ("#", ""),
    "<!--": ("<!--", " -->"),
    ";": (";", ""),
    "%": ("%", ""),
    # hash-family: python, shell, ruby, perl, yaml, toml, ini, make, terraform
    ".bash": ("#", ""),
    ".cfg": ("#", ""),
    ".cmake": ("#", ""),
    ".dockerfile": ("#", ""),
    ".ex": ("#", ""),
    ".exs": ("#", ""),
    ".gitignore": ("#", ""),
    ".ini": ("#", ""),
    ".jl": ("#", ""),
    ".mk": ("#", ""),
    ".nim": ("#", ""),
    ".pl": ("#", ""),
    ".py": ("#", ""),
    ".pyi": ("#", ""),
    ".r": ("#", ""),
    ".rb": ("#", ""),
    ".sh": ("#", ""),
    ".tf": ("#", ""),
    ".toml": ("#", ""),
    ".yaml": ("#", ""),
    ".yml": ("#", ""),
    ".zsh": ("#", ""),
    # slash-family: the C/JS/Rust family, java, kotlin, swift, scala, go, proto
    ".c": ("//", ""),
    ".cc": ("//", ""),
    ".cpp": ("//", ""),
    ".cs": ("//", ""),
    ".cxx": ("//", ""),
    ".dart": ("//", ""),
    ".go": ("//", ""),
    ".groovy": ("//", ""),
    ".h": ("//", ""),
    ".hpp": ("//", ""),
    ".java": ("//", ""),
    ".js": ("//", ""),
    ".jsx": ("//", ""),
    ".kt": ("//", ""),
    ".kts": ("//", ""),
    ".m": ("//", ""),
    ".mm": ("//", ""),
    ".mjs": ("//", ""),
    ".mts": ("//", ""),
    ".proto": ("//", ""),
    ".rs": ("//", ""),
    ".scala": ("//", ""),
    ".swift": ("//", ""),
    ".ts": ("//", ""),
    ".tsx": ("//", ""),
    ".cjs": ("//", ""),
    # dash-family: sql, lua, haskell, verilog, ada
    ".adb": ("--", ""),
    ".ads": ("--", ""),
    ".hs": ("--", ""),
    ".lua": ("--", ""),
    ".sql": ("--", ""),
    ".v": ("--", ""),
    ".vh": ("--", ""),
    # lisp family
    ".clj": (";", ""),
    ".cljs": (";", ""),
    ".el": (";", ""),
    ".lisp": (";", ""),
    ".scm": (";", ""),
    # latex and matlab
    ".tex": ("%", ""),
    ".sty": ("%", ""),
    ".cls": ("%", ""),
    ".matlab": ("%", ""),
    # markup, where a line comment does not exist
    ".htm": ("<!--", " -->"),
    ".html": ("<!--", " -->"),
    ".md": ("<!--", " -->"),
    ".mdx": ("<!--", " -->"),
    ".svg": ("<!--", " -->"),
    ".vue": ("<!--", " -->"),
    ".xml": ("<!--", " -->"),
}

#: Comment prefixes that make a line an "anchor": something a run may be lifted
#: out from behind or in front of. ``*`` is here for the interior of ``/** */``
#: blocks, ``;`` for lisp, ``%`` for latex.
_ANCHOR_PREFIXES = ("#", "//", "--", ";", "%", "*", "<!--", "/*")

#: A decorator line -- ``@app.route``, ``@pytest.mark.parametrize``,
#: ``@Override`` -- is a unit boundary like a blank line: it cannot be split.
_ANCHOR_RE = re.compile(r"^@[A-Za-z_]")

#: A line that carries no content. Used with :data:`MIN_LINE_CHARS`.
_WORD_RE = re.compile(r"\w")

#: Triple-quoted delimiters. A run touching one is never substituted: it is
#: either a docstring (deleting it deletes ``__doc__``) or the middle of a
#: string literal (deleting it changes the value). This is the single most
#: expensive false negative in the module and it is worth it -- see the module
#: docstring.
_TRIPLE_RE = re.compile(r'("""|\'\'\')')

#: ``id(norm) -> (norm, (depths, in_string))``. Keyed by identity and holding a
#: reference to the list, so the identity check makes a stale hit impossible;
#: see :func:`_bracket_depths` for why this has to exist.
_DEPTH_CACHE: dict[int, tuple[list[str], tuple[list[int], list[bool]]]] = {}
_DEPTH_CACHE_MAX = 64


def _split_lines(text: str) -> tuple[list[str], str]:
    """Split into logical lines plus the line ending to reassemble with.

    Deliberately **not** ``str.splitlines()``: that also breaks on form feed,
    ``\\x0b``, ``\\x1c`` and U+2028, all of which are ordinary *characters* in a
    source file. A file containing a form feed would be cut in half here and
    reassembled with a newline where the form feed used to be, silently
    corrupting it. Splitting on ``"\\n"`` after normalising CRLF is boring and
    reversible.

    Returns ``([], "\\n")`` for the empty string, which is why every caller
    treats "no lines" and "one empty line" the same way: neither can hold a
    substantive block.
    """
    if "\r\n" in text:
        return text.replace("\r\n", "\n").split("\n"), "\r\n"
    return text.split("\n"), "\n"


def _body(lines: list[str]) -> list[str]:
    """Drop the phantom final element ``split`` leaves for a newline-terminated file.

    ``"a\\nb\\n".split("\\n")`` is ``["a", "b", ""]``. That empty string is not a
    line -- it exists only so a join can restore the trailing newline -- but block
    discovery has no way to know that, so a block ending at end-of-file absorbs
    it. The consequences were real: the rewrite then consumed the phantom as part
    of the run and dropped the file's terminating newline, and ``Block.lines``
    counted a line the rendered block never showed.

    Only the last element can be phantom, so indices are unaffected and a
    position found against the trimmed list still addresses the same line in the
    untrimmed one used for reassembly.
    """
    if lines and lines[-1] == "":
        return lines[:-1]
    return lines


def _is_substantive(line: str) -> bool:
    """True if a single line is worth being part of a shared block.

    Three ways to fail, in increasing order of how often they save us: blank,
    shorter than :data:`MIN_LINE_CHARS`, or containing no word characters at
    all. The last test is what kills ``}}},`` and ``);`` -- windows of pure
    punctuation hash-match across every file in the repository.
    """
    stripped = line.strip()
    if len(stripped) < MIN_LINE_CHARS:
        return False
    return _WORD_RE.search(stripped) is not None


def _window_key(norm: list[str], start: int, width: int) -> bytes:
    """Digest of ``width`` consecutive normalised lines.

    Hashing the joined text rather than a tuple of lines keeps the hot loop to
    one call per window regardless of line length.
    """
    return hashlib.blake2b(
        "\n".join(norm[start : start + width]).encode("utf-8", "surrogatepass"),
        digest_size=_DIGEST_BYTES,
    ).digest()


def _is_anchor_line(line: str) -> bool:
    """True if ``line`` is a blank, comment or decorator line."""
    stripped = line.strip()
    if not stripped:
        return True
    return bool(
        stripped.startswith(_ANCHOR_PREFIXES) or _ANCHOR_RE.match(stripped)
    )


def _ends_a_unit(line: str) -> bool:
    """True if ``line`` is a complete top-level unit rather than a continuation.

    Deliberately strict about indentation: an indented ``}`` closes a nested
    block, so whatever follows it is still inside the enclosing one, and an
    indented ``foo(`` is unambiguously mid-statement.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith(_ANCHOR_PREFIXES):
        return True
    if line[:1] in (" ", "\t"):
        return False
    # Column zero, so the run has a chance of being complete.
    if stripped.startswith(("{", "[")):
        return stripped[-1] in ("}", "]")
    return stripped.endswith((":", "}", "]", ";"))


def _bracket_depths(norm: list[str]) -> tuple[list[int], list[bool]]:
    """Bracket depth entering each line, and whether each line is string content.

    Returns ``(depths, in_string)`` with ``len(norm) + 1`` entries, so
    ``depths[len(norm)]`` is the depth after the last line -- which is what a
    caller checking a run that ends at end-of-file needs.

    ``in_string[i]`` means *the text of line i is inside a triple-quoted
    string*, which is deliberately not the same as "a string was open when the
    line began". The opening and closing delimiter lines themselves report
    ``False``; rule 3 of :func:`_replaceable` already rejects a run whose own
    first or last line carries a delimiter, so the flag only needs to answer the
    question rule 3 cannot: is a run with its delimiters further away sitting
    in string content?

    Memoised per file. :func:`_replaceable` runs once per candidate occurrence,
    and recomputing the map each time would make the whole search quadratic in
    line count -- the mistake ``_locate`` was rewritten to avoid.
    """
    cached = _DEPTH_CACHE.get(id(norm))
    if cached is not None and cached[0] is norm:
        return cached[1]

    depths: list[int] = []
    in_string: list[bool] = []
    depth = 0
    open_delim: str | None = None
    for line in norm:
        depths.append(depth)
        code = "" if _is_anchor_line(line) else line
        inside = open_delim is not None
        if open_delim is None:
            for quote in ('"""', "\'\'\'"):
                if quote in code:
                    open_delim = quote
                    code = code.replace(quote, "", 1)
                    break
        if open_delim is None:
            depth += sum(code.count(c) for c in "([{")
            depth -= sum(code.count(c) for c in ")]}")
            if depth < 0:
                depth = 0
        elif open_delim in code:
            code = code.replace(open_delim, "", 1)
            open_delim = None
            inside = False  # the closing delimiter is not itself content
        in_string.append(inside)

    depths.append(depth)
    in_string.append(open_delim is not None)
    result = (depths, in_string)
    if len(_DEPTH_CACHE) >= _DEPTH_CACHE_MAX:
        _DEPTH_CACHE.clear()
    _DEPTH_CACHE[id(norm)] = (norm, result)
    return result


def _replaceable(norm: list[str], start: int, length: int) -> bool:
    """The conservatism rules. See the module docstring for why each exists.

    A run may be replaced only when all of these hold:

    1. The preceding line (or the start of the file) is blank, a comment or a
       decorator. A run that starts right after code is in the middle of a
       statement -- a call, a list literal, an indented body -- and lifting it
       out leaves the surrounding code unparseable.
    2. The run ends at a unit boundary: either the following line (or end of
       file) is blank/comment/decorator, or the run's last line is already a
       comment. The second half is what lets a licence header be hoisted when
       ``import os`` follows it directly with no blank line in between -- the
       marker is itself a comment, so nothing that followed the comment can
       have been part of it.
    3. The run touches no triple-quoted delimiter on its own first or last line.
    4. **The run is not inside anything.** Bracket depth is zero at both ends,
       and the run is not within a triple-quoted string.

    Rule 4 was the one that was missing, and it was the only one that mattered.
    Rules 1-3 all read the run *in isolation*, so a run cut out of the middle of
    an unclosed bracket passed every one of them: a comment inside the bracket
    satisfies rule 1, and a closing ``}`` on the last line satisfies rule 2.
    With a file-specific opener -- which is what real modules look like, one
    ``client_a_settings = merge(...)`` per service -- maximal-run extension
    cannot absorb the opener either, so the run genuinely looked standalone:

        client_a_settings = {
        # a note
        "k1": 1,
        }
        a_x = 1

    Factoring that produced ``client_a_settings = {`` followed by a marker, and
    the file stopped parsing. Valid Python in, invalid Python out, silently, in
    every affected file.

    Both counts are naive on purpose. A bracket inside a string is miscounted,
    which errs towards *rejecting* a run -- the safe direction, since the cost of
    a missed hoist is a few saved tokens and the cost of a wrong one is broken
    source.
    """
    if start > 0 and not _is_anchor_line(norm[start - 1]):
        return False
    end = start + length
    if not _ends_at_boundary(norm, start, end):
        return False
    first, last = norm[start], norm[end - 1]
    if _TRIPLE_RE.search(first) or _TRIPLE_RE.search(last):
        return False
    depths, in_string = _bracket_depths(norm)
    # Zero before the run means it does not begin inside an open bracket; zero
    # after means it does not leave one open. Both maps carry len(norm) + 1
    # entries, so `end` -- at most len(norm) -- is always a valid index.
    if depths[start] != 0 or depths[end] != 0:
        return False
    # And it must not sit inside a triple-quoted string whose delimiters are
    # more than one line away, which rule 3 above cannot see.
    return not (in_string[start] or in_string[end - 1])


def _ends_at_boundary(norm: list[str], start: int, end: int) -> bool:
    """True if a run ending at ``end`` cannot be continued by the next line.

    Two independent reasons to say yes:

    * The line after the run (or end of file) is blank, comment or decorator.
    * The run's last line is a comment, so the run was already comment-only
      text where it ended.

    A blank last line deliberately does *not* qualify on its own. A run ending
    ``}\\n\\n`` followed by code is the case that made the original rule unsafe:
    the ``}`` closes a block, and the blank line after it says nothing about
    whether the run was balanced. ``_ends_a_unit`` covers that case instead,
    and only for a run that is entirely at column zero.
    """
    if end >= len(norm) or _is_anchor_line(norm[end]):
        return True
    if norm[end - 1].strip().startswith(_ANCHOR_PREFIXES):
        return True
    # No shortcut left, so the run has to be provably self-contained.
    return all(
        not line.strip() or line[:1] not in (" ", "\t") for line in norm[start:end]
    ) and _ends_a_unit(norm[end - 1])


@dataclass(frozen=True)
class Block:
    """A run of lines that appears more than once, worth printing only once."""

    #: The block verbatim, lines joined with ``"\\n"`` and **no** trailing
    #: newline. Indentation and trailing whitespace are preserved: the block is
    #: shown to a reader as source. When two occurrences differ only in
    #: trailing whitespace (they match, because matching is rstrip-normalised)
    #: the copy kept is the one from the lexicographically first ``(file,
    #: line)`` position.
    text: str

    #: Number of lines in :attr:`text`.
    lines: int

    #: Distinct files containing the block, sorted.
    files: tuple[str, ...]

    #: Total occurrences, counting repeats *within* a single file. A block
    #: repeated four times in one file is four copies of context window.
    occurrences: int

    #: Cost of one copy. Always the dependency-free estimator from
    #: :mod:`ctxpack.tokens`, because :func:`find_blocks` takes no tokenizer and
    #: must produce the same blocks -- and therefore the same ids -- whatever
    #: counting method the caller ends up using. :func:`savings` re-measures
    #: with the real tokenizer.
    tokens: int

    @property
    def redundant(self) -> int:
        """Tokens spent beyond the first copy, ignoring the cost of markers.

        A projection, and deliberately labelled as one: replacing a block also
        costs a marker line in each file. It is the right number for *ordering*
        blocks, which is all it is used for.
        """
        return max(0, self.occurrences - 1) * self.tokens

    @property
    def in_one_file_only(self) -> bool:
        return len(self.files) < 2


@dataclass(frozen=True)
class _Candidate:
    """A verified run, still tied to the positions it was found at."""

    #: Matching identity: rstrip-normalised lines. Two occurrences differing
    #: only in trailing whitespace are the same candidate, which is why this is
    #: not :attr:`Block.text`.
    content: tuple[str, ...]

    #: Canonical ``(file, start)`` position, for extracting original text.
    origin: tuple[str, int]

    #: Every ``(file, start)`` the run occurs at, sorted.
    spots: tuple[tuple[str, int], ...]


class _ClaimBoard:
    """Tracks which line ranges of which files are already spoken for.

    Overlapping blocks are the failure mode that makes factored output look
    broken: hoisting the licence header and then hoisting the last nine lines
    of it *again* nests one marker inside another and the reference no longer
    resolves. Candidates are offered best-first and any part already claimed is
    dropped; a candidate with no unclaimed occurrence is discarded entirely.
    """

    __slots__ = ("_taken",)

    def __init__(self) -> None:
        self._taken: dict[str, list[tuple[int, int]]] = {}

    def free(
        self, spots: Sequence[tuple[str, int]], length: int
    ) -> list[tuple[str, int]]:
        """Spots that overlap nothing already claimed, and not each other.

        The self-overlap test matters for periodic content -- a run of
        ``    pass`` or a repeated ``value: 1`` table can match at overlapping
        offsets -- where two matches of the same block are the same block, not
        two of them.
        """
        out: list[tuple[str, int]] = []
        last_end: dict[str, int] = {}
        for path, start in spots:
            if start < last_end.get(path, -1):
                continue
            taken = self._taken.get(path)
            if taken and any(
                start < end and begin < start + length for begin, end in taken
            ):
                continue
            out.append((path, start))
            last_end[path] = start + length
        return out

    def take(self, spots: Sequence[tuple[str, int]], length: int) -> None:
        for path, start in spots:
            self._taken.setdefault(path, []).append((start, start + length))


def find_blocks(
    texts: dict[str, str],
    *,
    min_lines: int = DEFAULT_MIN_LINES,
    max_lines: int = DEFAULT_MAX_LINES,
    min_files: int = DEFAULT_MIN_FILES,
) -> list[Block]:
    """Find repeated runs of lines across ``{path: text}``.

    Hash every window of ``min_lines`` lines, group identical hashes, verify
    each group by comparing the real lines, then greedily extend one line at a
    time while every occurrence still agrees. Windows that fail the
    :data:`MIN_SUBSTANTIVE_RATIO` test never reach the hash table, which is
    both the quality filter and most of the speed.

    Runs are resolved against each other so that returned blocks never overlap:
    the longest, most-repeated run wins its lines and anything sitting inside
    it is dropped, so a licence header comes back as one block and not as
    fourteen nested candidates.

    Returned in descending order of :attr:`Block.redundant`, ties broken on
    text. Two runs over the same input return identical lists.
    """
    if min_lines < 1:
        raise CtxpackError(f"min_lines must be at least 1, got {min_lines}")
    if max_lines < min_lines:
        raise CtxpackError(
            f"max_lines ({max_lines}) must not be smaller than min_lines "
            f"({min_lines})"
        )
    if min_files < 1:
        raise CtxpackError(f"min_files must be at least 1, got {min_files}")
    if len(texts) < min_files:
        # Nothing can be shared across fewer files than the threshold. The
        # single-file case matters more than it looks: it is how a CLI call
        # with one file behaves, and the early exit keeps it from reading.
        return []

    paths = sorted(texts)
    raw: dict[str, list[str]] = {}
    norm: dict[str, list[str]] = {}
    for path in paths:
        lines, _ending = _split_lines(texts[path])
        # Discovery, extension and anchoring all work on the trimmed view, or a
        # block at end-of-file absorbs the phantom empty line and then only
        # matches in files where a *real* blank line happens to follow it. That
        # made occurrence counts depend on where a block sat in the file.
        body = _body(lines)
        raw[path] = lines
        norm[path] = [line.rstrip() for line in body]

    groups: dict[bytes, list[tuple[str, int]]] = {}
    width = min_lines
    floor = max(1, int(MIN_SUBSTANTIVE_RATIO * width))
    for path in paths:
        lines = norm[path]
        total = len(lines)
        if total < width:
            continue
        # Prefix sums of the substantive count turn the per-window quality
        # test into two list reads. Doing it per window instead is the
        # difference between linear and quadratic on a large repository.
        prefix = _substantive_prefix(lines)
        for start in range(total - width + 1):
            if prefix[start + width] - prefix[start] < floor:
                continue
            groups.setdefault(_window_key(lines, start, width), []).append((path, start))

    # Merge across seeds: different windows can grow into the same block, and
    # the same window can appear many times in one long repeated section.
    merged: dict[tuple[str, ...], list[tuple[str, int]]] = {}
    for spots in groups.values():
        verified: dict[tuple[str, ...], list[tuple[str, int]]] = {}
        for path, start in spots:
            window = tuple(norm[path][start : start + width])
            verified.setdefault(window, []).append((path, start))
        for members in verified.values():
            if len({path for path, _ in members}) < min_files:
                continue
            length = _extend(members, norm, width, max_lines)
            # Re-read the grown run from the file rather than slicing the seed
            # tuple: the seed is exactly ``width`` lines long, so slicing it
            # cannot possibly produce the extension.
            path, start = members[0]
            content = tuple(norm[path][start : start + length])
            merged.setdefault(content, []).extend(members)

    candidates = [
        _Candidate(
            content=content,
            origin=min(spots),
            spots=tuple(sorted(spots)),
        )
        for content, spots in merged.items()
    ]
    # Most valuable first, and fully determined by content so the order cannot
    # depend on dictionary iteration or hash seed.
    candidates.sort(key=lambda c: (-(len(c.content) * len(c.spots)), c.content))

    board = _ClaimBoard()
    blocks: list[Block] = []
    for candidate in candidates:
        length = len(candidate.content)
        spots = board.free(candidate.spots, length)
        if not spots:
            continue
        board.take(spots, length)
        path, start = candidate.origin
        text = "\n".join(raw[path][start : start + length])
        files = tuple(sorted({p for p, _ in spots}))
        blocks.append(
            Block(
                text=text,
                lines=length,
                files=files,
                occurrences=len(spots),
                tokens=estimate_tokens(text),
            )
        )
    return blocks


def _substantive_prefix(lines: list[str]) -> list[int]:
    """``prefix[i]`` is the number of substantive lines in ``lines[:i]``."""
    prefix = [0] * (len(lines) + 1)
    total = 0
    for index, line in enumerate(lines):
        if _is_substantive(line):
            total += 1
        prefix[index + 1] = total
    return prefix


def _extend(
    members: list[tuple[str, int]],
    norm: dict[str, list[str]],
    length: int,
    cap: int,
) -> int:
    """Grow a verified run while every occurrence still agrees, one line at a time.

    Greedy rather than binary search on purpose: the common case is a run that
    stops at the first line which differs, and a binary search would have to
    verify the whole member list per probe anyway. Terminates at ``cap`` lines
    or at the end of the shortest file.
    """
    while length < cap:
        reference: str | None = None
        for path, start in members:
            lines = norm[path]
            position = start + length
            if position >= len(lines):
                return length
            if reference is None:
                reference = lines[position]
            elif lines[position] != reference:
                return length
        length += 1
    return length


def _locate(
    texts: dict[str, str],
    blocks: list[Block],
) -> dict[str, list[tuple[str, int]]]:
    """Find where each block actually is, in the text given to :func:`factor`.

    :class:`Block` deliberately does not carry positions -- they would go stale
    the moment anything rewrites a file, and a stale position produces a marker
    in the wrong place, which is worse than no marker. So the search is redone
    from scratch, anchored on the first :data:`_ANCHOR_WIDTH` lines to keep the
    candidate lists short, then verified by comparing every line.
    """
    prepared: dict[str, list[str]] = {}
    anchors: dict[bytes, list[tuple[str, int]]] = {}
    for path in sorted(texts):
        lines, _ending = _split_lines(texts[path])
        norm = [line.rstrip() for line in _body(lines)]
        prepared[path] = norm
        for start in range(len(norm) - _ANCHOR_WIDTH + 1):
            anchors.setdefault(_window_key(norm, start, _ANCHOR_WIDTH), []).append(
                (path, start)
            )

    found: dict[str, list[tuple[str, int]]] = {}
    for block in blocks:
        content = tuple(line.rstrip() for line in block.text.split("\n"))
        width = len(content)
        spots: set[tuple[str, int]] = set()
        if width >= _ANCHOR_WIDTH:
            key = _window_key(list(content), 0, _ANCHOR_WIDTH)
            # Iterate the anchor hits, not the files. `anchors` is global, so
            # scanning every file and then re-walking the whole hit list for
            # each one made this O(files x total_hits) -- 15.2s on 800 files
            # where 0.38s suffices, and every second of it was wasted on
            # out-of-range slices that `norm[start:start+width]` silently
            # clamps. On a repo of generated files that is minutes of stall for
            # a result that is then discarded wholesale.
            for hit_path, start in anchors.get(key, ()):
                norm = prepared[hit_path]
                if start + width > len(norm):
                    continue
                if tuple(norm[start : start + width]) == content:
                    spots.add((hit_path, start))
        elif width == 1:
            # A one-line block has no two-line anchor; the linear scan is only
            # reached when a caller deliberately set min_lines=1.
            for path in sorted(prepared):
                norm = prepared[path]
                for start, line in enumerate(norm):
                    if line == content[0]:
                        spots.add((path, start))
        found[block.text] = sorted(spots)
    return found


def _dedupe(blocks: Sequence[Block]) -> list[Block]:
    """Collapse blocks with identical text, keeping the best-attested one.

    Two callers can hand us the same block twice (finding it, then finding it
    again after a rewrite). Assigning ids to a list with duplicates would give
    the second copy a different id for the same text, and the rendered section
    would list it twice.
    """
    best: dict[str, Block] = {}
    for block in blocks:
        current = best.get(block.text)
        if current is None or (block.occurrences, block.tokens) > (
            current.occurrences,
            current.tokens,
        ):
            best[block.text] = block
    return list(best.values())


def assign_ids(blocks: Sequence[Block]) -> dict[Block, str]:
    """Map blocks to their ``B1``, ``B2``, ... identifiers.

    Ids follow projected savings, descending, with ties broken on the block text.
    That is a total order over distinct texts, so the numbering is reproducible
    across processes -- unlike an order that leaned on dictionary insertion or
    on ``hash()``, which is salted per interpreter by default.

    Both :func:`factor` and :func:`render_shared` call this, and both are given
    the *same* list -- the blocks :func:`factor` actually used -- which is why a
    marker in a file always names an id that exists in the rendered section.
    """
    ordered = sorted(blocks, key=lambda b: (-b.redundant, b.text))
    return {block: f"B{index}" for index, block in enumerate(ordered, start=1)}


def comment_style(path: str) -> tuple[str, str]:
    """The ``(open, close)`` comment pair to use for ``path``."""
    dot = path.rfind(".")
    if dot < 0:
        return MARKER_STYLES["#"]
    return MARKER_STYLES.get(path[dot:].lower(), MARKER_STYLES["#"])


def marker_for(path: str, block_id: str, lines: int) -> str:
    """The one-line reference left in place of a hoisted block.

    Names the id and the line count so a reader who wants the original does not
    have to go looking, and quotes :data:`SHARED_SECTION_TITLE` so the marker
    and the section agree even if one of them is read in isolation.
    """
    opener, closer = comment_style(path)
    body = (
        f'[ctxpack: shared block {block_id} — {lines} lines, '
        f'see "{SHARED_SECTION_TITLE}"]'
    )
    return f"{opener} {body}{closer}"


def factor(
    texts: dict[str, str],
    blocks: list[Block],
    *,
    min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
) -> tuple[dict[str, str], list[Block]]:
    """Replace repeated blocks with markers, and say which ones were used.

    Returns the rewritten texts and the blocks that were actually substituted.
    Files with nothing replaced come back byte-identical, keys and all, so a
    caller can diff the two mappings to see exactly what moved.

    The returned blocks are *adjusted* copies: ``occurrences`` and ``files``
    count only the substitutions that passed :func:`_replaceable`, which is
    almost always fewer than :func:`find_blocks` found. Those adjusted numbers
    are the ones to price and to render, and :func:`assign_ids` numbers them in
    the same order :func:`render_shared` will, so markers and section agree.

    Factoring already-factored text is a no-op for the same block set, because
    the full block no longer exists anywhere: the marker is shorter than
    :data:`DEFAULT_MIN_LINES` and its lines do not match. With a *different*
    block set it is not guaranteed to be -- a marker can become part of a fresh
    match window -- so re-running on output is a caller decision, not a
    guarantee.
    """
    if min_occurrences < 2:
        raise CtxpackError(
            f"min_occurrences must be at least 2, got {min_occurrences}: the "
            "marker is not free, so hoisting a block seen once or twice can "
            "cost more tokens than it saves"
        )
    usable = [b for b in _dedupe(blocks) if b.occurrences >= min_occurrences]
    if not usable or not texts:
        return dict(texts), []
    usable.sort(key=lambda b: (-(b.lines * b.occurrences), b.text))

    positions = _locate(texts, usable)
    prepared: dict[str, tuple[list[str], list[str]]] = {}
    for path in sorted(texts):
        lines, _ending = _split_lines(texts[path])
        body = _body(lines)
        prepared[path] = (body, [line.rstrip() for line in body])

    board = _ClaimBoard()
    plan: dict[str, list[tuple[int, int, Block]]] = {}
    used: list[Block] = []
    for block in usable:
        width = len(block.text.split("\n"))
        spots = board.free(positions.get(block.text, ()), width)
        allowed = [
            (path, start)
            for path, start in spots
            if _replaceable(prepared[path][1], start, width)
        ]
        if not allowed:
            continue
        board.take(allowed, width)
        adjusted = replace(
            block,
            lines=width,
            files=tuple(sorted({path for path, _ in allowed})),
            occurrences=len(allowed),
        )
        used.append(adjusted)
        for path, start in allowed:
            plan.setdefault(path, []).append((start, width, adjusted))

    if not used:
        return dict(texts), []

    # Ids are assigned *after* the substitution set is final. Assigning earlier
    # -- over the blocks as found rather than as used -- means a block whose
    # occurrences were mostly rejected could change rank, and the marker
    # written into a file would name an id that :func:`render_shared` gives to a
    # different block.
    ids = assign_ids(used)

    factored: dict[str, str] = {}
    for path in sorted(texts):
        replacements = plan.get(path)
        if not replacements:
            factored[path] = texts[path]
            continue
        # ``target`` is bound as a default argument rather than closed over,
        # so the marker is built for this path and not for the last one the
        # loop happened to visit.
        target = path
        factored[target] = _rewrite(
            texts[target],
            replacements,
            lambda block, at=target: marker_for(at, ids[block], block.lines),
        )
    return factored, used


def _rewrite(
    text: str,
    replacements: list[tuple[int, int, Block]],
    make_marker,
) -> str:
    """Swap runs of lines for one marker line each, keeping the rest verbatim.

    Line endings and the presence or absence of a final newline are preserved,
    including CRLF: a factored Windows file comes back a Windows file.
    """
    lines, ending = _split_lines(text)
    starts = {start: (width, block) for start, width, block in replacements}
    out: list[str] = []
    skip_until = 0
    for index, line in enumerate(lines):
        if index < skip_until:
            continue
        hit = starts.get(index)
        if hit is None:
            out.append(line)
            continue
        width, block = hit
        out.append(make_marker(block))
        skip_until = index + width
    # ``split`` leaves a trailing empty element for a file that ended with a
    # newline, so joining puts it straight back.
    return ending.join(out)


def savings(
    tokenizer,
    original: dict[str, str],
    factored: dict[str, str],
    blocks: list[Block],
) -> int:
    """Measured token delta: how many tokens factoring actually gave back.

    Returned value is signed and is *measured*, not projected: every file is
    counted with the caller's real tokenizer, markers and all. A negative result
    is possible and honest -- it means the blocks handed in were too small or
    too rare to pay for their own markers, which is information worth having
    rather than something to clamp away.
    """
    return savings_report(tokenizer, original, factored, blocks).tokens_saved


@dataclass(frozen=True)
class SavingsReport:
    """The measurement behind :func:`savings`, with the reasons attached."""

    #: Tokens counted across every file in the input, before and after.
    tokens_before: int
    tokens_after: int

    #: Files whose text actually changed. Files passed through untouched are not
    #: counted twice just for being present.
    files_changed: tuple[str, ...]

    #: The blocks that were substituted, in :func:`factor` order.
    blocks_used: tuple[Block, ...]

    #: ``sum((occurrences - 1) * tokens)`` over :attr:`blocks_used`, measured
    #: with the same tokenizer. This is the optimistic figure: it never charges
    #: for the markers. Compare it against :attr:`tokens_saved` to see how much
    #: of the win the markers ate.
    projected: int

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def marker_cost(self) -> int:
        """Projected saving minus real saving. Negative means the markers paid."""
        return self.projected - self.tokens_saved


def savings_report(
    tokenizer,
    original: dict[str, str],
    factored: dict[str, str],
    blocks: list[Block],
) -> SavingsReport:
    """The long form of :func:`savings`."""
    paths = sorted(set(original) | set(factored))
    before = 0
    after = 0
    changed: list[str] = []
    for path in paths:
        old_text = original.get(path, "")
        new_text = factored.get(path, "")
        before += tokenizer.count(old_text).tokens
        after += tokenizer.count(new_text).tokens
        if old_text != new_text:
            changed.append(path)
    projected = sum(
        max(0, block.occurrences - 1) * tokenizer.count(block.text).tokens
        for block in blocks
    )
    return SavingsReport(
        tokens_before=before,
        tokens_after=after,
        files_changed=tuple(changed),
        blocks_used=tuple(blocks),
        projected=projected,
    )


def render_shared(blocks: list[Block], tokenizer) -> str:
    """The markdown section listing every hoisted block exactly once.

    This is the other half of the contract: markers in the files are pointers
    into this text, so it has to carry the blocks verbatim, in id order, and it
    has to say loudly that the files it came from no longer parse.

    Returns the empty string when nothing was factored, so a caller can append
    the result unconditionally.
    """
    if not blocks:
        return ""
    ids = assign_ids(blocks)
    parts: list[str] = [
        f"## {SHARED_SECTION_TITLE}",
        "",
        "Runs of text that appear repeatedly across the files above. Each one is",
        "printed once, here; in the files it was replaced by a marker naming its",
        "id and line count.",
        "",
        "> **A factored file is a reading aid, not source.** A marker stands in",
        "> for text that is no longer inline, so a file containing one will not",
        "> parse, will not lint, and must not be executed, compiled, or diffed",
        "> against the original. Read these bundles; run the repository.",
        "",
    ]
    for block in sorted(blocks, key=lambda b: ids[b]):
        tokens = tokenizer.count(block.text).tokens
        where = (
            f"{len(block.files)} file{'s' if len(block.files) != 1 else ''}"
        )
        parts.append(
            f"### {ids[block]} · {block.lines} lines · ~{tokens:,} tokens · "
            f"{block.occurrences:,} occurrences in {where}"
        )
        parts.append("")
        parts.append(_fence(block.text, _language_for(block.files)))
        parts.append("")
    dropped = sum(max(0, block.occurrences - 1) for block in blocks)
    parts.append(
        f"_Hoisted {len(blocks)} block(s), removing an estimated {dropped:,} "
        f"duplicate cop{'y' if dropped == 1 else 'ies'}. Estimate only: the "
        "markers cost tokens too._"
    )
    return "\n".join(parts) + "\n"


#: A deliberately small subset of :data:`ctxpack.render.LANGUAGES`, for the one
#: place a fence is needed here. Duplicated rather than imported because
#: ``render`` imports ``pack``, and ``pack`` is a plausible caller of this
#: module -- importing back would make the cycle possible.
_LANGUAGES: dict[str, str] = {
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".css": "css",
    ".go": "go",
    ".h": "c",
    ".hpp": "cpp",
    ".html": "html",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "jsx",
    ".kt": "kotlin",
    ".lua": "lua",
    ".md": "markdown",
    ".php": "php",
    ".proto": "protobuf",
    ".py": "python",
    ".rb": "ruby",
    ".rs": "rust",
    ".scala": "scala",
    ".sh": "shell",
    ".sql": "sql",
    ".swift": "swift",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".vue": "vue",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
}


def _language_for(paths: Sequence[str]) -> str:
    """Fence language for a block, from the most common extension among its files."""
    counts: dict[str, int] = {}
    for path in paths:
        dot = path.rfind(".")
        ext = path[dot:].lower() if dot >= 0 else ""
        counts[ext] = counts.get(ext, 0) + 1
    if not counts:
        return ""
    # ``max`` on a sorted key list keeps this deterministic when two extensions
    # tie, which happens the moment a block spans a python and a test file.
    best = max(sorted(counts), key=lambda ext: counts[ext])
    return _LANGUAGES.get(best, "")


def _fence(text: str, language: str) -> str:
    """Backtick fence long enough to survive the text inside it.

    Same hazard :func:`ctxpack.render.fence` handles: a block of markdown
    containing ``` would end the fence early and spill the rest of the section
    into prose.
    """
    longest = 0
    for run in re.findall(r"`+", text):
        longest = max(longest, len(run))
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{language}\n{text}\n{ticks}"

