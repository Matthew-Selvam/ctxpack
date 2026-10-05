"""Fitting the best possible context bundle into a token budget.

Given a ranked list of files and a number of tokens, choose a subset. The naive
answers -- take files in rank order, or take the smallest files that fit -- are
both bad: the first produces one giant package and nothing else, the second
fills the budget with fixtures and changelogs.

What this does instead is greedy selection under a set of diversity constraints:

* **Per-file cap.** One 300kB generated client must not eat the window.
* **Per-extension and per-directory caps.** Otherwise a repo with 400 Vue files
  packs 400 Vue files.
* **Near-duplicate detection.** Real repositories contain the same file
  dozens of times over -- generated locale bundles, vendored forks, parallel
  test fixtures. Those are pure token waste, and the single highest-value
  thing this module does.
* **Line-aware truncation.** A file too big for its cap is cut at a line
  boundary and marked, rather than dropped or mangled.

Every constraint is a knob on :class:`Budget` with a defensible default.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .errors import CtxpackError
from .rank import Scored, rank_all
from .tokens import Tokenizer
from .walk import Candidate, Discovery, read_text

__all__ = ["MODES", "Budget", "Document", "ManifestEntry", "PackResult", "Packer"]

MODES = ("balanced", "coverage", "depth")

#: Characters-per-token used for the cheap pre-pass when locating a truncation
#: point. Only used to pick a starting line; the result is exact-counted after.
_CHARS_PER_TOKEN = 3.4

#: Lines shorter than this, or containing no alphanumerics, are too generic to
#: identify a file. Counting them would make every brace-heavy file "similar".
_MIN_SIGNATURE_LEN = 12
_MAX_SIGNATURE_LINES = 400

#: Progress reporting interval, in files processed.
_REPORT_EVERY = 500


@dataclass(frozen=True)
class Budget:
    """Knobs for one packing run. Fractions are of ``total``."""

    total: int
    per_file_frac: float = 0.12
    per_ext_frac: float = 0.55
    per_dir_frac: float = 0.40
    max_per_dir: int = 8
    max_per_ext: int = 0  # 0 means the fraction cap is the only limit
    header_frac: float = 0.03
    manifest_frac: float = 0.12
    manifest: str = "full"  # full | paths | none
    dedupe: bool = True
    dedupe_threshold: float = 0.85
    dedupe_min_lines: int = 12
    truncate: bool = True
    max_scan: int = 3000

    def __post_init__(self) -> None:
        if self.total <= 0:
            raise CtxpackError("budget must be a positive number of tokens")
        if self.manifest not in ("full", "paths", "none"):
            raise CtxpackError(
                f"unknown manifest mode {self.manifest!r}; use full, paths or none"
            )
        if not 0.0 < self.dedupe_threshold <= 1.0:
            raise CtxpackError("dedupe_threshold must be in (0, 1]")

    @property
    def per_file(self) -> int:
        return max(200, int(self.total * self.per_file_frac))

    @property
    def per_ext(self) -> int:
        return max(400, int(self.total * self.per_ext_frac))

    @property
    def per_dir(self) -> int:
        return max(400, int(self.total * self.per_dir_frac))

    @property
    def header(self) -> int:
        """Tokens held back for the rendered header.

        The ``max(300, ...)`` floor is what keeps the header from being absurdly
        small on a large budget -- but left uncapped it lets a tiny budget be
        consumed entirely by the reserve, which is how a 400-token budget ended
        up 208 tokens over. The quarter-budget cap keeps tiny budgets usable
        while still guaranteeing ``total - header >= 0``.
        """
        return min(
            2000,
            max(300, int(self.total * self.header_frac)),
            max(0, self.total // 4),
        )

    @property
    def manifest_cap(self) -> int:
        return max(500, int(self.total * self.manifest_frac))

    @property
    def index_cap(self) -> int:
        """Hard ceiling on the index: whatever the header does not need."""
        return min(self.manifest_cap, max(0, self.total - self.header))

    def manifest_tokens_cap_exceeded(self, tokens: int) -> bool:
        """True when the *policy* cap (not the hard cap) is what bit."""
        return tokens > self.manifest_cap


@dataclass
class Document:
    """A file that made it into the bundle."""

    candidate: Candidate
    scored: Scored
    text: str
    tokens: int
    lines: int
    truncated: bool = False
    original_tokens: int = 0

    @property
    def path(self) -> str:
        return self.candidate.path

    @property
    def ext(self) -> str:
        return self.candidate.ext

    @property
    def dirname(self) -> str:
        return self.path.rsplit("/", 1)[0] if "/" in self.path else ""

    @property
    def saved(self) -> int:
        return max(0, self.original_tokens - self.tokens)


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    tokens: int
    size: int
    estimated: bool = False  # priced from size without reading
    included: bool = False


@dataclass
class Duplicate:
    path: str
    of_path: str
    similarity: float


@dataclass(frozen=True)
class SharedBlock:
    """A block hoisted out of several files into one place.

    A separate type from :mod:`ctxpack.boiler`'s ``Block`` on purpose: the
    boiler type is frozen and carries detection metadata (projected savings,
    which file it came from first). What a rendered bundle needs is just an id
    to point markers at plus the text, so the CLI converts rather than
    mutating.
    """

    id: str
    lines: int
    occurrences: int
    text: str


@dataclass
class PackResult:
    root: Path
    documents: list[Document]
    manifest: list[ManifestEntry]
    manifest_text: str
    manifest_tokens: int
    budget: int
    header_reserve: int
    mode: str
    encoding: str
    method: str
    calibration: float
    discovered: int
    scanned: int
    unscanned: int
    ignored: int = 0
    duplicates: list[Duplicate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: Blocks hoisted out of many files into one shared section, produced by
    #: :mod:`ctxpack.boiler`. Attached post-hoc by the CLI rather than computed
    #: here, because factoring is a rendering decision about the bundle rather
    #: than part of choosing which files belong in it.
    shared_blocks: list[SharedBlock] = field(default_factory=list)

    @property
    def content_tokens(self) -> int:
        return sum(d.tokens for d in self.documents)

    @property
    def accounted(self) -> int:
        """Tokens ctxpack committed to, before the header is rendered."""
        return self.content_tokens + self.manifest_tokens + self.header_reserve

    @property
    def over_budget(self) -> int:
        return max(0, self.accounted - self.budget)

    @property
    def truncated(self) -> list[Document]:
        return [d for d in self.documents if d.truncated]

    @property
    def unscanned_bytes(self) -> int:
        return sum(m.size for m in self.manifest if m.estimated)


def _signature(text: str) -> set[str]:
    """A whitespace-insensitive fingerprint of a file's distinctive lines."""
    out: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) < _MIN_SIGNATURE_LEN:
            continue
        if not re.search(r"[A-Za-z0-9]", stripped):
            continue
        out.add(stripped)
        if len(out) >= _MAX_SIGNATURE_LINES:
            break
    return out


class _LineIndex:
    """Inverted index from signature line to document, for cheap Jaccard.

    Comparing every candidate against every selected document pairwise is
    O(n*m) over sets of hundreds of lines. Going through postings instead makes
    the common case -- a candidate sharing nothing with anything already picked
    -- cost one dict miss per line.
    """

    __slots__ = ("_postings", "_sizes")

    def __init__(self) -> None:
        self._postings: dict[str, list[int]] = defaultdict(list)
        # Keyed by document id rather than appended, so a caller that skips a
        # registration cannot silently shift every later lookup by one.
        self._sizes: dict[int, int] = {}

    def add(self, doc_id: int, lines: set[str]) -> None:
        self._sizes[doc_id] = len(lines)
        for line in lines:
            self._postings[line].append(doc_id)

    def best_match(self, lines: set[str]) -> tuple[int, float] | None:
        if not lines or not self._sizes:
            return None
        shared: dict[int, int] = defaultdict(int)
        for line in lines:
            for doc_id in self._postings.get(line, ()):
                shared[doc_id] += 1
        best: tuple[int, float] | None = None
        for doc_id, count in shared.items():
            union = len(lines) + self._sizes[doc_id] - count
            if union <= 0:
                continue
            score = count / union
            if best is None or score > best[1]:
                best = (doc_id, score)
        return best


class Packer:
    """Builds a :class:`PackResult` from a :class:`Discovery`.

    ``rerank`` replaces the default ranking outright, which is how the import
    graph gets a vote: the caller computes hop distances and hands back a
    re-scored list. It is a parameter rather than a hard dependency because
    reachability analysis is optional and genuinely slower -- a caller that
    does not want it should not pay for it, and ``pack.py`` should not import a
    module it does not need.

    ``transform`` rewrites a file's text *before* it is counted, which is what
    makes ``--outline`` worth using. Shrinking bodies after selection would
    leave the freed budget unspent -- selection had already decided the file was
    too big to fit. Transforming first means the packer sees the outline's real
    cost and packs correspondingly more files.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        budget: Budget,
        *,
        mode: str = "balanced",
        progress=None,
        rerank: Callable[[list[Candidate]], list[Scored]] | None = None,
        transform: Callable[[Candidate, str], str] | None = None,
    ) -> None:
        if mode not in MODES:
            raise CtxpackError(
                f"unknown mode {mode!r}; choose one of " + ", ".join(MODES)
            )
        self.tokenizer = tokenizer
        self.budget = budget
        self.mode = mode
        self.progress = progress
        self.rerank = rerank or rank_all
        self.transform = transform

    # -- public ------------------------------------------------------------

    def pack(self, discovery: Discovery) -> PackResult:
        budget = self.budget
        notes: list[str] = []

        scored = self.rerank(discovery.files)
        if budget.max_scan and len(scored) > budget.max_scan:
            notes.append(
                f"scanned the top {budget.max_scan:,} of {len(scored):,} files by "
                f"rank; the rest appear in the index but their contents were not read"
            )
        considered = scored[: budget.max_scan] if budget.max_scan else scored
        unscanned = scored[budget.max_scan :] if budget.max_scan else []

        # 1. Read and price everything we are considering. This is the bulk of
        #    the runtime, so it happens once, up front.
        documents: list[Document] = []
        unreadable: list[Candidate] = []
        for index, item in enumerate(considered, start=1):
            text = read_text(item.candidate)
            if text is None:
                unreadable.append(item.candidate)
                continue
            if self.transform is not None:
                # Transform before counting, so the budget is spent against the
                # text that will actually be shipped.
                text = self.transform(item.candidate, text)
            count = self.tokenizer.count(text)
            documents.append(
                Document(
                    candidate=item.candidate,
                    scored=item,
                    text=text,
                    tokens=count.tokens,
                    lines=text.count("\n") + 1 if text else 0,
                )
            )
            if self.progress and index % _REPORT_EVERY == 0:
                self.progress(f"  priced {index:,}/{len(considered):,} files")

        for doc in documents:
            doc.original_tokens = doc.tokens

        # 2. The index is built from *every* file found, so the model always
        #    knows what exists even where the contents were not read.
        manifest = self._build_manifest(documents, unscanned)

        # 3. Reserve the index and the header before spending on content.
        manifest_text, manifest_tokens = self._render_manifest(manifest)
        if budget.manifest == "none":
            manifest_text, manifest_tokens = "", 0
        elif manifest_tokens > budget.index_cap:
            # Two separate ceilings: the policy cap (a share of the budget) and
            # the hard cap (whatever the header reserve leaves). The index must
            # never be able to crowd out content or push the total over.
            if budget.manifest_tokens_cap_exceeded(manifest_tokens):
                notes.append(
                    f"index trimmed from {manifest_tokens:,} to "
                    f"{budget.index_cap:,} tokens"
                )
            manifest_text, manifest_tokens = self._trim_manifest(
                manifest, budget.index_cap
            )

        content_budget = budget.total - manifest_tokens - budget.header
        selected, duplicates, skipped = self._select(documents, content_budget, notes)

        unused = content_budget - sum(d.tokens for d in selected)
        if unused > budget.total * 0.15:
            notes.append(self._explain_headroom(unused, budget))

        if skipped:
            notes.append(
                f"{len(skipped):,} files left out after budget and diversity limits"
            )
        if duplicates:
            notes.append(
                f"{len(duplicates):,} near-duplicate files collapsed "
                f"(>= {budget.dedupe_threshold:.0%} similar)"
            )
        if unreadable:
            notes.append(f"{len(unreadable):,} files unreadable or not text")
        if unscanned:
            notes.append(f"{len(unscanned):,} files below the scan cutoff")

        return PackResult(
            root=discovery.root,
            documents=selected,
            manifest=manifest,
            manifest_text=manifest_text,
            manifest_tokens=manifest_tokens,
            budget=budget.total,
            header_reserve=budget.header,
            mode=self.mode,
            encoding=self.tokenizer.encoding,
            method=self.tokenizer.method,
            calibration=self.tokenizer.calibration,
            discovered=len(discovery.files),
            scanned=len(documents),
            unscanned=len(unscanned),
            ignored=discovery.ignored,
            duplicates=duplicates,
            notes=notes,
        )

    # -- steps -------------------------------------------------------------

    def _build_manifest(
        self, documents: list[Document], unscanned: list[Scored]
    ) -> list[ManifestEntry]:
        entries = [
            ManifestEntry(doc.path, doc.tokens, doc.candidate.size) for doc in documents
        ]
        # Unread files are priced from their size rather than their contents.
        # Cheap, slightly imprecise, and honestly labelled in the output.
        for item in unscanned:
            size = item.candidate.size
            entries.append(
                ManifestEntry(
                    path=item.candidate.path,
                    tokens=max(1, math.ceil(size / _CHARS_PER_TOKEN)),
                    size=size,
                    estimated=True,
                )
            )
        return entries

    def _render_manifest(self, entries: list[ManifestEntry]) -> tuple[str, int]:
        if self.budget.manifest == "paths":
            lines = [f"  {e.path}" for e in entries]
        else:
            lines = [
                f"  {e.tokens:>9,}  {e.size:>10,}  {e.path}" for e in entries
            ]
        text = "\n".join(lines)
        return text, self.tokenizer.count(text).tokens

    def _trim_manifest(
        self, entries: list[ManifestEntry], cap: int
    ) -> tuple[str, int]:
        """Drop the lowest-ranked entries until the index fits its cap."""
        if not entries or cap <= 0:
            return "", 0
        # One proportional guess, then exact counting -- a binary search here
        # would cost more tokenizer calls than it saves.
        full_tokens = max(1, self.tokenizer.count(
            "\n".join(
                f"  {e.path}" if self.budget.manifest == "paths"
                else f"  {e.tokens:>9,}  {e.size:>10,}  {e.path}"
                for e in entries
            )
        ).tokens)
        keep = max(1, min(len(entries), int(len(entries) * cap / full_tokens)))

        def build(count: int) -> tuple[str, int]:
            shown = entries[:count]
            if self.budget.manifest == "paths":
                lines = [f"  {e.path}" for e in shown]
            else:
                lines = [
                    f"  {e.tokens:>9,}  {e.size:>10,}  {e.path}" for e in shown
                ]
            dropped = len(entries) - len(shown)
            if dropped:
                lines.append(f"  ... and {dropped:,} more files (not listed)")
            text = "\n".join(lines)
            return text, self.tokenizer.count(text).tokens

        text, tokens = build(keep)
        if tokens > cap:
            for _ in range(8):
                keep = int(keep * cap / tokens * 0.9)
                if keep < 1:
                    # Even a single entry does not fit; an honest empty index
                    # beats an index that breaks the budget.
                    return "", 0
                text, tokens = build(keep)
                if tokens <= cap:
                    break
            else:
                return "", 0
        return text, tokens

    def _select(
        self,
        documents: list[Document],
        content_budget: int,
        notes: list[str],
    ) -> tuple[list[Document], list[Duplicate], list[Document]]:
        budget = self.budget
        if content_budget <= 0:
            notes.append("budget is too small for any file content")
            return [], [], list(documents)

        per_file = budget.per_file
        per_ext = budget.per_ext
        per_dir = budget.per_dir

        ext_spend: dict[str, int] = defaultdict(int)
        ext_count: dict[str, int] = defaultdict(int)
        dir_spend: dict[str, int] = defaultdict(int)
        dir_count: dict[str, int] = defaultdict(int)

        index = _LineIndex()
        signatures: list[set[str]] = []
        selected: list[Document] = []
        duplicates: list[Duplicate] = []
        skipped: list[Document] = []
        blockers: dict[str, int] = defaultdict(int)
        used = 0

        def skip(doc: Document, reason: str) -> None:
            skipped.append(doc)
            blockers[reason] += 1

        for doc in self._ordered(documents):
            # Truncate *before* the budget test. Testing first compares the full
            # file against the remaining budget and skips anything oversized,
            # which starves the bundle: a 4,000-token file gets dropped instead
            # of being cut down to its per-file cap and included.
            if doc.tokens > per_file:
                if not budget.truncate:
                    skip(doc, "too big and --no-truncate")
                    continue
                self._truncate(doc, per_file)

            if used + doc.tokens > content_budget:
                skip(doc, "out of budget")
                continue

            ext, dirname = doc.ext, doc.dirname
            if budget.max_per_ext and ext_count[ext] >= budget.max_per_ext:
                skip(doc, f"{ext} file-count cap")
                continue
            if dir_count[dirname] >= budget.max_per_dir:
                skip(doc, f"{dirname or '.'} file-count cap")
                continue
            if selected and ext_spend[ext] + doc.tokens > per_ext:
                skip(doc, f"{ext} share cap")
                continue
            if selected and dir_spend[dirname] + doc.tokens > per_dir:
                skip(doc, f"{dirname or '.'} share cap")
                continue

            signature = _signature(doc.text)
            if budget.dedupe and len(signature) >= budget.dedupe_min_lines:
                match = index.best_match(signature)
                if match and match[1] >= budget.dedupe_threshold:
                    duplicates.append(
                        Duplicate(
                            path=doc.path,
                            of_path=selected[match[0]].path,
                            similarity=match[1],
                        )
                    )
                    blockers["near-duplicate"] += 1
                    continue

            doc_id = len(selected)
            selected.append(doc)
            signatures.append(signature)
            # Always add, even for an empty signature: `add` appends to
            # `_sizes` unconditionally, and skipping it here would desync
            # those sizes from document indices. That crashed with an
            # IndexError on any repo containing a file with no distinctive
            # lines (a licence header, a table of contents, a `.gitkeep`).
            index.add(doc_id, signature)
            ext_spend[ext] += doc.tokens
            ext_count[ext] += 1
            dir_spend[dirname] += doc.tokens
            dir_count[dirname] += 1
            used += doc.tokens

        self._last_blockers = dict(blockers)
        return selected, duplicates, skipped

    def _explain_headroom(self, unused: int, budget: Budget) -> str:
        """Say which limit stopped us, instead of leaving budget unexplained.

        A bundle that uses 60% of its budget looks like a bug. Usually it is
        diversity working as intended, but "usually" is not good enough -- this
        reports the actual blocker recorded during selection.
        """
        blockers = getattr(self, "_last_blockers", {})
        if blockers:
            reason, count = max(blockers.items(), key=lambda kv: kv[1])
            return (
                f"left ~{unused:,} tokens unused; the main blocker was "
                f"{reason} ({count:,} files)"
            )
        return (
            f"left ~{unused:,} tokens unused: there were no more candidates worth "
            f"packing within the diversity caps"
        )

    def _ordered(self, documents: list[Document]) -> list[Document]:
        """Files in the order this mode wants to consider them."""
        if self.mode != "coverage":
            return documents

        # Round-robin across directories: take the best file from every
        # directory before taking a second file from any of them. Breadth over
        # depth, for when you have no idea what matters yet.
        groups: dict[str, list[Document]] = defaultdict(list)
        for doc in documents:
            groups[doc.dirname].append(doc)
        for group in groups.values():
            group.sort(key=lambda d: (-d.scored.score, d.path))
        ordered: list[Document] = []
        for rank in range(max((len(g) for g in groups.values()), default=0)):
            for name in sorted(groups):
                group = groups[name]
                if rank < len(group):
                    ordered.append(group[rank])
        return ordered

    def _truncate(self, doc: Document, max_tokens: int) -> None:
        """Cut ``doc`` at a line boundary to fit ``max_tokens``."""
        if doc.tokens <= max_tokens:
            return
        original_lines = doc.lines
        lines = doc.text.splitlines(keepends=True)
        if not lines:
            return

        # Cheap proportional pass to find a starting line, then shrink by
        # exact counting. Never grows the kept prefix beyond the budget.
        target_chars = max_tokens * _CHARS_PER_TOKEN
        cut = len(lines)
        running = 0
        for index, line in enumerate(lines):
            running += len(line)
            if running > target_chars:
                cut = index
                break
        cut = max(1, cut)

        prefix = "".join(lines[:cut])
        guard = 0
        while cut > 1 and self.tokenizer.count(prefix).tokens > max_tokens and guard < 40:
            cut = max(1, int(cut * 0.9))
            prefix = "".join(lines[:cut])
            guard += 1

        marker = (
            f"\n... [ctxpack truncated: {original_lines - cut:,} of "
            f"{original_lines:,} lines omitted]\n"
        )
        kept_tokens = self.tokenizer.count(prefix + marker).tokens
        if kept_tokens > max_tokens:
            # The marker alone can matter in a tiny budget; drop it rather
            # than blow the cap.
            prefix = "".join(lines[: max(1, cut // 2)])
            kept_tokens = self.tokenizer.count(prefix).tokens

        doc.text = prefix + marker
        doc.tokens = kept_tokens
        doc.lines = cut
        doc.truncated = True
