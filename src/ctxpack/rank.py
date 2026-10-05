"""Scoring files by how much they are worth spending tokens on.

The ordering here is the difference between a useful bundle and an expensive
one. Rank by size or by directory listing and you get a pile of leaf modules;
rank by "what would a new engineer read first" and you get the README, the
entrypoint, the type definitions and the schema.

Every contribution is recorded as a :class:`Signal` so ``ctxpack explain`` can
answer "why did it pick this file?", which is the difference between a heuristic
you can trust and one you have to guess about.
"""

from __future__ import annotations

from dataclasses import dataclass

from .walk import Candidate

__all__ = [
    "Scored",
    "Signal",
    "rank_all",
    "score_candidate",
]


@dataclass(frozen=True)
class Signal:
    """One named contribution to a file's score."""

    key: str
    weight: float
    detail: str = ""

    def __str__(self) -> str:
        arrow = "+" if self.weight >= 0 else "-"
        return f"{arrow}{abs(self.weight):.2f} {self.key}"


@dataclass(frozen=True)
class Scored:
    candidate: Candidate
    score: float
    signals: tuple[Signal, ...]

    @property
    def path(self) -> str:
        return self.candidate.path

    def explain(self) -> str:
        parts = ", ".join(str(s) for s in self.signals)
        return f"{self.score:6.2f}  {self.path}\n        {parts}"


# Directory names that usually hold the code worth reading first.
ROOT_MARKERS = frozenset(
    {"src", "lib", "app", "packages", "internal", "cmd", "pkg", "core", "api"}
)

# Stems that mark an entrypoint or a module other code leans on.
ANCHOR_STEMS = frozenset(
    {
        "index", "main", "mod", "__init__", "app", "server", "cli", "setup",
        "lib", "entry", "bootstrap", "config", "schema", "types", "models",
        "router", "routes", "client", "api", "core",
    }
)

SOURCE_EXTS = frozenset(
    {
        ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".go",
        ".rs", ".java", ".kt", ".kts", ".rb", ".php", ".c", ".h", ".cc",
        ".cpp", ".cxx", ".hpp", ".cs", ".swift", ".scala", ".sql", ".graphql",
        ".gql", ".proto", ".vue", ".svelte", ".ex", ".exs", ".erl", ".hs",
        ".ml", ".clj", ".zig", ".lua", ".dart", ".sh", ".bash", ".zsh",
    }
)

DOC_EXTS = frozenset({".md", ".mdx", ".rst", ".txt", ".adoc"})
CONFIG_EXTS = frozenset({".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env"})

# Documents that describe history or policy rather than the system.
NOISE_STEMS = frozenset(
    {
        "changelog", "changes", "history", "license", "licence", "copying",
        "contributing", "code_of_conduct", "authors", "notice", "security",
        "funding", "governance", "maintainers",
    }
)

# Directories whose contents are structural but not interesting per token.
LOW_VALUE_DIRS = frozenset(
    {"test", "tests", "spec", "specs", "__tests__", "e2e", "fixtures",
     "testdata", "migrations", "examples", "example", "scripts", "tools",
     "docs", "doc", "assets", "static", "public", "vendor", "third_party"}
)

VENDOR_DIRS = frozenset({"vendor", "third_party", "node_modules", "extern", "deps"})

GENERATED_MARKERS = (".min.", ".generated.", ".g.", "_pb2", ".pb.", ".d.ts")


def _stem(name: str) -> str:
    base = name.rsplit("/", 1)[-1]
    stem = base.rsplit(".", 1)[0] if "." in base.lstrip(".") else base
    return stem.lower()


def _segments(path: str) -> list[str]:
    return [part.lower() for part in path.split("/")[:-1]]


def score_candidate(candidate: Candidate) -> Scored:
    """Score one file. Higher is better; the scale is arbitrary but stable."""
    path = candidate.path
    segments = _segments(path)
    stem = _stem(path)
    ext = candidate.ext
    size = candidate.size
    depth = candidate.depth
    top = segments[0] if segments else ""

    signals: list[Signal] = []

    # --- the obvious wins -------------------------------------------------
    if stem == "readme" or stem.startswith("readme"):
        # Weighted to win outright. The README is the one file whose job is to
        # orient a reader who has never seen the repository, so it earns its
        # tokens ahead of any individual module, however central that module is.
        signals.append(Signal("readme", 7.0, stem))
    elif stem in ANCHOR_STEMS:
        signals.append(Signal("anchor", 2.4, stem))
    elif stem.endswith(("server", "cli", "api", "main", "index")):
        signals.append(Signal("anchor-suffix", 1.3, stem))

    if top in ROOT_MARKERS:
        signals.append(Signal("source-root", 1.9, top))

    # --- what kind of file is it -----------------------------------------
    if ext in SOURCE_EXTS:
        signals.append(Signal("source", 1.5, ext))
    elif ext in DOC_EXTS:
        signals.append(Signal("docs", 1.0, ext))
    elif ext in CONFIG_EXTS:
        signals.append(Signal("config", 0.7, ext))

    if depth == 0:
        signals.append(Signal("top-level", 1.2, path))
    elif depth <= 2:
        signals.append(Signal("shallow", 0.6, f"depth {depth}"))

    # --- structural definitions, which everything else references ----------
    structural = stem.endswith(("types", "schema", "model", "models", "config", "constants"))
    if structural and ext in SOURCE_EXTS | CONFIG_EXTS:
        signals.append(Signal("structural", 1.4, stem))

    # --- penalties --------------------------------------------------------
    if stem in NOISE_STEMS:
        signals.append(Signal("boilerplate-doc", -4.0, stem))
    if any(marker in path for marker in GENERATED_MARKERS):
        signals.append(Signal("generated", -3.5, ""))
    if any(segment in VENDOR_DIRS for segment in segments):
        signals.append(Signal("vendored", -3.0, ""))
    if top in LOW_VALUE_DIRS:
        signals.append(Signal("low-value-dir", -1.5, top))

    if size > 300_000:
        signals.append(Signal("huge", -2.5, f"{size // 1024}kB"))
    elif size > 80_000:
        signals.append(Signal("large", -1.2, f"{size // 1024}kB"))
    elif size < 2_000:
        signals.append(Signal("small", 0.4, f"{size}B"))

    # Prefer files near the root, but only mildly: a monorepo's real code is
    # often three directories down.
    if depth > 2:
        signals.append(Signal("depth", -0.45 * min(depth - 2, 6), f"depth {depth}"))

    return Scored(
        candidate=candidate,
        score=round(sum(s.weight for s in signals), 4),
        signals=tuple(signals),
    )


def rank_all(candidates: list[Candidate]) -> list[Scored]:
    """Score everything, best first. Ties break on path for reproducibility."""
    scored = [score_candidate(c) for c in candidates]
    scored.sort(key=lambda s: (-s.score, s.path))
    return scored
