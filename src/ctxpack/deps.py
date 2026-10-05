"""Import graphs -- the one ranking signal that is not a guess.

``rank`` decides what matters from the shape of a file: is it a README, is it
an anchor filename, how deep is it.  Those are proxies for intent.  The signal
that is not a proxy is reachability: a file the program never imports is dead
code, and a file three hops from ``main`` is something the program genuinely
executes.  So this module reads the imports out of the files ctxpack already
found, resolves them against the files that are actually present, and returns
hop distances from the entrypoints.  :func:`boost` folds that into a score.

Every design choice below trades accuracy for the ability to run on a repo it
has never seen without asking permission:

* **Python is parsed with :mod:`ast`.**  Correct, or it yields nothing.
* **JavaScript is regexed.**  We want specifier *strings*, which are a small
  regular subset of the file.  A real ES module parser is a parser generator
  plus a language spec, and it would also raise on input we must survive.
* **Resolution is path-shaped, not semantic.**  No ``sys.path`` emulation, no
  ``node_modules``, no importlib/monkeypatch tracking, no conditional imports,
  no ``__getattr__`` re-export discovery.  An unresolvable import is dropped,
  which biases the graph towards under-claiming rather than inventing edges.
* **The graph is over known paths only.**  Nothing is read from disk by
  resolution, so the module cannot be defeated by a symlink loop or a mount
  point, and ``node_modules`` cannot quietly inflate the result.

The whole module degrades to "no useful signal".  Unparseable file, hostile
source, directory of generated code -- the answer is an empty import set, not
an exception.  Callers must treat "no entrypoints found" as "no signal", never
as "everything is reachable": a repo with no detectable entrypoint yields an
empty distance dict and :func:`boost` becomes a no-op.
"""

from __future__ import annotations

import ast
import math
import posixpath
import re
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TypeAlias

from .errors import CtxpackError
from .rank import CONFIG_EXTS, DOC_EXTS, Scored, Signal

__all__ = [
    "ENTRYPOINT_STEMS",
    "JS_EXTS",
    "PY_EXTS",
    "DepAnalysis",
    "Graph",
    "ModuleIndex",
    "analyse",
    "boost",
    "build_graph",
    "build_index",
    "find_entrypoints",
    "hop_distance",
    "parse_js_imports",
    "parse_python_imports",
    "reachable_from",
    "resolve_js",
    "resolve_python",
]

#: Adjacency over known paths only: importer -> imported paths we could resolve.
Graph: TypeAlias = dict[str, set[str]]

#: The only two families of file whose imports we try to read. Anything else is
#: a leaf in the graph. Parsing Markdown for the word "import" is not a thing.
PY_EXTS: frozenset[str] = frozenset({".py", ".pyi"})

#: Extension guesses, most-specific first, for a JS specifier with no extension.
#: ``.ts`` leads because in a TypeScript repo that is what ``./thing`` means,
#: even though a ``.js`` file may also sit right there.
JS_EXTS: tuple[str, ...] = (
    ".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs",
)

#: TypeScript's ``NodeNext`` extension substitution: a specifier ending in the
#: runtime extension also names the matching TypeScript source. Applied before
#: the extensionless guesses, because in a compiled-ESM repo this is not an edge
#: case, it is the norm -- and without it the graph comes out nearly empty while
#: appearing to have been built.
_TS_EXTENSION_SUBSTITUTION: dict[str, tuple[str, ...]] = {
    ".js": (".ts", ".tsx", ".d.ts"),
    # NodeNext also maps .jsx to .tsx. Missing this is the same bug as the
    # missing .js mapping one release earlier, one entry short: a React/TSX repo
    # silently loses every component edge.
    ".jsx": (".tsx",),
    ".mjs": (".mts", ".d.mts"),
    ".cjs": (".cts", ".d.cts"),
}

#: Stems that mark something the program starts from or fans out to. A
#: filename is the only evidence available -- nothing here reads a manifest,
#: a ``package.json`` main field or a ``[project.scripts]`` table.
ENTRYPOINT_STEMS: frozenset[str] = frozenset(
    {"main", "index", "__init__", "app", "server", "cli", "entry", "bootstrap"}
)

#: The handful of paths a reader expects to be found, on top of the stem rule.
#: Redundant with :data:`ENTRYPOINT_STEMS` as written; stated explicitly so the
#: intent survives someone narrowing the stem set.
_EXTRA_ENTRYPOINTS: frozenset[str] = frozenset(
    path
    for stem in ("index", "main")
    for ext in JS_EXTS
    for path in (f"{stem}{ext}", f"src/{stem}{ext}")
)

#: A bare JS specifier ("react", "components/Button") names a package, and we
#: deliberately do not walk ``node_modules``. We do try the stem index, which
#: can match many same-named files in a repo of leaf modules. Capped so one
#: ambiguous specifier cannot turn into hundreds of edges.
MAX_BARE_JS_MATCHES: int = 8

#: Hard ceiling on how many files one import specifier may resolve to, applied
#: to every fan-out path and to both languages. Measured on a synthetic
#: monorepo: 400 services each importing ``pkg.mod`` produced 320,000 edges
#: uncapped, growing quadratically, and made reachability wrong rather than
#: merely slow.
MAX_EDGES_PER_SPEC: int = 8

#: Per-hop decay for the reachable signal. Steep on purpose: a module four hops
#: out is plausibly still live code, but the graph cannot tell live from dead,
#: and over-claiming reachability would drown out the file's own signals.
HOP_DECAY: float = 0.6

#: ``import x from 'y'`` / ``import {a, b} from 'y'`` / ``import type {A} from
#: 'y'`` / ``import 'y'``. The ``from`` clause is optional, which covers both
#: the side-effect import and every named/default form in one pattern. Anchored
#: to line start so prose mentioning the word "import" is not an edge.
_JS_IMPORT: re.Pattern[str] = re.compile(
    r"""^[ \t]*import[ \t]+(?:[^'"]*?[ \t]from[ \t]+)?['"]([^'"]+)['"]""",
    re.MULTILINE,
)

#: ``export * from 'y'`` / ``export * as ns from 'y'`` / ``export {a} from 'y'``.
_JS_EXPORT: re.Pattern[str] = re.compile(
    r"""^[ \t]*export[ \t]+(?:\*|\{[^}]*\})[^'"]*?[ \t]from[ \t]+['"]([^'"]+)['"]""",
    re.MULTILINE,
)

#: ``require('y')`` in CommonJS and in bundler config files.
_JS_REQUIRE: re.Pattern[str] = re.compile(r"""\brequire[ \t]*\([ \t]*['"]([^'"]+)['"]""")

#: ``import('y')`` with a literal specifier. Computed specifiers are invisible
#: to us; that is a known hole, not an oversight.
_JS_DYNAMIC: re.Pattern[str] = re.compile(r"""\bimport[ \t]*\([ \t]*['"]([^'"]+)['"]""")

#: Dotted segments have to be identifiers, or they cannot be module names. This
#: is what rejects ``../..``, ``a//b``, ``a b`` and ``pkg.name`` noise from a
#: half-parsed specifier.
_IDENT: re.Pattern[str] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _language(path: str) -> str | None:
    """``"py"``, ``"js"`` or ``None`` for a file we will not try to parse."""
    ext = posixpath.splitext(path)[1].lower()
    if ext in PY_EXTS:
        return "py"
    if ext in JS_EXTS or path.endswith(".d.ts"):
        return "js"
    return None


def _stem(path: str) -> str:
    base = posixpath.basename(path)
    return base.rsplit(".", 1)[0] if "." in base else base


def _suffix_keys(path: str) -> tuple[str, ...]:
    """The path plus each left-trimmed suffix that keeps a directory component.

    This is what lets ``from ctxpack.walk import x`` resolve inside a ``src/``
    layout without pretending to know the project's import root: every file is
    reachable under every directory-stripped name it plausibly has. Suffixes
    that drop the directory entirely are not registered -- that would make
    ``import walk`` resolve to ``src/ctxpack/walk.py`` and turn every same-named
    leaf module in the repo into a false positive.
    """
    parts = path.split("/")
    return (path, *(("/".join(parts[start:])) for start in range(1, len(parts) - 1)))


@dataclass(frozen=True)
class ModuleIndex:
    """Known paths indexed two ways, so resolution never touches the disk.

    Built once per :func:`build_graph`. Two lookups, because two different
    questions get asked: does this candidate path exist, either exactly or under
    a directory-stripped name; and what files are *called* this, for the
    ambiguous bare-specifier case.
    """

    by_suffix: dict[str, tuple[str, ...]] = field(default_factory=dict)
    by_stem: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def lookup(self, candidate: str) -> tuple[str, ...]:
        """Real paths matching a candidate path.

        A single dict hit, not a scan: the exact path is always one of the keys
        :func:`_suffix_keys` registers, so this answers "does this exist" and
        "does this exist under a shorter root" in the same lookup.
        """
        return self.by_suffix.get(candidate, ())

    def lookup_stem(self, name: str) -> tuple[str, ...]:
        """Real paths whose filename stem is exactly ``name``."""
        return self.by_stem.get(name, ())

    def __len__(self) -> int:
        return len(self.by_suffix)


def build_index(paths: Iterable[str]) -> ModuleIndex:
    """Index ``paths`` by directory-stripped name and by filename stem.

    ``sorted(set(...))`` because the resulting tuples land in adjacency sets and
    the graph should be byte-identical across runs for the same input.
    """
    suffix: dict[str, list[str]] = {}
    stem: dict[str, list[str]] = {}
    for path in sorted(set(paths)):
        for key in _suffix_keys(path):
            suffix.setdefault(key, []).append(path)
        stem.setdefault(_stem(path), []).append(path)
    return ModuleIndex(
        by_suffix={key: tuple(hits) for key, hits in suffix.items()},
        by_stem={key: tuple(hits) for key, hits in stem.items()},
    )


def parse_python_imports(text: str) -> set[str]:
    """Every import target in ``text``, as raw dotted or dotted-with-dots names.

    Targets are symbolic rather than resolved, so the caller can see the
    distinction Python draws and this module does not: ``from a.b import c``
    yields both ``"a.b"`` (the module) and ``"a.b.c"`` (the thing taken from
    it).  We keep both because either may be the file that exists --
    ``a/b.py`` or ``a/b/c.py`` -- and guessing wrong would drop a real edge.
    ``c`` may equally be a class in ``a/b.py``, which is a harmless extra edge.

    Relative imports keep their leading dots, since the level is information:
    ``"."`` and ``"..x"`` cannot be read as plain names.

    ``from a.b import *`` contributes only ``"a.b"``; there is no symbol to
    name and guessing one would be worse than none.

    Returns an empty set for anything :mod:`ast` refuses. A single syntax error
    in a vendored file must not abort the walk of a repository.
    """
    try:
        tree = ast.parse(text)
    except Exception:
        # Deliberately broad: SyntaxError, ValueError (NUL bytes), RecursionError
        # and MemoryError are all "this file is not Python we understand", and
        # none of them are ctxpack's problem to report.
        return set()

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            # `import a.b as x` binds x to a.b; the alias is a local name, not a
            # module, so it never contributes a target.
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # Built by concatenation rather than by joining onto a base string:
            # the base may be dots only (``from . import x``), and appending
            # ``.x`` to ``"."`` would produce the meaningless spec ``"..x"``.
            dots = "." * (node.level or 0)
            base = dots + (node.module or "")
            found.add(base)
            prefix = f"{base}." if node.module else dots
            found.update(
                f"{prefix}{alias.name}" for alias in node.names if alias.name != "*"
            )
    return found


def parse_js_imports(text: str) -> set[str]:
    """Every module specifier in ``text``, unquoted and unresolved.

    Regex rather than a parser, for the reason in the module docstring: the
    regex cannot raise, cannot be defeated by a syntax error, and gets us the
    only thing we need.  It over-approximates -- a specifier inside a string or
    a comment becomes an edge -- which is the right direction to err, because
    an edge that resolves to nothing is dropped at the next step anyway.

    Dynamic ``import(spec)`` with a non-literal specifier is invisible.
    """
    found: set[str] = set()
    for pattern in (_JS_IMPORT, _JS_EXPORT, _JS_REQUIRE, _JS_DYNAMIC):
        found.update(pattern.findall(text))
    return found


def _python_candidates(dotted: str) -> tuple[str, ...]:
    """Paths a dotted module name could occupy, or ``()`` if the name is not one.

    Three shapes per name, and the third is the one people forget: importing
    ``a.b.c`` executes ``a/__init__.py`` and ``a/b/__init__.py`` on the way, so
    those files are edges of the import whether or not they were named.
    """
    parts = dotted.split(".") if dotted else []
    if not parts or not all(_IDENT.match(part) for part in parts):
        return ()
    base = "/".join(parts)
    out = [f"{base}.py", f"{base}.pyi", f"{base}/__init__.py"]
    parent = "/".join(parts[:-1])
    if parent:
        out.append(f"{parent}/__init__.py")
    return tuple(out)


def resolve_python(spec: str, importer: str) -> set[str]:
    """Candidate repo paths for one Python import target.

    ``importer`` is the file doing the importing, not its package: a file
    ``pkg/sub/mod.py`` belongs to package ``pkg.sub``, which is what the leading
    dots of a relative import count up from.  ``pkg/sub/__init__.py`` also
    belongs to ``pkg.sub`` -- ``__init__`` *is* its package -- so a single
    ``dirname`` serves both.

    A relative import that climbs above the repo root returns the empty set
    rather than an absolute path.  The graph only ever contains paths ctxpack
    already discovered, so a guess that escapes the root could not be matched
    anyway; returning nothing keeps that honest instead of accidental.
    """
    spec = spec.strip()
    if not spec:
        return set()

    if not spec.startswith("."):
        return set(_python_candidates(spec))

    level = len(spec) - len(spec.lstrip("."))
    rest = spec[level:]
    # ``__init__`` is its own package, so dirname is right for both a plain
    # module and a package's __init__.
    package = posixpath.dirname(importer)
    parts = package.split("/") if package else []
    climb = level - 1
    if climb > len(parts):
        return set()
    parts = parts[: len(parts) - climb]

    if not rest:
        # ``from . import x``: the bare dots name the package itself, which is
        # exactly one file. ``pkg/sub.py`` is not a candidate -- the sibling
        # module shape would be ``pkg/sub/x.py`` -- and the parent init is
        # already executed before this import is reached.
        return {f"{'/'.join(parts)}/__init__.py"} if parts else set()
    return set(_python_candidates(".".join(parts + rest.split("."))))


def resolve_js(spec: str, importer: str) -> set[str]:
    """Candidate repo paths for one JS/TS specifier.

    Relative specifiers are joined against the importer's directory and
    normalised with ``posixpath`` -- purely textually, no filesystem, so a
    symlink loop cannot hang us.  A specifier that climbs above the root
    returns nothing, for the same reason as in :func:`resolve_python`.

    Bare specifiers ("react", "components/Button") name packages as often as
    local files.  We emit root-anchored shapes so a real root-level file is
    still found, and deliberately do not guess ``node_modules``. When there is
    no root-level hit, :func:`build_graph` falls back to a filename-stem match,
    which is what makes ``import {x} from 'components/Button'`` work in a
    src-layout repo.

    A ``.d`` suffix is stripped from ``x.d`` so declaration files resolve to
    their implementation, and an explicit extension is trusted as written --
    except for TypeScript's ``NodeNext`` convention, below.

    **NodeNext extension substitution.** Most TypeScript projects that emit ESM
    write ``from '../agent/types.js'`` to mean ``../agent/types.ts``: the
    specifier carries the *runtime* extension while the source is TypeScript.
    Trusting the written extension therefore resolves to a file that does not
    exist, and the edge is silently dropped. Measured on one such repo, that
    turned 200-odd resolvable imports into 2 -- a graph that looked built and
    was empty. So a ``.js``/``.mjs``/``.cjs`` specifier also proposes the
    matching TypeScript sources.
    """
    spec = spec.strip()
    if not spec:
        return set()

    if spec.endswith(".d"):
        # A ``.d`` suffix marks the declaration half of a TS module pair; the
        # specifier names the pair, not the ``.d.ts`` file.
        spec = spec[: -len(".d")]

    def shapes(base: str) -> set[str]:
        """Every path ``base`` could name, honouring NodeNext substitution."""
        out: set[str] = {base}
        for runtime_ext, source_exts in _TS_EXTENSION_SUBSTITUTION.items():
            if base.endswith(runtime_ext):
                # The specifier names a module file, not a directory, so only
                # the sibling sources are proposed -- no ``/index.*`` variants.
                stem = base[: -len(runtime_ext)]
                out.update(f"{stem}{ext}" for ext in source_exts)
                return out
        if any(base.endswith(ext) for ext in JS_EXTS):
            return out  # an explicit TypeScript/JavaScript extension is trusted
        out.update(f"{base}{ext}" for ext in JS_EXTS)
        out.update(f"{base}/index{ext}" for ext in JS_EXTS)
        return out

    if spec.startswith("."):
        base = posixpath.normpath(posixpath.join(posixpath.dirname(importer), spec))
        if base in ("", ".") or base.startswith(".."):
            return set()
        return shapes(base)

    return shapes(spec)


def _specs_for(
    cache: dict[str, frozenset[str]],
    path: str,
    language: str,
    read: Callable[[str], str | None],
) -> frozenset[str]:
    """Parsed imports for one path, memoised for the life of the build."""
    cached = cache.get(path)
    if cached is not None:
        return cached
    try:
        text = read(path)
    except Exception:
        text = None
    if not text:
        parsed: frozenset[str] = frozenset()
    else:
        try:
            parsed = frozenset(
                parse_python_imports(text)
                if language == "py"
                else parse_js_imports(text)
            )
        except Exception:
            parsed = frozenset()
    cache[path] = parsed
    return parsed


def _edges_for(spec: str, language: str, importer: str, index: ModuleIndex) -> set[str]:
    """Resolve one import target against known paths only.

    Every fan-out path is capped, not just the stem fallback. The suffix branch
    (``index.lookup``) registers each path under its directory-stripped
    suffixes, so one bare specifier like ``pkg.mod`` can otherwise resolve to
    every ``*/pkg/mod.py`` in the tree. Uncapped that is two problems at once:
    the edge count goes quadratic in file count (800 files produced 320k edges),
    and -- worse -- reachability becomes confidently wrong, because
    ``svc0/app.py`` claims to import every other service's ``pkg/mod.py`` and
    ``--reach-weight`` then promotes all of them.
    """
    hits: set[str] = set()
    if language == "py":
        for candidate in resolve_python(spec, importer):
            hits.update(index.lookup(candidate))
        return _cap(hits, spec)

    relative = spec.startswith(".")
    for candidate in resolve_js(spec, importer):
        hits.update(index.lookup(candidate))
    if not relative and not hits:
        # No root-level file for this bare specifier. Try the filename stem,
        # which is how monorepos and src layouts actually spell local imports.
        name = spec.rsplit("/", 1)[-1]
        if name.endswith(".d"):
            name = name[: -len(".d")]
        hits.update(index.lookup_stem(name))
    return _cap(hits, spec)


def _cap(hits: set[str], spec: str) -> set[str]:
    """Bound fan-out per specifier, keeping a deterministic subset."""
    if len(hits) <= MAX_EDGES_PER_SPEC:
        return hits
    # Deterministic so two runs over the same tree agree, and alphabetical so
    # the truncation is at least reproducible rather than dict-order dependent.
    return set(sorted(hits)[:MAX_EDGES_PER_SPEC])


def build_graph(
    paths: list[str],
    read: Callable[[str], str | None],
) -> dict[str, set[str]]:
    """Import adjacency over ``paths``, restricted to files we know exist.

    ``read`` takes a path and returns its text or ``None``; :func:`walk.read_text`
    adapted to candidates fits. It may raise -- it is the caller's I/O -- and is
    called inside a guard anyway, because one unreadable file in three thousand
    is not worth an exception.

    Every input path gets an entry even if it imports nothing, is not source, or
    could not be parsed. A node with an empty set is information: it is a leaf,
    or a file we could not read, and callers can tell those apart from absence.

    Import resolution is deliberately not transitive, and re-exports through
    ``__init__.py`` are followed one hop only. A graph good enough to rank by
    reachability is not a graph good enough to be trusted for refactoring.
    """
    index = build_index(paths)
    graph: dict[str, set[str]] = {path: set() for path in paths}
    cache: dict[str, frozenset[str]] = {}

    for path in paths:
        language = _language(path)
        if language is None:
            continue
        specs = _specs_for(cache, path, language, read)
        if not specs:
            continue
        edges: set[str] = set()
        for spec in sorted(specs):
            edges.update(_edges_for(spec, language, path, index))
        edges.discard(path)
        if edges:
            graph[path] = edges
    return graph


def find_entrypoints(paths: list[str]) -> set[str]:
    """Files that plausibly start the program, by filename alone.

    This is a heuristic and a weak one. There is no manifest parsing, no
    ``package.json`` ``main``, no ``[project.scripts]``, no detection of a
    framework's convention -- so a project whose only entrypoint is called
    ``run_thing.py`` has none, and ``__init__`` makes the set generous enough
    that package-heavy repos seed the BFS from many nodes at once.

    Files whose extension says they are not code are excluded even when their
    stem matches: ``docs/index.md`` is not where a program starts, and letting
    it in would put a documentation file at hop distance zero. The exclusion is
    by extension, not by parseability -- a Ruby or shell ``bootstrap`` is a
    perfectly good entrypoint for reachability purposes, it just has no outgoing
    edges because we do not parse Ruby or shell.

    An empty result is normal and must not be read as "everything is an
    entrypoint" -- see :func:`hop_distance`.
    """
    found: set[str] = set()
    for path in paths:
        ext = posixpath.splitext(path)[1].lower()
        if ext in DOC_EXTS or ext in CONFIG_EXTS:
            continue
        if _stem(path).lower() in ENTRYPOINT_STEMS or path in _EXTRA_ENTRYPOINTS:
            found.add(path)
    return found


def reachable_from(graph: Graph, entrypoints: Iterable[str]) -> set[str]:
    """Every node reachable from ``entrypoints``, including distance zero.

    Entry points absent from ``graph`` are ignored rather than added: the graph
    is the truth about what exists, and an entrypoint that was filtered out of
    the pack should not drag its neighbours back in.
    """
    seen: set[str] = set()
    queue = deque(node for node in entrypoints if node in graph)
    seen.update(queue)
    while queue:
        node = queue.popleft()
        for neighbour in graph.get(node, ()):
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return seen


def hop_distance(graph: Graph, entrypoints: Iterable[str]) -> dict[str, int]:
    """BFS distance from the nearest entrypoint, per node.

    Unreachable nodes are **absent from the dict**, not present with ``-1``.
    Absence is the honest encoding: a file nobody reaches is not "minus one
    hops away", it is a fact about a different set. Callers that treat a missing
    key as zero -- which is what ``dict.get(path, 0)`` does -- silently promote
    dead code to top rank, so handle the miss explicitly.

    No entrypoints means an empty dict. That is the "no signal" answer, and it
    is deliberately *not* "every file is at distance zero".
    """
    distances: dict[str, int] = {}
    frontier = [node for node in entrypoints if node in graph]
    for node in frontier:
        distances[node] = 0
    depth = 0
    while frontier:
        depth += 1
        following: list[str] = []
        for node in frontier:
            for neighbour in graph.get(node, ()):
                if neighbour in graph and neighbour not in distances:
                    distances[neighbour] = depth
                    following.append(neighbour)
        frontier = following
    return distances


def boost(scored: list[Scored], distances: dict[str, int], weight: float) -> list[Scored]:
    """Return ``scored`` with a reachability bonus folded into each score.

    :class:`~ctxpack.rank.Scored` is frozen, so this builds new objects and
    never mutates the input -- rank order is the caller's to decide, and a
    function that re-sorted it would quietly change what "best" means. Input
    order is preserved exactly; only scores change.

    The bonus is ``weight * HOP_DECAY ** hops``, so the entrypoint itself gets
    the full weight, a direct import most of it, and a fourth-degree relative
    about a tenth. Files with no entry in ``distances`` are returned untouched:
    an empty dict is a no-op, not "everything is reachable".

    ``weight`` comes from a CLI flag, so a non-finite value is a user error and
    raises :class:`~ctxpack.errors.CtxpackError`. A NaN would poison every score
    downstream in a way that surfaces much later and nowhere useful.
    """
    if not math.isfinite(weight):
        raise CtxpackError(f"reachable weight must be finite, got {weight!r}")

    out: list[Scored] = []
    for item in scored:
        hops = distances.get(item.path)
        if hops is None:
            out.append(item)
            continue
        gain = round(weight * HOP_DECAY**hops, 4)
        out.append(
            Scored(
                candidate=item.candidate,
                score=round(item.score + gain, 4),
                signals=(
                    *item.signals,
                    Signal("reachable", gain, f"{hops} hops"),
                ),
            )
        )
    return out


@dataclass(frozen=True)
class DepAnalysis:
    """Graph, entrypoints and distances for one file set, computed together.

    Only here because the three are always wanted together and the pairing is
    easy to get wrong at the call site: distances must be built from *these*
    entrypoints over *this* graph, and callers that recompute one half get
    results that look fine and are meaningless.
    """

    graph: Graph
    entrypoints: frozenset[str]
    distances: dict[str, int]

    def reachable(self) -> set[str]:
        """Every node with a distance. Identical to ``distances.keys()``."""
        return set(self.distances)

    def boost(self, scored: list[Scored], weight: float) -> list[Scored]:
        """Shorthand for :func:`boost` with these distances."""
        return boost(scored, self.distances, weight)

    def __bool__(self) -> bool:
        return bool(self.distances)


def analyse(paths: list[str], read: Callable[[str], str | None]) -> DepAnalysis:
    """Build the graph, find entrypoints and compute distances in one pass."""
    graph = build_graph(paths, read)
    entrypoints = find_entrypoints(paths) & graph.keys()
    return DepAnalysis(
        graph=graph,
        entrypoints=frozenset(entrypoints),
        distances=hop_distance(graph, entrypoints),
    )
