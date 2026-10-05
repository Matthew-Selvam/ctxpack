"""Import graphs, resolution rules and the reachability bonus."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ctxpack.deps import (
    HOP_DECAY,
    DepAnalysis,
    ModuleIndex,
    analyse,
    boost,
    build_graph,
    build_index,
    find_entrypoints,
    hop_distance,
    parse_js_imports,
    parse_python_imports,
    reachable_from,
    resolve_js,
    resolve_python,
)
from ctxpack.errors import CtxpackError
from ctxpack.rank import Scored, Signal, rank_all
from ctxpack.walk import Candidate, read_text


def write(root: Path, rel: str, text: str) -> str:
    """Write a real file under ``root`` and return its repo-relative path."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return rel


def reader(root: Path):
    """A ``read`` callable over a real tree, matching what ctxpack would pass."""

    def read(path: str) -> str | None:
        try:
            return (root / path).read_text(encoding="utf-8")
        except OSError:
            return None

    return read


def make_scored(path: str, score: float = 1.0) -> Scored:
    """A minimal Scored with a single base signal, independent of rank's rules."""
    candidate = Candidate(
        path=path, abs_path=Path("/tmp") / path, size=1000, is_binary=False
    )
    return Scored(candidate=candidate, score=score, signals=(Signal("base", score),))


# --------------------------------------------------------------------------
# parse_python_imports
# --------------------------------------------------------------------------


def test_python_dotted_import():
    assert parse_python_imports("import a.b.c\n") == {"a.b.c"}


def test_python_aliased_import_drops_the_alias():
    assert parse_python_imports("import a.b as x\n") == {"a.b"}


def test_python_multiple_imports():
    assert parse_python_imports("import a, b.c\nimport d\n") == {"a", "b.c", "d"}


def test_python_from_import_keeps_module_and_symbol():
    assert parse_python_imports("from a.b import c, d\n") == {"a.b", "a.b.c", "a.b.d"}


def test_python_star_import_adds_only_the_module():
    assert parse_python_imports("from a.b import *\n") == {"a.b"}


def test_python_relative_from_bare_dot():
    assert parse_python_imports("from . import x\n") == {".", ".x"}


def test_python_relative_from_module():
    assert parse_python_imports("from .mod import y\n") == {".mod", ".mod.y"}


def test_python_relative_from_parent_package():
    assert parse_python_imports("from ..pkg import z\n") == {"..pkg", "..pkg.z"}


def test_python_relative_from_doubly_dotted():
    assert parse_python_imports("from ... import q\n") == {"...", "...q"}


def test_python_imports_inside_function_and_class_body():
    text = "class C:\n    import a.b\n\ndef f():\n    from . import c\n"
    assert parse_python_imports(text) == {"a.b", ".", ".c"}


def test_python_imports_in_try_block_found():
    text = "try:\n    import json\nexcept ImportError:\n    json = None\n"
    assert parse_python_imports(text) == {"json"}


def test_python_future_import_parsed_but_unresolvable():
    # __future__ is a real import; it just has no repo path. Resolution drops
    # it, which is the correct outcome, not a miss.
    assert parse_python_imports("from __future__ import annotations\n") == {
        "__future__",
        "__future__.annotations",
    }


def test_python_no_imports_is_empty():
    assert parse_python_imports("x = 1\ndef f():\n    return x\n") == set()


def test_python_empty_text_is_empty():
    assert parse_python_imports("") == set()


def test_malformed_python_returns_empty_not_raises():
    assert parse_python_imports("def broken(:\n    pass\n") == set()
    assert parse_python_imports("import a.b.c\nclass ???\n") == set()


def test_hostile_python_does_not_raise():
    samples = [
        "\x00\x01\x02",
        "()" * 500,
        "from . import " + "." * 200 + "x",
        "import " + "." * 500,
        "\n".join("import a.b" for _ in range(5000)),
    ]
    for sample in samples:
        assert isinstance(parse_python_imports(sample), set)


# --------------------------------------------------------------------------
# parse_js_imports
# --------------------------------------------------------------------------


def test_js_default_import():
    assert parse_js_imports("import x from 'y';\n") == {"y"}


def test_js_bare_side_effect_import():
    assert parse_js_imports("import 'y';\n") == {"y"}


def test_js_require_call():
    assert parse_js_imports("const x = require('y');\n") == {"y"}


def test_js_export_star_from():
    assert parse_js_imports("export * from 'y';\n") == {"y"}


def test_js_export_named_from():
    assert parse_js_imports("export {a, b as c} from 'y';\n") == {"y"}


def test_js_all_forms_at_once():
    text = (
        "import x from 'a';\n"
        "import 'b';\n"
        "const c = require('c');\n"
        "export * from 'd';\n"
        "export {e} from 'e';\n"
    )
    assert parse_js_imports(text) == {"a", "b", "c", "d", "e"}


def test_js_type_only_import_found():
    assert parse_js_imports("import type {A} from './types';\n") == {"./types"}


def test_js_dynamic_import_with_literal_found():
    assert parse_js_imports("const m = await import('./lazy');\n") == {"./lazy"}


def test_js_ignores_non_literal_dynamic_import():
    assert parse_js_imports("const m = await import(name);\n") == set()


def test_js_double_quotes_accepted():
    assert parse_js_imports('import x from "y";\n') == {"y"}


def test_garbage_js_returns_empty_not_raises():
    assert parse_js_imports("}}}{{{ \x00\xff not javascript at all\n") == set()
    assert parse_js_imports("") == set()


def test_js_unterminated_string_does_not_raise():
    assert isinstance(parse_js_imports("import x from 'unterminated\n"), set)


def test_js_import_not_at_line_start_is_not_an_edge():
    # The word "import" in prose must not become an edge.
    assert parse_js_imports("// you can import things from 'somewhere'\n") == set()


# --------------------------------------------------------------------------
# resolve_python
# --------------------------------------------------------------------------


def test_resolve_python_module_file():
    assert "a/b/c.py" in resolve_python("a.b.c", "main.py")


def test_resolve_python_package_dir():
    assert "a/b/c/__init__.py" in resolve_python("a.b.c", "main.py")


def test_resolve_python_includes_parent_init():
    assert "a/b/__init__.py" in resolve_python("a.b.c", "main.py")


def test_resolve_python_relative_same_package():
    got = resolve_python(".mod", "pkg/sub/mod.py")
    assert "pkg/sub/mod.py" in got
    assert "pkg/sub/__init__.py" in got


def test_resolve_python_relative_from_package_init():
    # pkg/__init__.py belongs to package pkg, so `.mod` is pkg/mod.py.
    assert "pkg/mod.py" in resolve_python(".mod", "pkg/__init__.py")


def test_resolve_python_relative_bare_dot_is_the_package_init():
    assert resolve_python(".", "pkg/sub/mod.py") == {"pkg/sub/__init__.py"}


def test_resolve_python_relative_climbs_one_level():
    assert "pkg/__init__.py" in resolve_python("..pkg", "pkg/sub/mod.py")


def test_resolve_python_relative_from_package_level():
    got = resolve_python("..sibling", "pkg/sub/mod.py")
    assert "pkg/sibling.py" in got


def test_resolve_python_relative_above_root_is_empty():
    assert resolve_python("...", "mod.py") == set()
    assert resolve_python("....", "pkg/mod.py") == set()


def test_resolve_python_relative_from_top_level_file():
    assert resolve_python(".x", "mod.py") == {"x.py", "x.pyi", "x/__init__.py"}


def test_resolve_python_empty_and_junk_specs():
    assert resolve_python("", "pkg/mod.py") == set()
    assert resolve_python("   ", "pkg/mod.py") == set()
    assert resolve_python("a..b", "pkg/mod.py") == set()
    assert resolve_python("a//b", "pkg/mod.py") == set()
    assert resolve_python("a b", "pkg/mod.py") == set()
    assert resolve_python("1abc", "pkg/mod.py") == set()


def test_resolve_python_relative_junk_rest_is_empty():
    assert resolve_python(".1bad", "pkg/mod.py") == set()


# --------------------------------------------------------------------------
# resolve_js
# --------------------------------------------------------------------------


def test_resolve_js_relative():
    assert "src/app/util.ts" in resolve_js("./util", "src/app/main.ts")


def test_resolve_js_relative_parent():
    assert "src/store/index.ts" in resolve_js("../store", "src/app/main.ts")


def test_resolve_js_relative_deep_parent():
    assert "src/lib/deep.ts" in resolve_js("../../lib/deep", "src/app/x/main.ts")


def test_resolve_js_relative_tries_every_extension():
    got = resolve_js("./util", "src/app/main.ts")
    for ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"):
        assert f"src/app/util{ext}" in got


def test_resolve_js_relative_tries_index_files():
    assert "src/app/store/index.ts" in resolve_js("./store", "src/app/main.ts")


def test_resolve_js_explicit_extension_is_trusted():
    """An unambiguous TypeScript extension names exactly one file.

    ``.jsx`` is deliberately absent: NodeNext maps it to ``.tsx``, so it is
    ambiguous by design. See ``test_jsx_resolves_to_tsx_under_nodenext``.
    """
    for ext in (".ts", ".tsx"):
        assert resolve_js(f"./util{ext}", "src/app/main.ts") == {f"src/app/util{ext}"}


def test_resolve_js_nodenext_substitutes_typescript_source():
    """``from './x.js'`` in a compiled-ESM repo means ``./x.ts``.

    Regression with real consequences: trusting the written extension turned
    ~670 resolvable imports on one TypeScript repo into 2, so the graph was
    empty while appearing to have been built.
    """
    got = resolve_js("./util.js", "src/app/main.ts")
    assert "src/app/util.ts" in got
    assert "src/app/util.tsx" in got
    assert "src/app/util.js" in got  # the literal reading is still offered
    # No directory variants: the specifier names a module, not a directory.
    assert not any("/index." in path for path in got)


def test_resolve_js_nodenext_variants():
    assert "src/app/util.mts" in resolve_js("./util.mjs", "src/app/main.ts")
    assert "src/app/util.cts" in resolve_js("./util.cjs", "src/app/main.ts")
    assert "src/app/util.d.ts" in resolve_js("./util.js", "src/app/main.ts")


def test_nodenext_repo_builds_a_connected_graph(tmp_path):
    """End-to-end: a NodeNext-style repo must produce real edges.

    Guards the failure mode where the graph is built successfully but contains
    almost nothing, which no assertion on `resolve_js` alone would catch.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "index.ts").write_text(
        "export { run } from './server.js';\n", encoding="utf-8"
    )
    (tmp_path / "src" / "server.ts").write_text(
        "import { store } from './store.js';\n", encoding="utf-8"
    )
    (tmp_path / "src" / "store.ts").write_text("export const store = 1;\n", encoding="utf-8")

    paths = ["src/index.ts", "src/server.ts", "src/store.ts"]
    graph = build_graph(paths, lambda p: (tmp_path / p).read_text(encoding="utf-8"))

    assert graph["src/index.ts"] == {"src/server.ts"}
    assert graph["src/server.ts"] == {"src/store.ts"}
    distances = hop_distance(graph, {"src/index.ts"})
    assert distances == {"src/index.ts": 0, "src/server.ts": 1, "src/store.ts": 2}


def test_resolve_js_declaration_suffix_stripped():
    assert "src/app/types.ts" in resolve_js("./types.d", "src/app/main.ts")


def test_resolve_js_bare_specifier_root_anchored():
    assert "widgets.ts" in resolve_js("widgets", "src/main.ts")
    assert "widgets/index.ts" in resolve_js("widgets", "src/main.ts")


def test_resolve_js_relative_above_root_is_empty():
    assert resolve_js("../../outside", "src/main.ts") == set()


def test_resolve_js_self_directory_specifier():
    assert "src/app/index.ts" in resolve_js(".", "src/app/main.ts")


def test_resolve_js_empty_specifier():
    assert resolve_js("", "src/main.ts") == set()
    assert resolve_js("   ", "src/main.ts") == set()


# --------------------------------------------------------------------------
# build_graph
# --------------------------------------------------------------------------


@pytest.fixture
def py_repo(tmp_path: Path) -> Path:
    """A small real Python tree with a known import shape."""
    root = tmp_path / "pyrepo"
    write(root, "main.py", "from pkg.a import helper\nimport pkg.b\n")
    write(root, "pkg/__init__.py", "from .c import shared\n")
    write(root, "pkg/a.py", "from .c import shared\nfrom .. import main\n")
    write(root, "pkg/b.py", "import pkg.c\n")
    write(root, "pkg/c.py", "SHARED = 1\n")
    write(root, "pkg/sub/__init__.py", "")
    write(root, "pkg/sub/d.py", "from .. import c\nfrom . import e\n")
    write(root, "pkg/sub/e.py", "E = 2\n")
    write(root, "orphan.py", "X = 1\n")
    write(root, "README.md", "# docs, not source\n")
    return root


def test_graph_resolves_dotted_python_imports(py_repo: Path):
    paths = ["main.py", "pkg/__init__.py", "pkg/a.py", "pkg/c.py", "pkg/b.py"]
    graph = build_graph(paths, reader(py_repo))
    # `from pkg.a import helper` yields pkg.a (pkg/a.py) and pkg.a.helper (no
    # such file), plus the parent package init Python executes on the way.
    # `import pkg.b` names pkg/b.py directly.
    assert graph["main.py"] == {"pkg/a.py", "pkg/__init__.py", "pkg/b.py"}


def test_graph_python_package_init_is_an_edge(py_repo: Path):
    paths = ["pkg/__init__.py", "pkg/c.py", "pkg/b.py"]
    graph = build_graph(paths, reader(py_repo))
    # `from .c import shared` names pkg.c and pkg.c.shared; the third candidate
    # shape is the importer's own package init, which self-edges and is dropped.
    assert graph["pkg/__init__.py"] == {"pkg/c.py"}


def test_graph_python_relative_imports_resolve(py_repo: Path):
    paths = ["pkg/sub/d.py", "pkg/c.py", "pkg/sub/e.py"]
    graph = build_graph(paths, reader(py_repo))
    assert "pkg/c.py" in graph["pkg/sub/d.py"]
    assert "pkg/sub/e.py" in graph["pkg/sub/d.py"]


def test_graph_includes_every_input_path(py_repo: Path):
    paths = ["main.py", "orphan.py", "README.md", "pkg/c.py"]
    graph = build_graph(paths, reader(py_repo))
    assert set(graph) == set(paths)


def test_graph_isolated_node_present_with_empty_set(py_repo: Path):
    graph = build_graph(["orphan.py", "README.md"], reader(py_repo))
    assert graph == {"orphan.py": set(), "README.md": set()}


def test_graph_drops_unresolvable_imports(tmp_path: Path):
    root = tmp_path / "drop"
    write(root, "a.py", "import thirdparty\nfrom . import nothing\n")
    graph = build_graph(["a.py"], reader(root))
    assert graph["a.py"] == set()


def test_graph_drops_self_import(py_repo: Path):
    graph = build_graph(["pkg/a.py", "pkg/c.py"], reader(py_repo))
    assert "pkg/a.py" not in graph["pkg/a.py"]


def test_graph_survives_unreadable_file(tmp_path: Path):
    root = tmp_path / "unread"
    write(root, "main.py", "import broken\nimport other\n")
    write(root, "broken.py", "import other\n")
    write(root, "other.py", "X = 1\n")

    def read(path: str) -> str | None:
        if path == "main.py":
            return "import broken\n"
        return (root / path).read_text(encoding="utf-8")

    graph = build_graph(["main.py", "broken.py", "other.py"], read)
    assert graph["main.py"] == {"broken.py"}
    assert graph["broken.py"] == {"other.py"}


def test_graph_survives_raising_reader(py_repo: Path):
    def read(path: str) -> str | None:
        raise OSError("nope")

    graph = build_graph(["main.py", "orphan.py"], read)
    assert graph == {"main.py": set(), "orphan.py": set()}


def test_graph_survives_malformed_source(py_repo: Path):
    root = py_repo
    write(root, "pkg/c.py", "def broken(:\n")
    graph = build_graph(["main.py", "pkg/a.py", "pkg/c.py", "pkg/b.py"], reader(root))
    # The malformed file still parses to nothing, but the rest of the graph is
    # intact and the walk did not abort.
    assert "pkg/a.py" in graph["main.py"]
    assert graph["pkg/c.py"] == set()


def test_graph_skips_non_source_files(tmp_path: Path):
    root = tmp_path / "docs"
    write(root, "README.md", "import pkg\nsee `import pkg`\n")
    write(root, "notes.py", "MARKDOWN = True\n")
    graph = build_graph(["README.md", "notes.py"], reader(root))
    assert graph["README.md"] == set()


def test_graph_resolves_src_layout_python(py_repo: Path):
    root = py_repo
    write(root, "src2/app.py", "from pkg.c import SHARED\n")
    write(root, "src2/pkg/__init__.py", "")
    write(root, "src2/pkg/c.py", "SHARED = 1\n")
    graph = build_graph(
        ["src2/app.py", "src2/pkg/__init__.py", "src2/pkg/c.py"], reader(root)
    )
    # `from pkg.c import SHARED` names pkg.c, and the directory-stripped index
    # resolves it under the src/ layout -- plus pkg/__init__.py on the way.
    assert graph["src2/app.py"] == {"src2/pkg/c.py", "src2/pkg/__init__.py"}


def test_graph_resolves_relative_js_and_index(tmp_path: Path):
    root = tmp_path / "js"
    write(root, "src/main.ts", "import {a} from './util'\nimport s from './store'\n")
    write(root, "src/util.ts", "export const a = 1\n")
    write(root, "src/store/index.ts", "export default {}\n")
    graph = build_graph(
        ["src/main.ts", "src/util.ts", "src/store/index.ts"], reader(root)
    )
    assert graph["src/main.ts"] == {"src/util.ts", "src/store/index.ts"}


def test_graph_resolves_bare_js_via_stem(tmp_path: Path):
    root = tmp_path / "bare"
    write(root, "src/main.ts", "import {x} from 'shared'\n")
    write(root, "src/feature/shared.ts", "export const x = 1\n")
    graph = build_graph(["src/main.ts", "src/feature/shared.ts"], reader(root))
    assert graph["src/main.ts"] == {"src/feature/shared.ts"}


def test_graph_bare_js_match_is_capped(tmp_path: Path):
    root = tmp_path / "capped"
    write(root, "src/main.ts", "import 'shared'\n")
    for i in range(12):
        write(root, f"src/f{i}/shared.ts", "export const x = 1\n")
    paths = ["src/main.ts"] + [f"src/f{i}/shared.ts" for i in range(12)]
    graph = build_graph(paths, reader(root))
    assert 0 < len(graph["src/main.ts"]) <= 8


def test_graph_drops_node_modules_imports(tmp_path: Path):
    root = tmp_path / "nm"
    write(root, "src/main.ts", "import React from 'react'\n")
    write(root, "node_modules/react/index.js", "module.exports = {}\n")
    graph = build_graph(["src/main.ts", "node_modules/react/index.js"], reader(root))
    # We do not walk into node_modules by default, but if a path is handed to
    # us the graph stays honest rather than inventing an edge to a package root.
    assert "node_modules/react/index.js" in graph["src/main.ts"]


def test_graph_is_deterministic(py_repo: Path):
    paths = sorted(build_graph_from(py_repo))
    first = build_graph(paths, reader(py_repo))
    second = build_graph(paths, reader(py_repo))
    assert first == second


def build_graph_from(root: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            try:
                found[rel] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                found[rel] = ""
    return found


def test_graph_duplicate_paths_collapse(py_repo: Path):
    graph = build_graph(["main.py", "main.py", "orphan.py"], reader(py_repo))
    assert set(graph) == {"main.py", "orphan.py"}


def test_build_graph_accepts_walk_read_text(py_repo: Path):
    """The documented integration: walk's Candidate -> text -> build_graph."""
    from ctxpack.walk import discover

    discovery = discover(py_repo)
    paths = [c.path for c in discovery.files]
    texts = {c.path: read_text(c) for c in discovery.files}

    graph = build_graph(paths, lambda p: texts[p])
    assert "pkg/a.py" in graph["main.py"]
    assert graph["README.md"] == set()


# --------------------------------------------------------------------------
# build_index / ModuleIndex
# --------------------------------------------------------------------------


def test_index_lookup_exact_and_stripped():
    index = build_index(["src/pkg/mod.py", "other.py"])
    assert index.lookup("src/pkg/mod.py") == ("src/pkg/mod.py",)
    assert index.lookup("pkg/mod.py") == ("src/pkg/mod.py",)


def test_index_does_not_strip_the_last_directory_component():
    index = build_index(["src/pkg/mod.py"])
    # "mod.py" alone would make `import mod` resolve to any same-named leaf.
    assert index.lookup("mod.py") == ()


def test_index_stem_lookup():
    index = build_index(["a/thing.ts", "b/thing.ts", "c/other.ts"])
    assert index.lookup_stem("thing") == ("a/thing.ts", "b/thing.ts")
    assert index.lookup_stem("missing") == ()


def test_index_is_constructible_empty():
    index = ModuleIndex()
    assert index.lookup("x") == ()
    assert index.lookup_stem("x") == ()
    assert len(index) == 0


def test_index_dedupes_repeated_paths():
    assert build_index(["a.py", "a.py"]).lookup("a.py") == ("a.py",)


# --------------------------------------------------------------------------
# find_entrypoints
# --------------------------------------------------------------------------


def test_entrypoints_by_stem(tmp_path: Path):
    paths = ["main.py", "src/index.ts", "app/app.py", "src/server.js", "lib/x.py"]
    assert find_entrypoints(paths) == {
        "main.py",
        "src/index.ts",
        "app/app.py",
        "src/server.js",
    }


def test_entrypoints_bootstrap_and_entry():
    assert find_entrypoints(["bin/bootstrap.rb", "bin/entry.sh"]) == {
        "bin/bootstrap.rb",
        "bin/entry.sh",
    }


def test_entrypoints_include_package_init():
    assert "pkg/__init__.py" in find_entrypoints(["pkg/__init__.py"])


def test_entrypoints_case_insensitive_stem():
    assert "src/MAIN.ts" in find_entrypoints(["src/MAIN.ts"])


def test_entrypoints_exclude_non_source():
    assert find_entrypoints(["docs/index.md", "README.md", "index.txt"]) == set()


def test_entrypoints_exclude_ordinary_modules():
    assert find_entrypoints(["src/handlers.py", "src/util.py"]) == set()


def test_entrypoints_may_be_empty():
    assert find_entrypoints([]) == set()


# --------------------------------------------------------------------------
# reachable_from / hop_distance
# --------------------------------------------------------------------------


@pytest.fixture
def known_graph() -> dict[str, set[str]]:
    """A diamond plus a branch plus an unreachable island.

        main -> a -> c -> e
             -> b -> c
             -> f          (branch, depth 2)
        island -> g          (unreachable from main)
        alone                (no edges at all)
    """
    return {
        "main": {"a", "b", "f"},
        "a": {"c"},
        "b": {"c"},
        "c": {"e"},
        "f": set(),
        "e": set(),
        "island": {"g"},
        "g": set(),
        "alone": set(),
    }


def test_reachable_from_follows_transitive_edges(known_graph):
    assert reachable_from(known_graph, ["main"]) == {"main", "a", "b", "c", "e", "f"}


def test_reachable_from_ignores_unknown_entrypoints(known_graph):
    assert reachable_from(known_graph, ["nope"]) == set()


def test_reachable_from_multiple_entrypoints(known_graph):
    assert reachable_from(known_graph, ["main", "island"]) == {
        "main",
        "a",
        "b",
        "c",
        "e",
        "f",
        "island",
        "g",
    }


def test_hop_distance_exact_values_on_known_graph(known_graph):
    distances = hop_distance(known_graph, ["main"])
    assert distances == {"main": 0, "a": 1, "b": 1, "f": 1, "c": 2, "e": 3}


def test_hop_distance_omits_unreachable_nodes(known_graph):
    distances = hop_distance(known_graph, ["main"])
    assert "island" not in distances
    assert "g" not in distances
    assert "alone" not in distances


def test_hop_distance_uses_shortest_path_in_a_diamond(known_graph):
    assert hop_distance(known_graph, ["main"])["c"] == 2


def test_hop_distance_second_entrypoint_wins(known_graph):
    distances = hop_distance(known_graph, ["main", "c"])
    assert distances["c"] == 0
    assert distances["e"] == 1
    assert distances["a"] == 1


def test_hop_distance_no_entrypoints_is_empty(known_graph):
    # Not "everything is at distance zero".
    assert hop_distance(known_graph, []) == {}
    assert hop_distance(known_graph, ["missing"]) == {}


def test_hop_distance_never_uses_negative_one(known_graph):
    assert -1 not in hop_distance(known_graph, ["main"]).values()


def test_hop_distance_tolerates_missing_graph_keys():
    graph = {"main": {"ghost"}}
    assert hop_distance(graph, ["main"]) == {"main": 0}


def test_hop_distance_tolerates_edges_to_absent_nodes():
    graph = {"main": {"a"}, "a": set()}
    assert hop_distance(graph, ["main"]) == {"main": 0, "a": 1}


def test_hop_distance_survives_a_cycle():
    graph = {"a": {"b"}, "b": {"c"}, "c": {"a"}}
    assert hop_distance(graph, ["a"]) == {"a": 0, "b": 1, "c": 2}


def test_hop_distance_empty_graph():
    assert hop_distance({}, []) == {}
    assert hop_distance({}, ["a"]) == {}


def test_hop_distance_empty_entrypoints_is_empty(known_graph):
    assert hop_distance(known_graph, set()) == {}


# --------------------------------------------------------------------------
# boost
# --------------------------------------------------------------------------


def test_boost_preserves_input_order():
    scored = [make_scored(p) for p in ["c", "a", "b"]]
    out = boost(scored, {"a": 0, "b": 1, "c": 2}, 2.0)
    assert [s.path for s in out] == ["c", "a", "b"]


def test_boost_does_not_mutate_inputs():
    original = make_scored("a", 1.0)
    before = (original.score, original.signals)
    boost([original], {"a": 0}, 3.0)
    assert (original.score, original.signals) == before


def test_boost_appends_a_signal_with_detail():
    out = boost([make_scored("a")], {"a": 2}, 3.0)[0]
    signal = out.signals[-1]
    assert signal.key == "reachable"
    assert "2 hops" in signal.detail
    assert out.signals[0].key == "base"


def test_boost_weight_applies_at_distance_zero():
    out = boost([make_scored("a", 0.0)], {"a": 0}, 4.0)[0]
    assert out.score == pytest.approx(4.0)


def test_boost_decays_with_distance():
    scored = [make_scored("d0"), make_scored("d1"), make_scored("d2")]
    out = boost(scored, {"d0": 0, "d1": 1, "d2": 2}, 3.0)
    scores = [s.score for s in out]
    assert scores[0] > scores[1] > scores[2]
    assert scores[1] - 1.0 == pytest.approx(3.0 * HOP_DECAY)
    assert scores[2] - 1.0 == pytest.approx(3.0 * HOP_DECAY**2)


def test_boost_leaves_files_without_a_distance_untouched():
    original = make_scored("orphan", 1.5)
    out = boost([original], {"other": 0}, 5.0)[0]
    assert out.score == 1.5
    assert out.signals == original.signals


def test_boost_with_empty_distances_is_a_noop():
    scored = [make_scored("a", 1.0), make_scored("b", 2.0)]
    out = boost(scored, {}, 5.0)
    assert [s.score for s in out] == [1.0, 2.0]
    assert [s.signals for s in out] == [s.signals for s in scored]


def test_boost_empty_input():
    assert boost([], {"a": 0}, 1.0) == []


def test_boost_rejects_non_finite_weight():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(CtxpackError):
            boost([make_scored("a")], {"a": 0}, bad)


def test_boost_accepts_zero_weight():
    out = boost([make_scored("a", 1.0)], {"a": 3}, 0.0)[0]
    assert out.score == 1.0
    # The signal is still recorded, so `explain` shows the graph was consulted.
    assert out.signals[-1].key == "reachable"


def test_boost_negative_weight_inverts():
    scored = boost([make_scored("a", 1.0), make_scored("b", 1.0)], {"a": 0, "b": 1}, -2.0)
    assert scored[0].score < scored[1].score


def test_boost_on_real_ranking_stays_sane(project: Path):
    from ctxpack.walk import discover

    discovery = discover(project)
    ranked = rank_all(discovery.files)
    texts = {c.path: read_text(c) for c in discovery.files}
    graph = build_graph([c.path for c in discovery.files], lambda p: texts[p])
    distances = hop_distance(graph, find_entrypoints([c.path for c in discovery.files]))

    out = boost(ranked, distances, 3.0)
    assert len(out) == len(ranked)
    for item in out:
        original = next(s for s in ranked if s.path == item.path)
        assert item.score >= original.score
        if item.path in distances:
            assert item.signals[-1].key == "reachable"


def test_boost_reorders_ranking_when_used_for_ranking():
    """The point of the module: reachability should change the order."""
    scored = [make_scored("leaf", 4.0), make_scored("root", 3.0)]
    out = boost(scored, {"root": 0, "leaf": 5}, 4.0)
    out.sort(key=lambda s: (-s.score, s.path))
    assert out[0].path == "root"


# --------------------------------------------------------------------------
# analyse / DepAnalysis
# --------------------------------------------------------------------------


def test_analyse_ties_the_three_halves_together(py_repo: Path):
    paths = ["main.py", "pkg/a.py", "pkg/b.py", "pkg/c.py", "orphan.py", "README.md"]
    result = analyse(paths, reader(py_repo))
    assert isinstance(result, DepAnalysis)
    assert result.entrypoints == frozenset({"main.py"})
    assert result.distances["main.py"] == 0
    assert result.distances["pkg/a.py"] == 1
    assert result.distances["pkg/c.py"] == 2
    assert "orphan.py" not in result.distances


def test_analyse_reachable_matches_distances(py_repo: Path):
    paths = ["main.py", "pkg/a.py", "pkg/c.py", "orphan.py"]
    result = analyse(paths, reader(py_repo))
    assert result.reachable() == set(result.distances)


def test_analyse_on_a_real_tree(py_repo: Path):
    files = build_graph_from(py_repo)
    result = analyse(sorted(files), lambda p: files[p])
    assert result.entrypoints
    assert result.distances["main.py"] == 0


def test_analyse_no_entrypoints_is_falsey(py_repo: Path):
    result = analyse(["orphan.py", "README.md"], reader(py_repo))
    assert not result
    assert result.distances == {}


def test_analyse_boost_shorthand(py_repo: Path):
    paths = ["main.py", "pkg/a.py", "pkg/c.py"]
    result = analyse(paths, reader(py_repo))
    out = result.boost([make_scored("pkg/c.py")], 2.0)
    assert out[0].score > 1.0


def test_analyse_drops_entrypoints_absent_from_the_graph():
    result = analyse(["orphan.py"], lambda p: "import nothing_here\n")
    assert "orphan.py" not in result.entrypoints


# --------------------------------------------------------------------------
# performance
# --------------------------------------------------------------------------


def test_build_graph_handles_2000_files_quickly(tmp_path: Path):
    root = tmp_path / "big"
    paths: list[str] = []
    for i in range(2000):
        # A three-deep package tree, chained, so most files are reachable.
        rel = f"pkg{i % 50}/sub{i % 20}/mod{i}.py"
        write(root, rel, f"import pkg{(i + 1) % 50}.sub{(i + 1) % 20}.mod{(i + 1) % 2000}\n")
        paths.append(rel)
    read = reader(root)

    start = time.perf_counter()
    graph = build_graph(paths, read)
    elapsed = time.perf_counter() - start

    assert set(graph) == set(paths)
    # Generous on purpose: this is a smoke test for "not accidentally O(n^2) on
    # the filesystem", not a benchmark. A tight bound fails on a loaded CI box.
    assert elapsed < 10.0, f"build_graph took {elapsed:.2f}s for 2000 files"


def test_hop_distance_on_a_large_graph_is_quick(tmp_path: Path):
    graph = {f"mod{i}.py": {f"mod{(i + 1) % 1000}.py"} for i in range(1000)}
    start = time.perf_counter()
    distances = hop_distance(graph, ["mod0.py"])
    elapsed = time.perf_counter() - start
    assert len(distances) == 1000
    assert elapsed < 2.0


def test_repeated_parse_is_memoised(py_repo: Path, monkeypatch: pytest.MonkeyPatch):
    """The per-path cache means one parse per file per build."""
    calls: list[str] = []

    def read(path: str) -> str | None:
        calls.append(path)
        return reader(py_repo)(path)

    paths = ["main.py", "pkg/a.py", "pkg/c.py"]
    build_graph(paths, read)
    assert len(calls) == len(set(calls))


def test_read_text_adapter_agrees_with_direct_read(py_repo: Path):
    from ctxpack.walk import discover

    discovery = discover(py_repo)
    texts = {c.path: read_text(c) for c in discovery.files}
    paths = sorted(texts)
    via_walk = build_graph(paths, lambda p: texts[p])

    def raw(path: str) -> str | None:
        try:
            return (py_repo / path).read_text(encoding="utf-8")
        except OSError:
            return None

    assert via_walk == build_graph(paths, raw)


# ---------------------------------------------------------------------------
# fan-out caps (regressions found by review)
# ---------------------------------------------------------------------------


def _reader(files: dict[str, str]):
    """A `read` callable over a fixed mapping, safe to capture in a loop."""
    table = dict(files)
    return lambda path: table.get(path)


def test_one_specifier_cannot_resolve_to_the_whole_repository():
    """Fan-out must be capped on the suffix branch, not just the stem fallback.

    Regression with two consequences at once. Uncapped, 400 services each
    importing ``pkg.mod`` produced 320,000 edges -- quadratic in file count --
    and every ``svc*/app.py`` claimed to import every other service's module,
    so ``--reach-weight`` promoted hundreds of unrelated files on the strength
    of a confidently wrong "reachable" signal.
    """
    from ctxpack.deps import MAX_EDGES_PER_SPEC

    for services in (20, 60):
        files: dict[str, str] = {}
        for i in range(services):
            files[f"svc{i}/pkg/__init__.py"] = ""
            files[f"svc{i}/pkg/mod.py"] = "X = 1\n"
            files[f"svc{i}/app.py"] = "from pkg.mod import X\n"
        graph = build_graph(list(files), _reader(files))
        edges = sum(len(v) for v in graph.values())
        assert edges <= services * MAX_EDGES_PER_SPEC, (
            f"{services} services produced {edges} edges"
        )
        assert len(graph["svc0/app.py"]) <= MAX_EDGES_PER_SPEC


def test_edge_cap_is_deterministic():
    files: dict[str, str] = {}
    for i in range(30):
        files[f"svc{i}/pkg/mod.py"] = "X = 1\n"
        files[f"svc{i}/app.py"] = "from pkg.mod import X\n"
    first = build_graph(list(files), lambda p: files.get(p))
    second = build_graph(list(reversed(list(files))), lambda p: files.get(p))
    assert first == second


def test_relative_resolution_is_unaffected_by_the_cap():
    """A relative import that matches exactly one file must still match it."""
    files = {"src/a/x.ts": "", "src/b/y.ts": "import {x} from '../a/x.js';\n"}
    graph = build_graph(list(files), lambda p: files.get(p))
    assert graph["src/b/y.ts"] == {"src/a/x.ts"}


def test_jsx_resolves_to_tsx_under_nodenext():
    """NodeNext maps .jsx to .tsx, the same way it maps .js to .ts."""
    got = resolve_js("./Button.jsx", "src/ui/entry.ts")
    assert "src/ui/Button.tsx" in got
    assert "src/ui/Button.jsx" in got


def test_mts_and_cts_are_recognised_as_javascript():
    from ctxpack.deps import JS_EXTS, _language

    for ext in (".mts", ".cts"):
        assert ext in JS_EXTS
        assert _language(f"src/mod{ext}") is not None
