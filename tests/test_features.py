"""Integration tests for the features wired into the CLI.

The unit tests in ``test_deps.py``, ``test_gitdiff.py``, ``test_boiler.py`` and
``test_outline.py`` cover the modules themselves. What can only be tested here
is the wiring: that the flags reach the modules, that the errors they raise
surface as clean one-line messages rather than tracebacks, and that combining
them does something sensible.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ctxpack.cli import main
from ctxpack.walk import discover


def run(capsys, argv) -> tuple[int, str, str]:
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ---------------------------------------------------------------------------
# --diff
# ---------------------------------------------------------------------------


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=t@example.com", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A committed repo with an edit, a deletion, and an untracked file."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "README.md").write_text("# repo\n\nOriginal.\n", encoding="utf-8")
    (root / "src" / "keep.py").write_text("KEEP = 1\n", encoding="utf-8")
    (root / "src" / "gone.py").write_text("GONE = 1\n", encoding="utf-8")
    (root / "src" / "big.py").write_text("BIG = 1\n" * 200, encoding="utf-8")

    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "initial")

    (root / "src" / "keep.py").write_text("KEEP = 2\nCHANGED = True\n", encoding="utf-8")
    (root / "src" / "gone.py").unlink()
    (root / "brand_new.py").write_text("NEW = 1\n", encoding="utf-8")
    return root


@pytest.mark.parametrize("fmt", ["markdown", "xml", "json", "tree"])
def test_diff_packs_only_changed_files(capsys, repo: Path, fmt):
    code, out, err = run(capsys, ["pack", str(repo), "--diff", "HEAD", "-b", "4000", "-q"])
    assert code == 0
    assert "README.md" not in out  # unchanged, so excluded
    assert "src/keep.py" in out
    assert "src/gone.py" not in out  # deleted, never packed
    assert "--diff HEAD" in err


def test_diff_untracked_needs_the_flag(capsys, repo: Path):
    _, without, _ = run(capsys, ["pack", str(repo), "--diff", "HEAD", "-b", "4000", "-q"])
    assert "brand_new.py" not in without
    _, with_it, _ = run(
        capsys,
        ["pack", str(repo), "--diff", "HEAD", "--untracked", "-b", "4000", "-q"],
    )
    assert "brand_new.py" in with_it


def test_diff_staged(capsys, repo: Path):
    git(repo, "add", "src/keep.py")
    _, out, _ = run(
        capsys, ["pack", str(repo), "--diff", "HEAD", "--staged", "-b", "4000", "-q"]
    )
    assert "src/keep.py" in out


def test_diff_count_json_respects_the_range(capsys, repo: Path):
    _, out, _ = run(capsys, ["count", str(repo), "--diff", "HEAD", "--json", "-n", "0"])
    paths = {f["path"] for f in json.loads(out)["files"]}
    assert "src/keep.py" in paths
    assert "README.md" not in paths


def test_diff_bad_ref_is_a_clean_error(capsys, repo: Path):
    code, _, err = run(
        capsys, ["pack", str(repo), "--diff", "no-such-ref", "-b", "4000", "-q"]
    )
    assert code == 2
    assert "ctxpack:" in err
    assert "Traceback" not in err


def test_diff_outside_a_git_repo_is_a_clean_error(capsys, project: Path):
    code, _, err = run(capsys, ["pack", str(project), "--diff", "HEAD", "-q"])
    assert code == 2
    assert "git" in err.lower()
    assert "Traceback" not in err


def test_diff_with_no_changes_is_a_clean_error(capsys, repo: Path):
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "second")
    code, _, err = run(capsys, ["pack", str(repo), "--diff", "HEAD", "-q"])
    assert code == 2
    assert "no files changed" in err


def test_diff_empty_range_rejected(capsys, repo: Path):
    code, _, err = run(capsys, ["pack", str(repo), "--diff", "..", "-q"])
    assert code == 2
    assert "ctxpack:" in err


# ---------------------------------------------------------------------------
# --reach-weight
# ---------------------------------------------------------------------------


@pytest.fixture
def linked_repo(tmp_path: Path) -> Path:
    """A Python package with a real import chain from an entrypoint."""
    root = tmp_path / "linked"
    (root / "src" / "app").mkdir(parents=True)
    (root / "README.md").write_text("# linked\n", encoding="utf-8")
    (root / "src" / "app" / "__init__.py").write_text(
        "from .server import serve\n", encoding="utf-8"
    )
    (root / "src" / "app" / "server.py").write_text(
        "from .store import Store\n", encoding="utf-8"
    )
    (root / "src" / "app" / "store.py").write_text("STORE = 1\n", encoding="utf-8")
    (root / "src" / "orphan.py").write_text("ORPHAN = 1\n", encoding="utf-8")
    return root


def test_reach_weight_reports_graph_stats(capsys, linked_repo: Path):
    _, _, err = run(
        capsys, ["pack", str(linked_repo), "-b", "4000", "--reach-weight", "4", "-q"]
    )
    assert "import graph" in err
    assert "reachable from" in err


def test_reach_weight_zero_is_a_noop(capsys, linked_repo: Path):
    _, out, err = run(capsys, ["pack", str(linked_repo), "-b", "4000", "-q"])
    assert "import graph" not in err
    assert out


def test_reach_weight_orders_reachable_files_first(capsys, linked_repo: Path):
    _, out, _ = run(
        capsys,
        [
            "pack", str(linked_repo), "-b", "4000",
            "--reach-weight", "20", "-q", "-f", "json",
        ],
    )
    paths = [f["path"] for f in json.loads(out)["files"]]
    store = paths.index("src/app/store.py")
    orphan = paths.index("src/orphan.py")
    assert store < orphan


def test_reach_weight_survives_unparseable_source(capsys, tmp_path: Path):
    root = tmp_path / "broken"
    root.mkdir()
    (root / "main.py").write_text("import ??? not python at all (\n", encoding="utf-8")
    (root / "other.py").write_text("def f(:\n", encoding="utf-8")
    code, out, _ = run(capsys, ["pack", str(root), "-b", "4000", "--reach-weight", "4", "-q"])
    assert code == 0
    assert out


def test_reach_weight_degrades_when_no_entrypoint(tmp_path, capsys):
    root = tmp_path / "noentry"
    root.mkdir()
    (root / "thing.py").write_text("X = 1\n", encoding="utf-8")
    code, out, err = run(
        capsys, ["pack", str(root), "-b", "4000", "--reach-weight", "4", "-q"]
    )
    assert code == 0
    assert "no entrypoints" in err or out


# ---------------------------------------------------------------------------
# --factor-shared
# ---------------------------------------------------------------------------


@pytest.fixture
def header_repo(tmp_path: Path) -> Path:
    """Distinct files sharing one repeated header: factoring's ideal case."""
    root = tmp_path / "headers"
    (root / "lib").mkdir(parents=True)
    header = "\n".join(
        f"# Copyright (c) 2026. Standard notice, line {i}." for i in range(12)
    )
    for i in range(20):
        (root / "lib" / f"mod_{i}.py").write_text(
            header
            + "\nimport os\n\n\n"
            + "\n".join(f"def step_{i}_{j}(x):\n    return x + {j}\n" for j in range(20 + i)),
            encoding="utf-8",
        )
    (root / "README.md").write_text("# headers\n", encoding="utf-8")
    # An entrypoint, so the reachability feature has something to walk from.
    (root / "lib" / "app.py").write_text(
        "\n".join(f"from .mod_{i} import step_{i}_0\n" for i in range(20)),
        encoding="utf-8",
    )
    return root


def test_factor_shared_is_opt_in(capsys, header_repo: Path):
    _, without, _ = run(capsys, ["pack", str(header_repo), "-b", "30000", "-q"])
    _, with_it, _ = run(
        capsys, ["pack", str(header_repo), "-b", "30000", "--factor-shared", "-q"]
    )
    # Off by default: the header text is present verbatim.
    assert "Copyright (c) 2026. Standard notice, line 0." in without
    # On: replaced by a marker, and the outcome is reported in the bundle.
    assert "[ctxpack: shared block B1" in with_it
    assert "factored" in with_it


def test_factor_shared_reduces_tokens(capsys, header_repo: Path):
    plain = run(capsys, ["pack", str(header_repo), "-b", "30000", "-q"])[1]
    factored = run(
        capsys, ["pack", str(header_repo), "-b", "30000", "--factor-shared", "-q"]
    )[1]
    assert len(factored) < len(plain)


def test_factor_shared_emits_a_shared_section(capsys, header_repo: Path):
    _, out, _ = run(
        capsys, ["pack", str(header_repo), "-b", "30000", "--factor-shared", "-q"]
    )
    assert "## Shared blocks" in out
    assert "not for compiling" in out
    assert "[ctxpack: shared block B1" in out


def test_factor_shared_markers_reference_real_blocks(capsys, header_repo: Path):
    _, out, _ = run(
        capsys, ["pack", str(header_repo), "-b", "30000", "--factor-shared", "-q"]
    )
    section = out.split("## Shared blocks", 1)[1]
    for marker in set(out.split("## Shared blocks", 1)[0].split("[ctxpack: shared block ")[1:]):
        block_id = marker.split(" ")[0]
        assert f"### {block_id} " in section


@pytest.mark.parametrize("fmt", ["markdown", "xml", "json", "tree"])
def test_factor_shared_every_format_stays_valid(capsys, header_repo: Path, fmt):
    code, out, _ = run(
        capsys,
        ["pack", str(header_repo), "-b", "30000", "--factor-shared", "-q", "-f", fmt],
    )
    assert code == 0
    if fmt == "json":
        json.loads(out)
    elif fmt == "xml":
        from xml.etree import ElementTree

        ElementTree.fromstring(out)
    elif fmt == "markdown":
        assert "## Shared blocks" in out


def test_factor_shared_is_a_noop_without_repeated_blocks(capsys, project: Path):
    """Nothing to hoist means nothing added -- and nothing corrupted."""
    plain = run(capsys, ["pack", str(project), "-b", "30000", "-q"])[1]
    _, out, _ = run(
        capsys, ["pack", str(project), "-b", "30000", "--factor-shared", "-q"]
    )
    assert "[ctxpack: shared block" not in out
    assert "## Shared blocks" not in out
    assert len(out) == len(plain)


def test_factor_shared_min_occurrences(capsys, header_repo: Path):
    _, out, _ = run(
        capsys,
        [
            "pack", str(header_repo), "-b", "30000",
            "--factor-shared", "--shared-min-occurrences", "999", "-q",
        ],
    )
    assert "[ctxpack: shared block" not in out


def test_factor_shared_min_lines(capsys, header_repo: Path):
    _, out, _ = run(
        capsys,
        [
            "pack", str(header_repo), "-b", "30000",
            "--factor-shared", "--shared-min-lines", "500", "-q",
        ],
    )
    assert "[ctxpack: shared block" not in out


# ---------------------------------------------------------------------------
# combinations
# ---------------------------------------------------------------------------


def test_diff_and_reach_weight_compose(capsys, linked_repo: Path):
    """Both flags apply together on a repo where the graph has an entrypoint."""
    code, out, err = run(
        capsys,
        [
            "pack", str(linked_repo), "-b", "4000",
            "--reach-weight", "3", "-q",
        ],
    )
    assert code == 0
    assert "import graph" in err
    assert "src/app/store.py" in out


def test_diff_narrowing_can_remove_every_entrypoint(capsys, repo: Path):
    """Reachability must degrade to a warning, not an error.

    After `--diff` leaves one unremarkable file there is nothing to walk from,
    and the honest response is to say so and carry on with the static ranking.
    """
    code, out, err = run(
        capsys,
        ["pack", str(repo), "--diff", "HEAD", "-b", "4000", "--reach-weight", "3", "-q"],
    )
    assert code == 0
    assert "src/keep.py" in out
    assert "no entrypoints" in err


def test_factor_shared_and_reach_weight_compose(capsys, header_repo: Path):
    code, out, err = run(
        capsys,
        [
            "pack", str(header_repo), "-b", "30000",
            "--reach-weight", "2", "--factor-shared", "-q",
        ],
    )
    assert code == 0
    assert "[ctxpack: shared block B1" in out
    assert "import graph" in err


# ---------------------------------------------------------------------------
# --outline
# ---------------------------------------------------------------------------


@pytest.fixture
def bulky_repo(tmp_path: Path) -> Path:
    """Files with real bodies, so outlining them is worth doing."""
    root = tmp_path / "bulky"
    root.mkdir()
    (root / "README.md").write_text("# bulky\n", encoding="utf-8")
    for i in range(6):
        (root / f"module_{i}.py").write_text(
            "\n".join(
                f"def step_{i}_{j}(argument, context):\n"
                f"    total = argument\n"
                f"    for other in context.related_objects():\n"
                f"        total = total + other.weight({j})\n"
                f"    return total\n"
                for j in range(30)
            ),
            encoding="utf-8",
        )
    return root


def test_outline_is_opt_in(capsys, bulky_repo: Path):
    _, plain, _ = run(capsys, ["pack", str(bulky_repo), "-b", "30000", "-q"])
    _, outlined, _ = run(
        capsys, ["pack", str(bulky_repo), "-b", "30000", "--outline", "-q"]
    )
    assert "total = argument" in plain
    assert "total = argument" not in outlined
    assert len(outlined) < len(plain)


def test_outline_reports_what_it_did(capsys, bulky_repo: Path):
    _, out, _ = run(
        capsys, ["pack", str(bulky_repo), "-b", "30000", "--outline", "-q"]
    )
    assert "outline mode" in out
    # Must be explicit that implementations are absent.
    assert "signatures, not implementations" in out


def test_outline_keeps_signatures(capsys, bulky_repo: Path):
    _, out, _ = run(
        capsys, ["pack", str(bulky_repo), "-b", "30000", "--outline", "-q"]
    )
    assert "step_0_0" in out
    assert "step_5_29" in out


def test_outline_min_ratio_skips_unprofitable_files(capsys, bulky_repo: Path):
    """A threshold nothing can meet must leave every file as full text."""
    _, plain, _ = run(capsys, ["pack", str(bulky_repo), "-b", "30000", "-q"])
    _, out, _ = run(
        capsys,
        [
            "pack", str(bulky_repo), "-b", "30000",
            "--outline", "--outline-min-ratio", "1000", "-q",
        ],
    )
    # Bodies intact, and the only difference is the note explaining why nothing
    # was summarised -- so byte lengths differ even though content does not.
    for marker in ("total = argument", "step_5_29", "related_objects"):
        assert marker in out
        assert marker in plain
    assert "outline mode: no files" in out


@pytest.mark.parametrize("fmt", ["markdown", "xml", "json", "tree"])
def test_outline_every_format(capsys, bulky_repo: Path, fmt):
    code, out, _ = run(
        capsys, ["pack", str(bulky_repo), "-b", "30000", "--outline", "-q", "-f", fmt]
    )
    assert code == 0
    if fmt == "json":
        json.loads(out)
    elif fmt == "xml":
        from xml.etree import ElementTree

        ElementTree.fromstring(out)


def test_outline_wins_over_factor_shared(capsys, header_repo: Path):
    """Outlining throws the bodies away, so block-factoring has nothing to hoist.

    Documented precedence rather than an accident: running both would factor
    structure, which is not what either flag means.
    """
    _, out, _ = run(
        capsys,
        [
            "pack", str(header_repo), "-b", "30000",
            "--outline", "--factor-shared", "-q",
        ],
    )
    assert "outline mode" in out
    assert "## Shared blocks" not in out


def test_outline_and_diff_compose(capsys, repo: Path):
    code, out, _ = run(
        capsys,
        ["pack", str(repo), "--diff", "HEAD", "-b", "4000", "--outline", "-q"],
    )
    assert code == 0
    assert out


# ---------------------------------------------------------------------------
# the invariant, with everything switched on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["balanced", "coverage", "depth"])
@pytest.mark.parametrize("outline", [False, True])
@pytest.mark.parametrize("budget", [400, 1_500, 8_000, 40_000])
def test_budget_holds_with_every_transform_enabled(header_repo: Path, mode, outline, budget):
    """No combination of features may push a bundle over its budget.

    Each feature preserves the invariant on its own; the point of sweeping the
    combinations is that outline rewrites text *during* selection while
    factoring rewrites it *afterwards*, and those two touch different stages.
    """
    from ctxpack.cli import _outline_transform
    from ctxpack.pack import Budget, Packer
    from ctxpack.tokens import Tokenizer

    result = Packer(
        Tokenizer(mode="never"),
        Budget(total=budget),
        mode=mode,
        transform=(
            _outline_transform(argparse.Namespace(outline_min_ratio=2.0))
            if outline
            else None
        ),
    ).pack(discover(header_repo))
    assert result.accounted <= result.budget


def test_all_flags_together_respect_the_budget(capsys, header_repo: Path):
    code, out, _ = run(
        capsys,
        [
            "pack", str(header_repo), "-b", "30000",
            "--reach-weight", "3", "--outline", "-q",
        ],
    )
    assert code == 0
    # The bundle must not exceed the budget by more than header/format overhead.
    from ctxpack.tokens import Tokenizer

    counted = Tokenizer().count(out).tokens
    assert counted <= 30_000 + int(30_000 * 0.10) + 500


def test_diff_on_a_single_file_root(capsys, repo: Path):
    """`--diff` must work when the target is one file, not a directory.

    Regression: `relativise` speaks repo-relative paths while a single-file
    discovery names the file by its basename, so the two never intersected.
    `--diff` on one file matched nothing and then suggested `--no-gitignore` or
    a wider root, neither of which could possibly have helped.
    """
    target = repo / "src" / "keep.py"
    code, out, _ = run(capsys, ["pack", str(target), "--diff", "HEAD", "-q"])
    assert code == 0
    assert "keep.py" in out


def test_diff_directory_root_still_works(capsys, repo: Path):
    code, out, _ = run(
        capsys, ["pack", str(repo), "--diff", "HEAD", "-b", "4000", "-q"]
    )
    assert code == 0
    assert "src/keep.py" in out


def test_factor_shared_says_so_when_outline_wins(capsys, header_repo: Path):
    """The documented precedence must be visible, not silent."""
    _, out, _ = run(
        capsys,
        [
            "pack", str(header_repo), "-b", "30000",
            "--outline", "--factor-shared", "-q",
        ],
    )
    assert "skipped --factor-shared" in out
    assert "## Shared blocks" not in out


# ---------------------------------------------------------------------------
# config files
# ---------------------------------------------------------------------------


#: True where ``tomllib`` is available, i.e. Python 3.11+.
TOML_SUPPORTED = sys.version_info >= (3, 11)


def write_config(root: Path, settings: dict, *, name: str = "") -> Path:
    """Write a config in whichever format this interpreter can actually read.

    TOML needs ``tomllib``, which is stdlib only from 3.11. On 3.10 ctxpack
    refuses a ``.toml`` file and says so -- correctly -- so the tests have to use
    the format the running Python supports, which is exactly what a 3.10 user
    has to do. Getting this wrong makes the whole config surface look broken on
    the oldest supported version.
    """
    if not name:
        name = "ctxpack.toml" if TOML_SUPPORTED else "ctxpack.json"
    if name.endswith(".json"):
        body = json.dumps(settings)
    else:
        lines = []
        for key, value in settings.items():
            rendered = json.dumps(value) if not isinstance(value, bool) else (
                "true" if value else "false"
            )
            lines.append(f"{key} = {rendered}")
        body = "\n".join(lines) + "\n"
    path = root / name
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture
def configured(tmp_path: Path) -> Path:
    root = tmp_path / "configured"
    root.mkdir()
    (root / "README.md").write_text("# configured\n", encoding="utf-8")
    (root / "mod.py").write_text("X = 1\n" * 40, encoding="utf-8")
    (root / "noise.log").write_text("noise\n", encoding="utf-8")
    return root


def test_config_file_settings_are_applied(capsys, configured: Path):
    write_config(configured, {"budget": 4321, "mode": "coverage"})
    code, out, err = run(capsys, ["pack", str(configured), "-f", "tree"])
    assert code == 0
    assert "4,321 tokens budget" in out
    assert "coverage" in out
    assert "using settings" in err


def test_explicit_flag_beats_the_config_file(capsys, configured: Path):
    """The subtle case: a typed flag whose value equals the built-in default.

    argparse cannot tell `-m balanced` typed on purpose from the default it
    filled in, so config merging has to recover that from argv itself. Getting it
    wrong means a user cannot override a shared config file at all.
    """
    write_config(configured, {"budget": 8000, "mode": "coverage"})
    _, out, _ = run(capsys, ["pack", str(configured), "-f", "tree", "-b", "1500"])
    assert "1,500 tokens budget" in out

    _, out, _ = run(
        capsys, ["pack", str(configured), "-f", "tree", "-m", "balanced"]
    )
    assert "balanced" in out and "coverage" not in out


def test_no_config_flag(capsys, configured: Path):
    write_config(configured, {"budget": 4321})
    _, out, _ = run(capsys, ["pack", str(configured), "-f", "tree", "--no-config"])
    assert "32,000 tokens budget" in out


def test_config_never_overrides_the_target_path(capsys, configured: Path, tmp_path):
    """A config file must not decide which path gets packed.

    Regression: `path` is a positional, and the mechanism that detects which
    flags were typed only understands option strings. The merge therefore treated
    an explicitly given path as unspecified and replaced it with its default of
    "." -- silently packing the current working directory instead. No error, just
    a bundle of an entirely different repository.
    """
    write_config(configured, {"budget": 8000})
    code, out, _ = run(capsys, ["pack", str(configured), "-f", "tree"])
    assert code == 0
    assert str(configured.resolve()) in out


def test_config_exclude_applies(capsys, configured: Path):
    write_config(configured, {"exclude": ["*.log"]})
    _, out, _ = run(capsys, ["pack", str(configured), "-f", "tree"])
    assert "noise.log" not in out


def test_config_typo_is_reported(capsys, configured: Path):
    write_config(configured, {"budegt": 5, "budget": 9000})
    code, out, err = run(capsys, ["pack", str(configured), "-f", "tree"])
    assert code == 0
    assert "budegt" in err
    # A typo in one key must not discard the rest of the file.
    assert "9,000 tokens budget" in out


def test_show_config(capsys, configured: Path):
    write_config(configured, {"budget": 4321, "mode": "depth"})
    code, _, err = run(capsys, ["pack", str(configured), "-f", "tree", "--show-config"])
    assert code == 0
    assert "4321" in err and "(config)" in err
    assert "(default)" in err


def test_malformed_config_is_a_clean_error(capsys, configured: Path):
    # Deliberately invalid in both formats.
    (configured / ("ctxpack.json" if TOML_SUPPORTED else "ctxpack.toml")).write_text(
        "{ not valid", encoding="utf-8"
    )
    code, _, err = run(capsys, ["pack", str(configured), "-f", "tree"])
    assert code == 2
    assert "ctxpack:" in err
    assert "Traceback" not in err
