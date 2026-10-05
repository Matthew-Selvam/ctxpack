"""Git-aware file selection.

Every test here drives a real ``git`` in a temporary repository.  Mocking the
subprocess layer would test the mock: rename detection, copy detection and
three-dot merge bases all have surprising behaviour, and those behaviours are
precisely what this module has to survive.  The only concession to portability
is skipping the module when there is no git to talk to.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from ctxpack.errors import CtxpackError
from ctxpack.gitdiff import (
    USAGE_EXIT,
    MergeBase,
    _parse_name_status,
    _parse_numstat,
    _run_merge_base_fallback,
    changed_count,
    changed_files,
    diffstat,
    is_git_repo,
    parse_refs,
    relativise,
    repo_root,
)
from ctxpack.gitdiff import _Plan as _Plan
from ctxpack.walk import discover

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git is not on PATH"
)

# Passed on every invocation so the tests never depend on the developer's
# ~/.gitconfig: no signing, no GPG prompt, and an identity that always exists.
IDENTITY = (
    "-c",
    "user.name=ctxpack tests",
    "-c",
    "user.email=tests@example.invalid",
    "-c",
    "commit.gpgsign=false",
)


def git(root: Path, *args: str) -> str:
    """Run git in ``root``, failing loudly -- a broken fixture is not a failure
    of the thing under test, and pytest's traceback says which command died."""
    proc = subprocess.run(
        ["git", *IDENTITY, *args],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"git {' '.join(args)} failed: {proc.stderr}"
    return proc.stdout


def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def commit(root: Path, message: str) -> None:
    git(root, "add", "-A")
    git(root, "commit", "-m", message)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository with one commit and a clean work tree."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "docs").mkdir(parents=True)

    write(root, "README.md", "# demo\n")
    write(root, "src/main.py", "def main():\n    return 1\n")
    write(root, "src/keep.py", "KEEP = 1\n")
    write(root, "docs/old.txt", "old\n")
    write(root, "docs/gone.txt", "gone\n")

    git(root, "init", "-q", ".")
    commit(root, "initial")
    return root


@pytest.fixture
def branched(tmp_path: Path) -> Path:
    """A repository where ``base`` and ``feature`` have diverged.

    ``docs/base_after_fork.txt`` is the file that separates two dots from
    three: it is edited on ``base`` after ``feature`` branched, so a two-dot
    diff sees a change and a merge-base diff does not.
    """
    root = tmp_path / "branched"
    (root / "src").mkdir(parents=True)

    write(root, "src/shared.py", "VALUE = 'fork point'\n")
    write(root, "src/only_base.py", "VALUE = 'fork point'\n")
    git(root, "init", "-q", ".")
    commit(root, "fork point")
    git(root, "branch", "base")

    git(root, "checkout", "-q", "-b", "feature")
    write(root, "src/shared.py", "VALUE = 'feature work'\n")
    write(root, "src/feature_work.txt", "added on feature\n")
    commit(root, "feature work")

    git(root, "checkout", "-q", "base")
    write(root, "src/only_base.py", "VALUE = 'base work'\n")
    commit(root, "base work after the fork")
    return root


@pytest.fixture
def empty_repo(tmp_path: Path) -> Path:
    """A repository with an unborn HEAD."""
    root = tmp_path / "empty"
    root.mkdir()
    git(root, "init", "-q", ".")
    return root


# -- parse_refs --------------------------------------------------------------


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("", (None, None)),
        ("   ", (None, None)),
        ("main", ("main", None)),
        ("HEAD", ("HEAD", None)),
        ("HEAD~3", ("HEAD~3", None)),
        ("main..", ("main", None)),
        ("..HEAD", (None, "HEAD")),
        ("main..HEAD", ("main", "HEAD")),
        ("v1.0.0..v2.0.0", ("v1.0.0", "v2.0.0")),
        ("origin/main...HEAD", ("origin/main", "HEAD")),
    ],
)
def test_parse_refs(spec, expected):
    assert parse_refs(spec) == expected


def test_three_dot_yields_a_merge_base_tagged_base():
    base, head = parse_refs("main...HEAD")
    assert base == "main"
    assert head == "HEAD"
    # Still a plain string for every purpose that takes one.
    assert isinstance(base, str)
    assert isinstance(base, MergeBase)


@pytest.mark.parametrize(
    "spec",
    ["..", "...", "....", "a..b..c", "main..HEAD~1..x", "a...b...c"],
)
def test_parse_refs_rejects_garbage(spec):
    with pytest.raises(CtxpackError):
        parse_refs(spec)


@pytest.mark.parametrize("spec", ["-x", "--cached", "a b", "a\tb", "mai\nn", "a  b"])
def test_parse_refs_rejects_specs_that_are_not_revisions(spec):
    with pytest.raises(CtxpackError):
        parse_refs(spec)


def test_parse_refs_tolerates_surrounding_whitespace():
    # Shell quoting makes " main...HEAD " easy to type by accident.
    assert parse_refs("  main...HEAD  ") == ("main", "HEAD")
    assert parse_refs(" HEAD~3 ") == ("HEAD~3", None)


def test_three_dot_needs_both_sides():
    with pytest.raises(CtxpackError, match="both sides"):
        parse_refs("main...")


# -- locating the repository -------------------------------------------------


def test_is_git_repo_true(repo: Path):
    assert is_git_repo(repo) is True


def test_is_git_repo_is_true_from_a_subdirectory(repo: Path):
    assert is_git_repo(repo / "src") is True


def test_is_git_repo_false_outside_a_repo(tmp_path: Path):
    plain = tmp_path / "plain"
    plain.mkdir()
    assert is_git_repo(plain) is False


def test_is_git_repo_false_for_a_bare_repository(tmp_path: Path):
    bare = tmp_path / "bare.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", str(bare)],
        capture_output=True,
        text=True,
        check=True,
    )
    # History yes, files no: there is nothing to pack, so this is False.
    assert is_git_repo(bare) is False


def test_is_git_repo_false_for_a_missing_path(tmp_path: Path):
    assert is_git_repo(tmp_path / "nope") is False


def test_a_single_file_is_inside_the_repository(repo: Path):
    """discover() takes a file as its root, so gitdiff has to as well."""
    assert is_git_repo(repo / "README.md") is True
    assert repo_root(repo / "README.md") == repo.resolve()
    write(repo, "README.md", "# changed\n")
    assert changed_files(repo / "README.md") == ["README.md"]


def test_repo_root_is_the_work_tree_top(repo: Path):
    assert repo_root(repo) == repo.resolve()
    # Asking from a subdirectory must still find the top, not the subdirectory.
    assert repo_root(repo / "src") == repo.resolve()


def test_repo_root_rejects_a_non_repository(tmp_path: Path):
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(CtxpackError, match="work tree"):
        repo_root(plain)


def test_changed_files_rejects_a_non_repository(tmp_path: Path):
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(CtxpackError, match="work tree"):
        changed_files(plain)


def test_changed_files_rejects_a_missing_path(tmp_path: Path):
    with pytest.raises(CtxpackError, match="no such path"):
        changed_files(tmp_path / "nowhere")


def test_error_is_one_line(repo: Path):
    with pytest.raises(CtxpackError) as excinfo:
        changed_files(repo, "nope")
    assert "\n" not in str(excinfo.value)


# -- uncommitted changes -----------------------------------------------------


def test_modified_file_is_listed(repo: Path):
    write(repo, "src/main.py", "def main():\n    return 2\n")
    assert changed_files(repo) == ["src/main.py"]


def test_untracked_file_is_invisible_by_default(repo: Path):
    write(repo, "src/brand_new.py", "NEW = 1\n")
    assert changed_files(repo) == []
    assert changed_count(repo) == 0


def test_untracked_file_is_included_on_request(repo: Path):
    write(repo, "src/brand_new.py", "NEW = 1\n")
    assert changed_files(repo, include_untracked=True) == ["src/brand_new.py"]


def test_untracked_respects_gitignore(repo: Path):
    write(repo, ".gitignore", "build/\n")
    write(repo, "build/out.js", "var x = 1;\n")
    assert changed_files(repo, include_untracked=True) == [".gitignore"]


def test_deleted_file_is_excluded(repo: Path):
    (repo / "docs" / "gone.txt").unlink()
    assert changed_files(repo) == []
    assert changed_count(repo) == 0


def test_clean_tree_reports_nothing(repo: Path):
    assert changed_files(repo) == []
    assert diffstat(repo) == {}


def test_results_are_sorted_and_stable(repo: Path):
    write(repo, "src/zeta.py", "Z = 1\n")
    write(repo, "src/alpha.py", "A = 1\n")
    git(repo, "add", "-A")
    first = changed_files(repo)
    assert first == sorted(first)
    assert first == changed_files(repo)


def test_multiple_changes(repo: Path):
    write(repo, "src/main.py", "def main():\n    return 2\n")
    (repo / "docs" / "gone.txt").unlink()
    write(repo, "src/new.py", "N = 1\n")
    git(repo, "add", "src/new.py")
    assert changed_files(repo) == ["src/main.py", "src/new.py"]


def test_paths_with_spaces_and_non_ascii(repo: Path):
    """-z is what makes this work; without it git quotes and escapes."""
    write(repo, "src/with space.py", "S = 1\n")
    write(repo, "src/ünïcode.py", "U = 1\n")
    git(repo, "add", "-A")
    # Compared against what the filesystem actually holds, because macOS stores
    # decomposed names and the two spellings are not equal.
    on_disk = {path.name for path in (repo / "src").iterdir()}
    expected = sorted(f"src/{name}" for name in on_disk - {"main.py", "keep.py"})
    assert changed_files(repo) == expected
    assert len(expected) == 2


# -- staged vs uncommitted ---------------------------------------------------


def test_staged_versus_uncommitted(repo: Path):
    write(repo, "src/main.py", "STAGED = 1\n")
    git(repo, "add", "src/main.py")
    write(repo, "src/keep.py", "UNSTAGED = 1\n")

    assert changed_files(repo, staged=True) == ["src/main.py"]
    # The work tree against HEAD: staged and unstaged together.
    assert changed_files(repo, uncommitted=True) == ["src/keep.py", "src/main.py"]


def test_staged_and_uncommitted_together_is_contradictory(repo: Path):
    with pytest.raises(CtxpackError, match="not both"):
        changed_files(repo, staged=True, uncommitted=True)


def test_uncommitted_and_a_head_ref_is_contradictory(repo: Path):
    with pytest.raises(CtxpackError, match="head revision"):
        changed_files(repo, None, "HEAD", uncommitted=True)


def test_a_clean_index_has_nothing_staged(repo: Path):
    assert changed_files(repo, staged=True) == []


# -- renames and copies ------------------------------------------------------


def test_rename_yields_only_the_new_path(repo: Path):
    git(repo, "mv", "docs/old.txt", "docs/new.txt")
    assert changed_files(repo) == ["docs/new.txt"]


def test_rename_across_a_commit_range(repo: Path):
    git(repo, "mv", "docs/old.txt", "docs/new.txt")
    commit(repo, "rename")
    assert changed_files(repo, "HEAD~1", "HEAD") == ["docs/new.txt"]


def test_rename_counts_against_the_new_path(repo: Path):
    git(repo, "mv", "docs/old.txt", "docs/new.txt")
    # A whole-file move is 0 added and 0 removed. The counts land on the
    # destination and the source appears nowhere -- not even as a zero.
    assert diffstat(repo) == {"docs/new.txt": {"added": 0, "removed": 0}}


def test_copy_yields_only_the_destination(repo: Path):
    # --find-copies-harder is what turns this A into a C; without it the source
    # is unmodified anyway so the file set is the same either way.
    write(repo, "src/copy.py", "KEEP = 1\n")
    git(repo, "add", "src/copy.py")
    assert changed_files(repo) == ["src/copy.py"]


def test_rename_and_copy_in_one_diff(repo: Path):
    """Two rename-like rows back to back, both needing the trailing-tab shape."""
    body = "\n".join(f"line {i}" for i in range(20)) + "\n"
    write(repo, "docs/old.txt", body)
    commit(repo, "a file worth renaming")

    git(repo, "mv", "docs/old.txt", "docs/renamed.txt")
    write(repo, "docs/renamed.txt", body + "one extra line\n")
    git(repo, "add", "-A")
    write(repo, "src/copy.py", "KEEP = 1\n")
    git(repo, "add", "src/copy.py")

    assert changed_files(repo) == ["docs/renamed.txt", "src/copy.py"]
    assert diffstat(repo) == {
        "docs/renamed.txt": {"added": 1, "removed": 0},
        "src/copy.py": {"added": 0, "removed": 0},
    }


# -- revision ranges ---------------------------------------------------------


def test_two_dot_compares_the_two_tips(branched: Path):
    paths = changed_files(branched, "base", "feature")
    assert "src/only_base.py" in paths
    assert "src/shared.py" in paths


def test_three_dot_compares_from_the_merge_base(branched: Path):
    base, head = parse_refs("base...feature")
    paths = changed_files(branched, base, head)
    assert "src/shared.py" in paths
    assert "src/feature_work.txt" in paths
    # Edited on base *after* feature forked, so only a two-dot diff sees it.
    assert "src/only_base.py" not in paths


def test_three_dot_from_parse_refs_equals_an_explicit_merge_base(branched: Path):
    assert changed_files(branched, *parse_refs("base...feature")) == changed_files(
        branched, "base", "feature", merge_base=True
    )


def test_two_dot_and_three_dot_disagree_where_they_should(branched: Path):
    two = changed_files(branched, "base", "feature")
    three = changed_files(branched, "base", "feature", merge_base=True)
    assert two != three
    assert set(two) - set(three) == {"src/only_base.py"}


def test_a_bare_ref_diffs_against_the_work_tree(repo: Path):
    write(repo, "src/main.py", "def main():\n    return 2\n")
    commit(repo, "second")
    write(repo, "src/main.py", "def main():\n    return 3\n")
    write(repo, "src/never_committed.py", "N = 1\n")
    git(repo, "add", "src/never_committed.py")
    # `git diff <rev>`: the work tree, index and all, against that revision.
    assert changed_files(repo, "HEAD") == ["src/main.py", "src/never_committed.py"]


def test_an_explicit_two_dot_spec(repo: Path):
    write(repo, "src/main.py", "def main():\n    return 2\n")
    commit(repo, "second")
    write(repo, "src/main.py", "def main():\n    return 3\n")
    commit(repo, "third")
    assert changed_files(repo, *parse_refs("HEAD~1..HEAD")) == ["src/main.py"]


def test_a_commit_range_ignores_the_work_tree(repo: Path):
    write(repo, "src/main.py", "def main():\n    return 2\n")
    git(repo, "add", "-A")
    # A two-sided diff compares two commits; the file on disk is irrelevant.
    assert changed_files(repo, "HEAD", "HEAD") == []


def test_merge_base_needs_two_revisions(repo: Path):
    with pytest.raises(CtxpackError, match="merge-base"):
        changed_files(repo, "HEAD", merge_base=True)


def test_an_unknown_option_is_what_usage_exit_detects(repo: Path):
    """The fallback trigger: what a git too old for --merge-base actually does."""
    proc = subprocess.run(
        ["git", "diff", "--merge-base", "--no-such-option"],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == USAGE_EXIT
    assert "unknown option" in proc.stderr or "invalid option" in proc.stderr


def test_the_old_git_fallback_matches_the_modern_answer(branched: Path):
    """Exercise the fallback directly; no git old enough to trigger it for real.

    It runs real git -- `git merge-base` then a two-sided diff -- which is
    exactly what ``git diff --merge-base A B`` does internally, so the answers
    have to be identical.
    """
    plan = _Plan(
        root=branched.resolve(), revs=("base", "feature"), merge_base=True
    )
    raw = _run_merge_base_fallback(plan, "--name-status")
    entries = _parse_name_status(raw)
    assert {entry.path for entry in entries if entry.status != "D"} == set(
        changed_files(branched, "base", "feature", merge_base=True)
    )
    assert "src/only_base.py" not in {entry.path for entry in entries}

    counts = {
        entry.path: (entry.added, entry.removed)
        for entry in _parse_numstat(_run_merge_base_fallback(plan, "--numstat"))
    }
    assert counts["src/feature_work.txt"] == (1, 0)
    assert "src/only_base.py" not in counts


# -- diffstat ----------------------------------------------------------------


def test_diffstat_counts_added_and_removed_lines(repo: Path):
    write(repo, "src/main.py", "def main():\n    return 2\n    # extra\n")
    assert diffstat(repo) == {"src/main.py": {"added": 2, "removed": 1}}


def test_diffstat_reports_deletions_even_though_changed_files_drops_them(
    repo: Path,
):
    (repo / "docs" / "gone.txt").unlink()
    assert changed_files(repo) == []
    assert diffstat(repo) == {"docs/gone.txt": {"added": 0, "removed": 1}}


def test_diffstat_keys_are_sorted(repo: Path):
    write(repo, "src/zeta.py", "Z = 1\n")
    write(repo, "src/alpha.py", "A = 1\n")
    git(repo, "add", "-A")
    assert list(diffstat(repo)) == ["src/alpha.py", "src/zeta.py"]


def test_diffstat_includes_untracked_at_zero(repo: Path):
    write(repo, "src/brand_new.py", "NEW = 1\n")
    assert diffstat(repo, include_untracked=True) == {
        "src/brand_new.py": {"added": 0, "removed": 0}
    }


def test_binary_files_report_zero(repo: Path):
    (repo / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    git(repo, "add", "logo.png")
    assert diffstat(repo) == {"logo.png": {"added": 0, "removed": 0}}


def test_diffstat_matches_a_three_dot_range(branched: Path):
    assert diffstat(branched, "base", "feature", merge_base=True) == {
        "src/feature_work.txt": {"added": 1, "removed": 0},
        "src/shared.py": {"added": 1, "removed": 1},
    }


# -- bad input ---------------------------------------------------------------


def test_bad_ref_reports_gits_own_message(repo: Path):
    with pytest.raises(CtxpackError) as excinfo:
        changed_files(repo, "nope")
    message = str(excinfo.value)
    assert "unknown revision" in message
    assert "nope" in message


def test_bad_ref_in_a_range_reports_gits_own_message(repo: Path):
    with pytest.raises(CtxpackError) as excinfo:
        changed_files(repo, "HEAD", "alsonope")
    assert "unknown revision" in str(excinfo.value)


def test_diffstat_reports_bad_refs_too(repo: Path):
    with pytest.raises(CtxpackError, match="unknown revision"):
        diffstat(repo, "nope")


def test_changed_count_reports_bad_refs_too(repo: Path):
    with pytest.raises(CtxpackError, match="unknown revision"):
        changed_count(repo, "nope")


def test_unborn_head_is_explained(empty_repo: Path):
    with pytest.raises(CtxpackError, match="no commits yet"):
        changed_files(empty_repo)


def test_unborn_head_still_allows_a_staged_diff(empty_repo: Path):
    write(empty_repo, "first.txt", "hello\n")
    git(empty_repo, "add", "first.txt")
    # --cached compares the index to an empty tree, so it works with no commits.
    assert changed_files(empty_repo, staged=True) == ["first.txt"]


# -- relativise --------------------------------------------------------------


def test_relativise_is_the_identity_at_the_repo_root(repo: Path):
    paths = ["src/main.py", "README.md"]
    assert relativise(paths, repo, repo) == paths


def test_relativise_drops_paths_outside_a_subdirectory(repo: Path):
    assert relativise(["src/main.py", "docs/gone.txt"], repo, repo / "src") == [
        "main.py"
    ]


def test_relativise_returns_nothing_when_everything_is_outside(repo: Path):
    assert relativise(["docs/gone.txt", "README.md"], repo, repo / "src") == []


def test_relativise_with_a_discovery_root_outside_the_repo(repo: Path, tmp_path: Path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    assert relativise(["src/main.py"], repo, outside) == []


def test_relativise_with_a_file_as_the_discovery_root(repo: Path):
    assert relativise(
        ["src/main.py", "src/keep.py", "README.md"], repo, repo / "src" / "main.py"
    ) == ["src/main.py"]


def test_relativise_preserves_order_and_deduplicates(repo: Path):
    assert relativise(
        ["src/main.py", "README.md", "src/main.py"], repo, repo
    ) == ["src/main.py", "README.md"]


def test_relativise_handles_a_symlinked_discovery_root(repo: Path, tmp_path: Path):
    link = tmp_path / "link-to-src"
    link.symlink_to(repo / "src", target_is_directory=True)
    # discover() resolves what it is handed, so relativise has to as well or
    # macOS's /tmp vs /private/tmp would silently drop everything.
    assert relativise(["src/main.py"], repo, link) == ["main.py"]


def test_relativised_paths_are_found_by_discovery(repo: Path):
    """The seam: what relativise returns must be selectable from a Discovery."""
    write(repo, "src/main.py", "def main():\n    return 2\n")
    write(repo, "docs/old.txt", "moved on\n")
    git(repo, "add", "-A")
    git(repo, "mv", "docs/old.txt", "docs/new.txt")

    paths = relativise(changed_files(repo), repo_root(repo), repo / "src")
    assert paths == ["main.py"]
    found = {candidate.path for candidate in discover(repo / "src").files}
    assert set(paths) <= found


# ---------------------------------------------------------------------------
# regressions found by review
# ---------------------------------------------------------------------------


def test_git_invocations_have_a_timeout(tmp_path, monkeypatch):
    """A hung `git` must not hang ctxpack.

    Regression: `subprocess.run` had no timeout, so a git that never returns --
    a blocked filesystem, a hook waiting on input, a pathological
    `--find-copies-harder` over a huge tree -- left the CLI waiting with no
    output and no way out.
    """
    import subprocess as sp

    from ctxpack import gitdiff

    seen: dict = {}

    def fake_run(argv, **kwargs):
        seen.update(kwargs)
        raise sp.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(gitdiff.subprocess, "run", fake_run)
    with pytest.raises(CtxpackError) as exc:
        gitdiff.is_git_repo(tmp_path)
    assert "did not finish" in str(exc.value)
    assert seen.get("timeout"), "no timeout was passed to subprocess.run"


def test_git_timeout_is_configurable(monkeypatch):
    import importlib

    from ctxpack import gitdiff

    monkeypatch.setenv("CTXPACK_GIT_TIMEOUT", "90")
    reloaded = importlib.reload(gitdiff)
    try:
        assert reloaded.GIT_TIMEOUT == 90.0
    finally:
        monkeypatch.delenv("CTXPACK_GIT_TIMEOUT")
        importlib.reload(gitdiff)
