"""File discovery and gitignore translation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ctxpack.errors import CtxpackError
from ctxpack.walk import (
    Candidate,
    IgnoreStack,
    compile_pattern,
    discover,
    parse_ignore_text,
    read_text,
)


def matches(pattern: str, path: str) -> bool:
    rule = compile_pattern(pattern)
    assert rule is not None
    return rule.matches(path)


# -- pattern translation -----------------------------------------------------


def test_comments_and_blanks_are_ignored():
    assert compile_pattern("") is None
    assert compile_pattern("   ") is None
    assert compile_pattern("# a comment") is None


@pytest.mark.parametrize(
    "path,expected",
    [
        ("app.log", True),
        ("deep/nested/app.log", True),
        ("app.logs", False),
    ],
)
def test_star_does_not_cross_slash(path, expected):
    assert matches("*.log", path) is expected


@pytest.mark.parametrize(
    "path,expected",
    [
        ("build", True),
        ("build/output.js", True),
        ("deep/build", True),
        ("deep/build/output.js", True),
        ("rebuild", False),
    ],
)
def test_trailing_slash_is_directory_only(path, expected):
    assert matches("build/", path) is expected


@pytest.mark.parametrize(
    "path,expected",
    [
        ("root-only.txt", True),
        ("deep/root-only.txt", False),
    ],
)
def test_leading_slash_anchors(path, expected):
    assert matches("/root-only.txt", path) is expected


def test_internal_slash_anchors():
    assert matches("src/main.py", "src/main.py")
    assert not matches("src/main.py", "lib/src/main.py")


@pytest.mark.parametrize(
    "path,expected",
    [
        ("a.txt", True),
        ("ab.txt", False),
        ("dir/a.txt", True),
    ],
)
def test_question_mark_is_single_character(path, expected):
    assert matches("?.txt", path) is expected


@pytest.mark.parametrize(
    "path,expected",
    [
        ("docs/a/b/x.md", True),
        ("docs/x.md", True),  # git: `/**/` matches zero directory levels
        ("docs/a/x.md", True),
        ("other/x.md", False),
    ],
)
def test_double_star_crosses_directories(path, expected):
    assert matches("docs/**/x.md", path) is expected


@pytest.mark.parametrize(
    "path,expected",
    [
        ("alpha.md", True),
        ("beta.md", True),
        ("gamma.md", False),
        ("dir/alpha.md", True),
    ],
)
def test_character_class(path, expected):
    assert matches("[ab]*.md", path) is expected


def test_negation_is_flagged():
    rule = compile_pattern("!keep.log")
    assert rule is not None
    assert rule.negated


def test_escaped_hash_is_literal():
    assert matches("\\#file", "#file")
    assert compile_pattern("\\#file") is not None


def test_unterminated_character_class_is_literal():
    assert compile_pattern("[abc") is not None


# -- the stack ---------------------------------------------------------------


def test_last_match_wins():
    stack = IgnoreStack(parse_ignore_text("*.log\n!keep.log\n"))
    assert stack.ignored("debug.log", False)
    assert not stack.ignored("keep.log", False)


def test_layers_are_consulted_outermost_first():
    stack = IgnoreStack(parse_ignore_text("*.log\n"))
    stack.push("sub", parse_ignore_text("!debug.log\n"))
    assert not stack.ignored("sub/debug.log", False)
    assert stack.ignored("other/debug.log", False)
    stack.pop()
    assert stack.ignored("sub/debug.log", False)


def test_ignored_directory_swallows_contents():
    stack = IgnoreStack(parse_ignore_text("node_modules\n"))
    assert stack.ignored("node_modules/pkg/index.js", False)


# -- discovery ---------------------------------------------------------------


def paths(found) -> set[str]:
    return {c.path for c in found.files}


def test_finds_source_files(project: Path):
    found = discover(project)
    found_paths = paths(found)
    assert "src/app/main.py" in found_paths
    assert "README.md" in found_paths


def test_default_ignores_drop_lockfiles(project: Path):
    assert "package-lock.json" not in paths(discover(project))


def test_default_ignores_drop_build_dirs(project: Path):
    assert not any(p.startswith("build/") for p in paths(discover(project)))


def test_gitignore_is_respected(project: Path):
    assert "debug.log" not in paths(discover(project))


def test_gitignore_negation_reincludes(project: Path):
    assert "keep.log" in paths(discover(project))


def test_nested_gitignore(nested_project: Path):
    found_paths = paths(discover(nested_project))
    assert "a/deep/visible.md" in found_paths
    assert "a/deep/secret.md" not in found_paths
    assert "a/top.md" in found_paths


def test_no_gitignore_flag(nested_project: Path):
    found_paths = paths(discover(nested_project, respect_gitignore=False))
    assert "a/deep/secret.md" in found_paths


def test_binary_files_skipped(project: Path):
    assert "logo.png" not in paths(discover(project))


def test_no_default_ignores(project: Path):
    found_paths = paths(discover(project, use_default_ignores=False))
    assert "package-lock.json" in found_paths


def test_include_filter(project: Path):
    found_paths = paths(discover(project, include=("*.py",)))
    assert all(p.endswith(".py") for p in found_paths)
    assert "src/app/main.py" in found_paths


def test_exclude_filter(project: Path):
    found_paths = paths(discover(project, exclude=("*.md",)))
    assert not any(p.endswith(".md") for p in found_paths)


def test_max_size(project: Path):
    found = discover(project, max_size=400)
    assert found.too_large > 0
    assert all(c.size <= 400 for c in found.files)


def test_single_file_target(project: Path):
    target = project / "src" / "app" / "main.py"
    found = discover(target)
    assert len(found.files) == 1
    assert read_text(found.files[0]) is not None


def test_missing_path_raises(tmp_path: Path):
    with pytest.raises(CtxpackError, match="no such path"):
        discover(tmp_path / "does-not-exist")


def test_results_are_sorted(project: Path):
    found = discover(project)
    assert [c.path for c in found.files] == sorted(c.path for c in found.files)


def test_read_text_ignores_cwd(project: Path, tmp_path, monkeypatch):
    """read_text must work from any working directory."""
    found = discover(project)
    monkeypatch.chdir(tmp_path)
    text = read_text(found.files[0])
    assert text is not None
    assert text


def test_read_text_returns_none_for_binary():
    candidate = Candidate("x.png", Path("/nonexistent/x.png"), 10, True)
    assert read_text(candidate) is None


def test_read_text_survives_missing_file():
    candidate = Candidate("gone.py", Path("/nonexistent/gone.py"), 10, False)
    assert read_text(candidate) is None


def test_candidate_properties(project: Path):
    found = discover(project)
    py = next(c for c in found.files if c.path.endswith("main.py"))
    assert py.ext == ".py"
    assert py.name == "main.py"
    assert py.depth == 2


def test_symlinked_directories_are_not_followed(tmp_path: Path):
    real = tmp_path / "real"
    (real / "inner").mkdir(parents=True)
    (real / "inner" / "file.txt").write_text("hi\n", encoding="utf-8")
    link_root = tmp_path / "linkroot"
    link_root.mkdir()
    os.symlink(real, link_root / "link")
    found = discover(link_root)
    assert found.symlinks == 1
    assert "link/inner/file.txt" not in paths(found)


def test_empty_directory(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert discover(empty).files == []


def test_counts_are_sane(project: Path):
    found = discover(project)
    assert found.directories > 0
    assert found.total_bytes > 0
    assert found.text_files == len(found.files)
