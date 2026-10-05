"""Ranking: which files deserve tokens."""

from __future__ import annotations

from pathlib import Path

from ctxpack.rank import rank_all, score_candidate
from ctxpack.walk import Candidate


def make(path: str, size: int = 1000) -> Candidate:
    return Candidate(path=path, abs_path=Path("/tmp") / path, size=size, is_binary=False)


def test_readme_outranks_ordinary_source():
    readme = score_candidate(make("README.md"))
    module = score_candidate(make("src/thing.py", 5_000))
    assert readme.score > module.score


def test_changelog_is_penalised():
    changelog = score_candidate(make("CHANGELOG.md", 5_000))
    guide = score_candidate(make("docs/guide.md", 5_000))
    assert changelog.score < guide.score


def test_license_is_penalised():
    assert score_candidate(make("LICENSE", 1_000)).score < score_candidate(
        make("src/core.py", 1_000)
    ).score


def test_source_root_beats_test_directory():
    real = score_candidate(make("src/app/handler.py"))
    test = score_candidate(make("tests/app/handler_test.py"))
    assert real.score > test.score


def test_depth_is_penalised():
    shallow = score_candidate(make("src/a.py"))
    deep = score_candidate(make("src/a/b/c/d/e/f/g.py"))
    assert shallow.score > deep.score


def test_generated_files_are_penalised():
    generated = score_candidate(make("src/api.generated.ts"))
    handwritten = score_candidate(make("src/api.ts"))
    assert generated.score < handwritten.score


def test_vendored_is_penalised():
    vendored = score_candidate(make("vendor/lib/index.js"))
    assert vendored.score < score_candidate(make("src/lib/index.js")).score


def test_anchor_files_score_well():
    assert score_candidate(make("src/app/index.ts")).score > score_candidate(
        make("src/app/helpers.ts")
    ).score


def test_structural_files_score_well():
    assert score_candidate(make("src/models.py")).score > score_candidate(
        make("src/utils2.py")
    ).score


def test_huge_files_are_penalised():
    assert score_candidate(make("src/gen.py", 900_000)).score < score_candidate(
        make("src/gen.py", 1_000)
    ).score


def test_signals_sum_to_score():
    scored = score_candidate(make("src/app/main.py", 4_000))
    assert sum(s.weight for s in scored.signals) == pytest_approx(scored.score)


def pytest_approx(value: float) -> float:
    import pytest

    return pytest.approx(value)


def test_explain_is_readable():
    text = score_candidate(make("README.md")).explain()
    assert "README.md" in text
    assert "readme" in text


def test_signal_renders_sign():
    from ctxpack.rank import Signal

    assert str(Signal("readme", 5.0)).startswith("+5.00")
    assert str(Signal("noise", -3.0)).startswith("-3.00")


def test_ranking_is_deterministic():
    candidates = [
        make("src/a.py"),
        make("src/b.py"),
        make("src/c.py"),
        make("README.md"),
    ]
    first = [s.path for s in rank_all(candidates)]
    second = [s.path for s in rank_all(list(reversed(candidates)))]
    assert first == second


def test_ranks_readme_first_in_real_project(project: Path):
    from ctxpack.walk import discover

    ranked = rank_all(discover(project).files)
    assert ranked[0].path == "README.md"


def test_ties_break_on_path():
    a = score_candidate(make("src/aaa.py"))
    b = score_candidate(make("src/bbb.py"))
    ranked = rank_all([a.candidate, b.candidate])
    assert [s.path for s in ranked] == ["src/aaa.py", "src/bbb.py"]


def test_score_is_finite_for_odd_paths():
    for path in ("no-extension", "a.b.c.d", "UPPER/Case.PY", "weird name.md", "x"):
        assert score_candidate(make(path)).score == score_candidate(make(path)).score


def test_uppercase_extension_recognised():
    assert score_candidate(make("src/Thing.PY")).score > score_candidate(
        make("src/Thing.zzzz")
    ).score
