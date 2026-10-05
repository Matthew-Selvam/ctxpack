"""Budget packing: the constraints that make a bundle useful."""

from __future__ import annotations

from pathlib import Path

import pytest

from ctxpack.errors import CtxpackError
from ctxpack.pack import Budget, Packer
from ctxpack.tokens import Tokenizer
from ctxpack.walk import discover


@pytest.fixture
def packer(heuristic: Tokenizer):
    def make(total: int = 20_000, **kwargs) -> Packer:
        mode = kwargs.pop("mode", "balanced")
        return Packer(heuristic, Budget(total=total, **kwargs), mode=mode)

    return make


def discovered(project: Path, **kwargs):
    return discover(project, **kwargs)


# -- budget validation -------------------------------------------------------


def test_budget_must_be_positive():
    with pytest.raises(CtxpackError, match="positive"):
        Budget(total=0)


def test_unknown_manifest_rejected():
    with pytest.raises(CtxpackError, match="manifest"):
        Budget(total=100, manifest="sideways")


def test_bad_dedupe_threshold_rejected():
    with pytest.raises(CtxpackError, match="dedupe_threshold"):
        Budget(total=100, dedupe_threshold=1.5)


def test_unknown_mode_rejected(heuristic):
    with pytest.raises(CtxpackError, match="unknown mode"):
        Packer(heuristic, Budget(total=100), mode="sideways")


def test_budget_derived_limits():
    budget = Budget(total=10_000, per_file_frac=0.2, per_dir_frac=0.5, per_ext_frac=0.6)
    assert budget.per_file == 2_000
    assert budget.per_dir == 5_000
    assert budget.per_ext == 6_000


def test_budget_derived_limits_have_floors():
    """Per-group floors keep a tiny budget from degenerating, with one exception."""
    budget = Budget(total=100)
    assert budget.per_file >= 200
    assert budget.per_ext >= 400
    assert budget.per_dir >= 400
    # The header reserve is the exception: it is capped at a quarter of the
    # budget, because a fixed floor there would swallow a small budget whole.
    assert budget.header <= budget.total // 4


def test_large_budget_gets_a_proportional_header():
    assert Budget(total=32_000).header == 960
    assert Budget(total=100_000).header == 2_000  # absolute ceiling


# -- the core guarantee ------------------------------------------------------


@pytest.mark.parametrize("budget", [2_000, 5_000, 20_000, 60_000])
def test_never_exceeds_budget(packer, project: Path, budget):
    result = packer(budget).pack(discovered(project))
    assert result.accounted <= result.budget
    assert result.over_budget == 0


@pytest.mark.parametrize("budget", [200, 300, 400, 600, 900, 1_500, 4_000])
@pytest.mark.parametrize("mode", ["balanced", "coverage", "depth"])
def test_never_exceeds_budget_at_any_size(packer, project: Path, budget, mode):
    """The budget must hold even when it is smaller than the fixed reserves.

    Regression: `header` had a `max(300, ...)` floor and `manifest_cap` a
    `max(500, ...)`, so at a 400-token budget the reserve alone exceeded the
    whole budget and the bundle came out 208 tokens over.
    """
    result = packer(budget, mode=mode).pack(discovered(project))
    assert result.accounted <= result.budget, (
        f"budget {budget} {mode}: accounted {result.accounted} "
        f"(manifest {result.manifest_tokens} + header {result.header_reserve} "
        f"+ content {result.content_tokens})"
    )


def test_header_reserve_never_exceeds_a_quarter_of_the_budget():
    assert Budget(total=400).header <= 100
    assert Budget(total=100).header <= 25
    assert Budget(total=32_000).header == 960


def test_index_cap_leaves_room_for_content():
    budget = Budget(total=2_000)
    assert budget.index_cap + budget.header <= budget.total
    assert budget.index_cap <= budget.manifest_cap


def test_manifest_gives_way_rather_than_breaking_a_tiny_budget(packer, project: Path):
    result = packer(300, manifest="full").pack(discovered(project))
    budget = Budget(total=300)
    assert result.manifest_tokens <= budget.index_cap
    assert result.accounted <= result.budget


def test_index_is_dropped_entirely_when_it_cannot_fit(packer, tmp_path: Path):
    """With a path list too long for the cap, the index goes rather than lies."""
    for i in range(120):
        (tmp_path / f"module_with_a_fairly_long_name_{i}.py").write_text(
            f"VALUE_{i} = {i}\n", encoding="utf-8"
        )
    result = packer(400, manifest="full").pack(discover(tmp_path))
    assert result.accounted <= result.budget
    assert result.manifest_tokens <= Budget(total=400).index_cap


def test_tiny_budgets_do_not_crash(packer, project: Path):
    result = packer(500).pack(discovered(project))
    assert result.accounted <= result.budget


def test_manifest_none_leaves_the_whole_budget_for_content(packer, project: Path):
    with_index = packer(3_000).pack(discovered(project))
    without = packer(3_000, manifest="none").pack(discovered(project))
    assert without.manifest_tokens == 0
    assert without.content_tokens >= with_index.content_tokens


def test_most_important_file_is_included(packer, project: Path):
    result = packer(20_000).pack(discovered(project))
    assert "README.md" in {d.path for d in result.documents}


def test_documents_are_ordered_by_rank(packer, project: Path):
    result = packer(30_000).pack(discovered(project))
    scores = [d.scored.score for d in result.documents]
    assert scores == sorted(scores, reverse=True)


# -- truncation --------------------------------------------------------------


def test_large_files_are_truncated(packer, project: Path):
    result = packer(3_000, per_file_frac=0.1).pack(discovered(project))
    big = next(d for d in result.documents if d.path.endswith("big.py"))
    assert big.truncated
    assert big.tokens < big.original_tokens
    assert big.saved > 0


def test_truncation_keeps_whole_lines(packer, project: Path):
    result = packer(3_000, per_file_frac=0.08).pack(discovered(project))
    for doc in result.truncated:
        # Every kept line came from the original file.
        assert "\n" in doc.text
        assert doc.lines <= doc.candidate.size


def test_truncation_is_marked(packer, project: Path):
    result = packer(3_000, per_file_frac=0.08).pack(discovered(project))
    for doc in result.truncated:
        assert "ctxpack truncated" in doc.text


def test_no_truncate_drops_instead(packer, project: Path):
    result = packer(3_000, per_file_frac=0.08, truncate=False).pack(
        discovered(project)
    )
    assert not result.truncated


# -- duplicates --------------------------------------------------------------


def test_near_duplicates_collapsed(packer, project: Path):
    result = packer(40_000).pack(discovered(project))
    packed = {d.path for d in result.documents}
    assert ("locales/en.py" in packed) != ("locales/fr.py" in packed)
    assert result.duplicates


def test_duplicate_records_similarity(packer, project: Path):
    result = packer(40_000).pack(discovered(project))
    for dup in result.duplicates:
        assert 0.85 <= dup.similarity <= 1.0
        assert dup.path and dup.of_path


def test_dedupe_can_be_disabled(packer, project: Path):
    result = packer(40_000, dedupe=False).pack(discovered(project))
    packed = {d.path for d in result.documents}
    assert "locales/en.py" in packed and "locales/fr.py" in packed
    assert not result.duplicates


def test_identical_files_but_for_name_collide(tmp_path: Path, packer):
    body = "\n".join(
        f"    value_number_{i} = some_distinctive_expression({i})" for i in range(60)
    )
    for name in ("one", "two"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "copy.py").write_text(
            f"CONTENT = '''\n{body}\n'''\n", encoding="utf-8"
        )
    result = packer(30_000).pack(discover(tmp_path))
    assert len(result.duplicates) == 1


def test_line_index_tolerates_gaps_in_document_ids():
    """The index is keyed by document id, so a skipped registration is harmless.

    Direct unit test of the invariant, because an integration test only catches
    this when a signature-less file happens to be selected first -- which is
    order-dependent and therefore a weak regression guard.
    """
    from ctxpack.pack import _LineIndex

    index = _LineIndex()
    index.add(0, {"a shared line of text"})
    index.add(2, {"a shared line of text"})  # doc 1 was never registered

    match = index.best_match({"a shared line of text"})
    assert match is not None
    doc_id, similarity = match
    assert doc_id in (0, 2)
    assert similarity == pytest.approx(1.0)


def test_line_index_reports_no_match_for_unrelated_text():
    from ctxpack.pack import _LineIndex

    index = _LineIndex()
    index.add(0, {"completely different content here"})
    assert index.best_match({"nothing whatsoever in common"}) is None
    assert index.best_match(set()) is None


def test_line_index_partial_similarity():
    from ctxpack.pack import _LineIndex

    index = _LineIndex()
    index.add(0, {"alpha line", "beta line", "gamma line", "delta line"})
    match = index.best_match({"alpha line", "beta line", "omega line", "psi line"})
    assert match is not None
    _, similarity = match
    assert 0.3 < similarity < 0.6  # 2 shared of 6 in the union


def test_signature_less_file_does_not_break_packing(packer, tmp_path: Path):
    """A file with no distinctive lines, ranked first, packs cleanly.

    The README here is made entirely of short lines, so its signature is empty,
    and it ranks first -- so it occupies document index 0 with nothing to
    register. This is the shape that used to desync the duplicate index and
    raise IndexError on a 23k-file repo. ``_LineIndex`` is now keyed by
    document id so the gap is harmless, and ``_select`` registers every
    document regardless; this test guards the behaviour end to end.
    """
    (tmp_path / "README.md").write_text("# Hi\n\n- a\n- b\n", encoding="utf-8")

    body = "\n".join(
        f"    distinctive_value_{i} = compute(something, meaningful)"
        for i in range(40)
    )
    for name in ("alpha", "beta"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "copy.py").write_text(
            f"CONTENT = '''\n{body}\n'''\n", encoding="utf-8"
        )

    result = packer(40_000).pack(discover(tmp_path))

    # The README really is first, and the copies really are collapsed.
    assert result.documents[0].path == "README.md"
    assert result.duplicates
    for dup in result.duplicates:
        assert dup.of_path in {d.path for d in result.documents}


# -- diversity caps ----------------------------------------------------------


def test_max_per_dir_respected(packer, project: Path):
    result = packer(60_000, max_per_dir=2).pack(discovered(project))
    counts: dict[str, int] = {}
    for doc in result.documents:
        counts[doc.dirname] = counts.get(doc.dirname, 0) + 1
    assert all(count <= 2 for count in counts.values())


def test_max_per_ext_respected(packer, project: Path):
    result = packer(60_000, max_per_ext=2).pack(discovered(project))
    counts: dict[str, int] = {}
    for doc in result.documents:
        counts[doc.ext] = counts.get(doc.ext, 0) + 1
    assert all(count <= 2 for count in counts.values())


def test_per_dir_share_respected(packer, project: Path):
    result = packer(20_000, per_dir_frac=0.25, max_per_dir=99).pack(
        discovered(project)
    )
    spend: dict[str, int] = {}
    for doc in result.documents:
        spend[doc.dirname] = spend.get(doc.dirname, 0) + doc.tokens
    cap = Budget(total=20_000, per_dir_frac=0.25).per_dir
    # A greedy loop may overshoot by at most the last file it accepted.
    for amount in spend.values():
        assert amount <= cap + result.budget * 0.12


def test_coverage_mode_spreads_across_directories(packer, project: Path):
    balanced = packer(6_000, mode="balanced").pack(discovered(project))
    coverage = packer(6_000, mode="coverage").pack(discovered(project))

    def dirs(result):
        return len({d.dirname for d in result.documents})

    assert dirs(coverage) >= dirs(balanced)


def test_depth_mode_is_pure_rank_order(packer, project: Path):
    result = packer(6_000, mode="depth").pack(discovered(project))
    scores = [d.scored.score for d in result.documents]
    assert scores == sorted(scores, reverse=True)


# -- manifest ----------------------------------------------------------------


def test_manifest_covers_every_discovered_file(packer, project: Path):
    result = packer(20_000).pack(discovered(project))
    assert len(result.manifest) == result.discovered


def test_manifest_none(packer, project: Path):
    result = packer(20_000, manifest="none").pack(discovered(project))
    assert result.manifest_tokens == 0
    assert result.manifest_text == ""


def test_manifest_paths_mode(packer, project: Path):
    result = packer(20_000, manifest="paths").pack(discovered(project))
    assert result.manifest_tokens > 0
    assert "bytes" not in result.manifest_text.splitlines()[0]


def test_manifest_is_trimmed_to_its_cap(packer, tmp_path: Path):
    """A big repo must not spend the whole budget on its own index."""
    for i in range(200):
        sub = tmp_path / f"pkg{i:03d}"
        sub.mkdir()
        (sub / "mod.py").write_text(f"VALUE_{i} = {i}\n" * 3, encoding="utf-8")
    result = packer(4_000, manifest="full").pack(discover(tmp_path))
    cap = Budget(total=4_000).manifest_cap
    assert result.manifest_tokens <= cap * 1.3


def test_unscanned_files_are_priced_and_labelled(packer, tmp_path: Path):
    for i in range(30):
        (tmp_path / f"file{i:02d}.py").write_text(f"X = {i}\n", encoding="utf-8")
    result = packer(20_000, max_scan=5).pack(discover(tmp_path))
    assert result.scanned <= 5
    assert result.unscanned > 0
    assert any(entry.estimated for entry in result.manifest)
    assert len(result.manifest) == 30


# -- reporting ---------------------------------------------------------------


def test_notes_explain_unused_budget(packer, project: Path):
    result = packer(30_000).pack(discovered(project))
    if result.accounted < result.budget * 0.85:
        assert result.notes
        assert any("unused" in note for note in result.notes)


def test_summary_accessors(packer, project: Path):
    result = packer(20_000).pack(discovered(project))
    assert result.content_tokens == sum(d.tokens for d in result.documents)
    assert result.truncated == [d for d in result.documents if d.truncated]


def test_result_records_tokenizer_provenance(packer, project: Path):
    result = packer(20_000).pack(discovered(project))
    assert result.method == "estimate"
    assert result.encoding == "o200k_base"
    assert result.mode == "balanced"


def test_empty_discovery(tmp_path: Path, packer):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = packer(10_000).pack(discover(empty))
    assert result.documents == []
    assert result.discovered == 0


def test_content_budget_accounting_adds_up(packer, project: Path):
    result = packer(4_000).pack(discovered(project))
    assert (
        result.content_tokens + result.manifest_tokens + result.header_reserve
        == result.accounted
    )
