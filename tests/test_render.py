"""Rendering: content must survive the trip intact."""

from __future__ import annotations

import json
from xml.etree import ElementTree

import pytest

from ctxpack.errors import CtxpackError
from ctxpack.pack import Budget, Packer
from ctxpack.render import FORMATS, fence, language_for, render
from ctxpack.walk import discover


@pytest.fixture
def result(heuristic, project):
    return Packer(heuristic, Budget(total=20_000)).pack(discover(project))


# -- helpers -----------------------------------------------------------------


def test_language_detection():
    assert language_for("a/b.py") == "python"
    assert language_for("a/b.tsx") == "tsx"
    assert language_for("a/b.unknown") == ""
    assert language_for("Makefile") == ""


def test_fence_uses_three_ticks_by_default():
    assert fence("hello").startswith("```")


def test_fence_escalates_past_content_backticks():
    text = "here is a fence:\n```python\nprint(1)\n```\n"
    out = fence(text)
    assert out.startswith("````")
    assert out.rstrip().endswith("````")


def test_fence_escalates_past_long_runs():
    out = fence("``````\ncode\n``````")
    assert out.startswith("```````")


def test_fence_keeps_language_tag():
    assert fence("x", "python").startswith("```python")


# -- markdown ----------------------------------------------------------------


def test_markdown_contains_paths_and_content(result):
    text = render(result, "markdown")
    for doc in result.documents:
        assert doc.path in text
    assert "## Index" in text
    assert "## Files" in text


def test_markdown_states_the_budget(result):
    assert f"{result.budget:,} tokens" in render(result, "markdown")


def test_markdown_marks_truncated_documents(heuristic, tmp_path):
    (tmp_path / "big.py").write_text(
        "\n".join(f"def f{i}():\n    return {i}\n" for i in range(500)),
        encoding="utf-8",
    )
    packed = Packer(heuristic, Budget(total=2_000)).pack(discover(tmp_path))
    text = render(packed, "markdown")
    assert "*(truncated)*" in text


def test_markdown_with_backticks_in_content_is_well_formed(heuristic, tmp_path):
    (tmp_path / "doc.md").write_text(
        "# Doc\n\n```python\nprint('nested fence')\n```\n\nend.\n",
        encoding="utf-8",
    )
    packed = Packer(heuristic, Budget(total=8_000)).pack(discover(tmp_path))
    text = render(packed, "markdown")
    # The index uses a plain ``` fence; the file containing backticks must
    # escalate past it so its own fence cannot terminate the block early.
    fences = [line for line in text.splitlines() if line.startswith("```")]
    assert fences[0] == "```"
    assert "````markdown" in fences
    doc = next(d for d in packed.documents if d.path.endswith("doc.md"))
    assert text.count(doc.text.strip().splitlines()[0]) >= 1


# -- xml ---------------------------------------------------------------------


def test_xml_is_well_formed(result):
    ElementTree.fromstring(render(result, "xml"))


def test_xml_escapes_path_attributes(heuristic, tmp_path):
    (tmp_path / "a&b<c>.py").write_text("X = 1\n", encoding="utf-8")
    packed = Packer(heuristic, Budget(total=8_000)).pack(discover(tmp_path))
    xml = render(packed, "xml")
    root = ElementTree.fromstring(xml)  # must parse at all
    # The raw path is escaped in the source, and round-trips to the original.
    assert "a&b<c>.py" not in xml
    assert any(
        node.get("path") == "a&b<c>.py" for node in root.iter("entry")
    )


def test_xml_cdata_survives_closing_sequence(heuristic, tmp_path):
    (tmp_path / "tricky.py").write_text(
        'X = "]]> not the end of the section"\n', encoding="utf-8"
    )
    packed = Packer(heuristic, Budget(total=8_000)).pack(discover(tmp_path))
    xml = render(packed, "xml")
    root = ElementTree.fromstring(xml)
    found = [
        node.text or ""
        for node in root.iter("file")
    ]
    assert any("]]> not the end" in text for text in found)


def test_xml_round_trips_content(heuristic, tmp_path):
    content = 'def f():\n    return "quotes & <angles> and unicode é"\n'
    (tmp_path / "mod.py").write_text(content, encoding="utf-8")
    packed = Packer(heuristic, Budget(total=8_000)).pack(discover(tmp_path))
    root = ElementTree.fromstring(render(packed, "xml"))
    texts = [node.text for node in root.iter("file")]
    assert any(t and t.strip() == content.strip() for t in texts)


# -- json --------------------------------------------------------------------


def test_json_is_valid(result):
    payload = json.loads(render(result, "json"))
    assert payload["budget"] == result.budget
    assert len(payload["files"]) == len(result.documents)


def test_json_carries_metadata(result):
    payload = json.loads(render(result, "json"))
    assert payload["method"] == "estimate"
    assert payload["mode"] == "balanced"
    assert payload["index"]
    assert payload["summary"]


def test_json_content_survives(heuristic, tmp_path):
    content = 'X = "unicode é and \\ backslash and \n newline"\n'
    (tmp_path / "mod.py").write_text(content, encoding="utf-8")
    packed = Packer(heuristic, Budget(total=8_000)).pack(discover(tmp_path))
    payload = json.loads(render(packed, "json"))
    assert payload["files"][0]["content"].strip() == content.strip()


# -- tree --------------------------------------------------------------------


def test_tree_is_cheap(result):
    text = render(result, "tree")
    assert "root:" in text
    for doc in result.documents:
        assert doc.path in text
    # No file bodies, only the index.
    assert "```" not in text


def test_tree_without_manifest(heuristic, project):
    packed = Packer(
        heuristic, Budget(total=8_000, manifest="none")
    ).pack(discover(project))
    assert "root:" in render(packed, "tree")


# -- dispatch ----------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS)
def test_every_format_renders(result, fmt):
    assert render(result, fmt).strip()


def test_unknown_format_rejected(result):
    with pytest.raises(CtxpackError, match="unknown format"):
        render(result, "pdf")


def test_markdown_is_the_default(result):
    assert render(result) == render(result, "markdown")
