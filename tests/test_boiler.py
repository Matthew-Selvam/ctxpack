"""Repeated-block detection and factoring.

The contract these tests defend is narrow and loud: factoring may lose
coverage, but it must never emit a marker that does not resolve, never rewrite a
file it did not change, and never produce different output twice.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ctxpack.boiler import (
    SHARED_SECTION_TITLE,
    Block,
    assign_ids,
    comment_style,
    factor,
    find_blocks,
    marker_for,
    render_shared,
    savings,
    savings_report,
)
from ctxpack.errors import CtxpackError
from ctxpack.tokens import Tokenizer

# A licence header in the shape every Apache-licensed repository uses: eleven
# comment lines, long enough to be worth hoisting, ending in a line that is not a
# statement opener so it can be followed directly by code.
LICENSE_CLAUSES = [
    "// Licensed under the Apache License, Version 2.0 (the \"License\");",
    "// you may not use this file except in compliance with the License.",
    "// You may obtain a copy of the License at",
    "//",
    "//     http://www.apache.org/licenses/LICENSE-2.0",
    "//",
    "// Unless required by applicable law or agreed to in writing, software",
    "// distributed under the License is distributed on an \"AS IS\" BASIS,",
    "// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
    "// See the License for the specific language governing permissions and",
    "// limitations under the License.",
]


def header(comment: str = "#", pad: str = "") -> str:
    """The repeated run: eleven clauses plus the blank line after them.

    The blank line is part of the block, not decoration around it. Extension
    keeps going while every occurrence agrees, and every file has the blank, so
    twelve lines is the honest answer and not eleven. ``pad`` appends trailing
    whitespace to the last clause, which matching ignores and extraction does not.

    Returned as block text -- lines joined with ``"\\n"`` and no trailing newline,
    exactly as :class:`Block.text` is defined.
    """
    lines = [line.replace("//", comment) for line in LICENSE_CLAUSES]
    return "\n".join([*lines[:-1], lines[-1] + pad, ""])


def file_header(comment: str = "#", pad: str = "") -> str:
    """The same run as a file prefix: block text, plus the newline ending it."""
    return f"{header(comment, pad)}\n"


PY_LICENSE = header()
LICENSE_LINES = 12


def module(index: int, head: str = file_header()) -> str:
    """A file with the shared header and a body unique to ``index``."""
    body = "\n".join(
        f"def function_{index}_{step}(alpha, beta):\n"
        f"    return alpha + beta + {step}"
        for step in range(12)
    )
    return f"{head}{body}\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Eight real files on disk that share a licence header."""
    root = tmp_path / "licenced"
    (root / "src").mkdir(parents=True)
    for index in range(8):
        (root / "src" / f"mod_{index}.py").write_text(module(index), encoding="utf-8")
    return root


def texts_of(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): p.read_text(encoding="utf-8")
        for p in sorted(root.rglob("*.py"))
    }


# -- detection ---------------------------------------------------------------


def test_finds_a_shared_licence_header(repo: Path):
    blocks = find_blocks(texts_of(repo))
    assert len(blocks) == 1
    block = blocks[0]
    assert block.text == PY_LICENSE
    assert block.lines == LICENSE_LINES
    assert block.files == tuple(f"src/mod_{i}.py" for i in range(8))
    assert block.occurrences == 8
    assert block.tokens > 0
    assert not block.in_one_file_only


def test_header_is_not_extended_into_the_body(repo: Path):
    """Greedy extension must stop where the files stop agreeing."""
    texts = texts_of(repo)
    lines = texts["src/mod_0.py"].split("\n")
    assert lines[LICENSE_LINES].startswith("def function_0_0")
    assert [block.lines for block in find_blocks(texts)] == [LICENSE_LINES]


def test_single_file_is_never_a_shared_block(repo: Path):
    texts = texts_of(repo)
    one = {"src/mod_0.py": texts["src/mod_0.py"]}
    assert find_blocks(one) == []
    # ...and the early exit must not have eaten the other files' evidence.
    assert len(find_blocks(texts)) == 1


def test_a_block_in_only_one_file_is_not_found(tmp_path: Path):
    solo = "\n".join(f"# unique note number {i} about this module" for i in range(6))
    texts = {
        "a.py": f"{solo}\n\nA = 1\n",
        "b.py": "# an entirely unrelated line of prose\nB = 2\n",
    }
    assert find_blocks(texts) == []
    # min_files=1 lets it through, which is how we can show that a run inside a
    # single file is *detected* and then declined by the conservatism rules
    # rather than simply never existing.
    found = find_blocks(texts, min_files=1)
    assert [b.occurrences for b in found] == [1]
    assert found[0].in_one_file_only
    factored, used = factor(texts, found)
    assert used == []
    assert factored == texts


def test_repeats_within_one_file_count_separately():
    """Four occurrences: two positions in each of two files.

    The tails differ between files on purpose. Extension stops at the first line
    where the occurrences disagree, so each run stops at the tail instead of
    swallowing the rest of the file, and both positions stay separate.
    """
    body = "\n".join(f"SHARED_CONSTANT_{i} = compute({i})" for i in range(6))
    texts = {
        "a.py": f"{body}\n\nONLY_IN_A = 1\n\n{body}\n",
        "b.py": f"{body}\n\nONLY_IN_B = 2\n\n{body}\n",
    }
    found = find_blocks(texts)
    # Six lines, and no trailing blank. `split` leaves a phantom empty element
    # for a newline-terminated file, and extension used to absorb it -- which made
    # the block seven lines and matched the second copy only because the phantom
    # stood in for the real blank that follows the first one.
    assert [(b.text, b.lines) for b in found] == [(body, 6)]
    assert found[0].occurrences == 4
    assert found[0].files == ("a.py", "b.py")

    factored, used = factor(texts, found)
    assert used[0].occurrences == 4
    assert all(text.count(marker_for("x.py", "B1", 6)) == 2 for text in factored.values())
    # And the terminating newline survives, since the phantom is no longer part
    # of any run.
    assert all(text.endswith("\n") for text in factored.values())


def test_min_lines_is_respected():
    """A two-line shared run is below the default and ignored.

    The surrounding lines are unique per file so the run is exactly two lines
    and cannot grow into the code that follows it.
    """
    texts = {
        f"m{i}.js": (
            f"const unique{i} = {i};\n"
            "// shared line one of two\n"
            "// shared line two of two\n"
            f"const tail{i} = {i};\n"
        )
        for i in range(5)
    }
    assert find_blocks(texts) == []
    assert find_blocks(texts, min_lines=3) == []
    assert [block.lines for block in find_blocks(texts, min_lines=2)] == [2]


def test_a_four_line_run_is_found_by_default():
    texts = {
        f"m{i}.js": (
            f"const unique{i} = {i};\n"
            "// shared line one of four\n"
            "// shared line two of four\n"
            "// shared line three of four\n"
            "// shared line four of four\n"
            f"const tail{i} = {i};\n"
        )
        for i in range(5)
    }
    assert [block.lines for block in find_blocks(texts)] == [4]


def test_max_lines_caps_the_run():
    header = "\n".join(f"# clause {i} of the shared notice" for i in range(30))
    texts = {f"m{i}.py": header + f"\n\nX{i} = 1\n" for i in range(5)}
    assert max(b.lines for b in find_blocks(texts, max_lines=6)) == 6


def test_punctuation_and_blank_blocks_are_not_blocks():
    """The substantive filter: `}` and `pass` repeat by coincidence."""
    noise = "\n".join(["}", "    pass", "});", "", "//"]) * 3
    texts = {f"m{i}.ts": noise + f"\nexport const x{i} = {i};\n" for i in range(20)}
    assert find_blocks(texts) == []


def test_trailing_whitespace_does_not_prevent_a_match():
    """Matching is rstrip-normalised; the extracted text is not."""
    texts = {
        "a.py": module(0, file_header(pad="   ")),
        "b.py": module(1),
        "c.py": module(2, file_header(pad="\t")),
    }
    found = find_blocks(texts)
    assert len(found) == 1
    assert found[0].lines == LICENSE_LINES
    assert found[0].files == ("a.py", "b.py", "c.py")
    # The block is taken from the first occurrence in path order, trailing
    # whitespace and all, so the copy is real source rather than a tidied one.
    assert found[0].text == header(pad="   ")
    assert found[0].text.split("\n")[10].endswith("License.   ")


def test_indentation_is_preserved_in_the_extracted_text():
    indented = "\n".join(f"    value_{i} = compute(alpha, {i})" for i in range(6))
    texts = {
        f"m{i}.py": f"def f{i}():\n{indented}\n    return value_{i}\n"
        for i in range(4)
    }
    found = find_blocks(texts)
    assert [b.text for b in found] == [indented]
    assert all(line.startswith("    ") for line in found[0].text.split("\n"))


# -- substitution ------------------------------------------------------------


def test_factor_hoists_the_header(heuristic: Tokenizer, repo: Path):
    texts = texts_of(repo)
    factored, used = factor(texts, find_blocks(texts))
    assert [b.lines for b in used] == [LICENSE_LINES]
    assert used[0].occurrences == 8
    assert used[0].files == tuple(sorted(texts))
    marker = marker_for("x.py", "B1", LICENSE_LINES)
    for path, text in factored.items():
        assert PY_LICENSE not in text
        assert text.startswith(marker)
        assert text.count("ctxpack: shared block") == 1
        # The body survives untouched behind the marker.
        index = path.removeprefix("src/mod_").removesuffix(".py")
        assert f"def function_{index}_0(alpha, beta):" in text
        assert text.endswith("    return alpha + beta + 11\n")


def test_total_tokens_drop_and_savings_agrees(heuristic: Tokenizer, repo: Path):
    texts = texts_of(repo)
    factored, used = factor(texts, find_blocks(texts))

    before = sum(heuristic.count(t).tokens for t in texts.values())
    after = sum(heuristic.count(t).tokens for t in factored.values())
    assert after < before

    measured = savings(heuristic, texts, factored, used)
    assert measured == before - after
    assert measured > 0

    report = savings_report(heuristic, texts, factored, used)
    assert report.tokens_before == before
    assert report.tokens_after == after
    assert report.tokens_saved == measured
    assert len(report.files_changed) == len(texts)
    assert report.projected > measured  # the markers cost something
    assert report.marker_cost > 0


def test_marker_is_present_and_the_body_appears_once(heuristic: Tokenizer, repo: Path):
    texts = texts_of(repo)
    factored, used = factor(texts, find_blocks(texts))
    section = render_shared(used, heuristic)

    markers = [
        line
        for text in factored.values()
        for line in text.split("\n")
        if "ctxpack: shared block" in line
    ]
    assert len(markers) == 8
    assert all(m == marker_for("x.py", "B1", LICENSE_LINES) for m in markers)

    assert section.count(PY_LICENSE) == 1
    assert all(PY_LICENSE not in text for text in factored.values())
    assert PY_LICENSE in texts["src/mod_0.py"]
    # Every id named in a file is defined in the section.
    assert SHARED_SECTION_TITLE in section
    assert "### B1 ·" in section


def test_files_without_substitutions_come_back_untouched(repo: Path):
    texts = texts_of(repo)
    texts["docs/readme.md"] = "# Docs\n\nNothing shared in here at all.\n"
    factored, _used = factor(texts, find_blocks(texts))
    assert factored["docs/readme.md"] == texts["docs/readme.md"]


def test_marker_language_follows_the_file(heuristic: Tokenizer):
    """Same six words, four comment syntaxes, four separate blocks.

    Each family gets two files so the block clears ``min_occurrences``; the
    marker then has to be spelled in that family's comment form.
    """
    clauses = "\n".join(f"shared clause number {i} of the notice" for i in range(6))
    families = {
        "#": ("py", "# "),
        "//": ("js", "// "),
        "--": ("sql", "-- "),
        "<!--": ("html", "<!-- "),
    }
    texts = {}
    for opener, (ext, prefix) in families.items():
        for index in range(2):
            header = "\n".join(prefix + line for line in clauses.split("\n"))
            suffix = " -->" if opener == "<!--" else ""
            texts[f"f{ext}{index}.{ext}"] = f"{header}{suffix}\n\nX{index} = 1\n"

    factored, used = factor(texts, find_blocks(texts), min_occurrences=2)
    assert len(used) == 4
    assert savings(heuristic, texts, factored, used) > 0
    ids = assign_ids(used)
    for path, text in factored.items():
        block = next(b for b in used if path in b.files)
        stem = Path(path).stem  # fpy0, fjs0, fsql0, fhtml0
        assert text.startswith(marker_for(path, ids[block], 7))
        assert "shared clause" not in text
        assert text.endswith(f"X{stem[-1]} = 1\n")
    assert factored["fhtml0.html"].startswith("<!-- [ctxpack: shared block")
    assert factored["fhtml0.html"].split("\n")[0].endswith("-->")
    assert factored["fsql0.sql"].startswith("-- [ctxpack: shared block")
    assert factored["fjs0.js"].startswith("// [ctxpack: shared block")
    assert factored["fpy0.py"].startswith("# [ctxpack: shared block")


@pytest.mark.parametrize(
    ("path", "opener"),
    [
        ("a.py", "#"),
        ("a.PY", "#"),
        ("a.js", "//"),
        ("a.tsx", "//"),
        ("a.go", "//"),
        ("a.rs", "//"),
        ("a.sql", "--"),
        ("a.lua", "--"),
        ("a.hs", "--"),
        ("a.html", "<!--"),
        ("a.xml", "<!--"),
        ("a.md", "<!--"),
        ("a.el", ";"),
        ("a.tex", "%"),
        ("Makefile", "#"),  # no extension at all
        ("weird.unknownext", "#"),  # unknown extension falls back
    ],
)
def test_comment_style_per_language(path: str, opener: str):
    opener_out, _closer = comment_style(path)
    assert opener_out == opener
    assert marker_for(path, "B1", 12).startswith(opener)


def test_a_block_too_rare_to_pay_for_its_marker_is_left_alone():
    """Two occurrences do not clear the default of three."""
    header = "\n".join(f"# clause {i} of the notice" for i in range(6)) + "\n"
    texts = {f"m{i}.py": f"{header}\nX{i} = 1\n" for i in range(2)}
    found = find_blocks(texts)
    assert [(b.lines, b.occurrences) for b in found] == [(7, 2)]

    factored, used = factor(texts, found)
    assert used == []
    assert factored == texts
    assert savings(_tokenizer(), texts, factored, used) == 0

    # ...and the caller can insist, and then pays for it.
    forced, used2 = factor(texts, found, min_occurrences=2)
    assert used2[0].occurrences == 2
    assert all(forced[p].startswith(marker_for(p, "B1", 7)) for p in forced)


def test_min_occurrences_rejects_nonsense():
    with pytest.raises(CtxpackError, match="at least 2"):
        factor({}, [], min_occurrences=1)


# -- conservatism ------------------------------------------------------------


def test_a_shared_block_inside_a_function_body_is_left_alone():
    """The conservatism test.

    Two files agree on nine consecutive lines that begin at column zero and end
    just before code that differs. Detecting that is correct; substituting it is
    not, and the whole run is declined rather than trimmed to its safe middle.
    """

    def source(tag: str) -> str:
        return (
            "import os\n\n"
            "def handler(request, ctx):\n"
            "    value = load(request)\n"
            "    step_one = transform(value, mode=1)\n"
            "    step_two = transform(value, mode=2)\n"
            "    step_three = transform(value, mode=3)\n"
            "    return respond(step_one, step_two, step_three)\n\n"
            f"def only_in_{tag}():\n    return 1\n"
        )

    texts = {f"h{i}.py": source(tag) for i, tag in enumerate(["a", "b", "c"])}
    found = find_blocks(texts)
    assert len(found) == 1
    assert found[0].occurrences == 3

    factored, used = factor(texts, found)
    assert used == []
    assert factored == texts
    assert savings_report(_tokenizer(), texts, factored, used).tokens_saved == 0


def test_a_docstring_is_never_substituted():
    """Deleting a docstring deletes ``__doc__``, so the run is declined."""
    docstring = (
        '"""Shared module documentation for this particular package."""\n'
        "\n"
        "import os\n"
        "\n"
        "A = os.sep\n"
    )
    texts = {
        "a.py": docstring + "\nB = 1\n",
        "b.py": docstring + "\nB = 2\n",
        "c.py": docstring + "\nB = 3\n",
    }
    factored, used = factor(texts, find_blocks(texts))
    assert used == []
    assert factored == texts


def test_conservatism_is_per_occurrence():
    """The same run, safe in one file and unsafe in another, is hoisted once.

    ``a.py`` has the run at top level between two blank lines. ``b.py`` has it
    in the middle of a function body, after a statement. Detection cannot tell
    them apart -- the lines are identical -- so the decision is per occurrence.
    """
    body = "\n".join(f"    step_{i} = transform(value, mode={i})" for i in range(6))
    clean = "\n".join(f"# shared clause {i} of the notice" for i in range(6))
    texts = {
        "a.py": f"{clean}\n\n{body}\n\n",
        "b.py": (
            f"def h(request):\n    value = load(request)\n{body}\n    return step_0\n"
        ),
    }
    blocks = find_blocks(texts)
    assert [(b.lines, b.files) for b in blocks] == [(6, ("a.py", "b.py"))]

    factored, used = factor(texts, blocks, min_occurrences=2)
    assert used[0].files == ("a.py",)
    assert used[0].occurrences == 1
    assert marker_for("a.py", "B1", 6) in factored["a.py"]
    assert factored["b.py"] == texts["b.py"]


# -- determinism -------------------------------------------------------------


def test_two_runs_are_byte_identical(heuristic: Tokenizer, repo: Path):
    texts = texts_of(repo)
    first_blocks = find_blocks(texts)
    first, first_used = factor(texts, first_blocks)
    second, second_used = factor(texts, find_blocks(texts))
    assert first == second
    assert first_used == second_used
    assert render_shared(first_used, heuristic) == render_shared(second_used, heuristic)


def test_ids_are_stable_regardless_of_input_order(heuristic: Tokenizer, tmp_path: Path):
    (tmp_path / "x").mkdir()
    for index in range(6):
        (tmp_path / "x" / f"m{index}.py").write_text(module(index), encoding="utf-8")
    texts = texts_of(tmp_path / "x")
    forwards = find_blocks(texts)
    backwards = find_blocks(dict(reversed(list(texts.items()))))
    assert forwards == backwards
    assert render_shared(forwards, heuristic) == render_shared(backwards, heuristic)


def test_ids_follow_savings_then_text():
    small = Block("a\nb\nc\nd\n", 4, ("x.py",), 9, 10)
    large = Block("e\nf\ng\nh\n", 4, ("y.py",), 5, 100)
    ids = assign_ids([small, large])
    assert ids[large] == "B1"
    assert ids[small] == "B2"
    # Ties fall back to the text, so numbering never depends on list order.
    tie_a = Block("aaa\n", 1, ("x.py",), 3, 10)
    tie_b = Block("bbb\n", 1, ("x.py",), 3, 10)
    assert assign_ids([tie_b, tie_a]) == {tie_a: "B1", tie_b: "B2"}


def test_markers_and_section_agree_on_ids(heuristic: Tokenizer, tmp_path: Path):
    """Ids are assigned from the same list both sides, so they cannot drift."""
    short = "\n".join(f"# short clause {i}" for i in range(8))
    long = "\n".join(f"# longer clause {i} of the notice text" for i in range(8))
    texts = {}
    for index in range(6):
        texts[f"m{index}.py"] = f"{short}\n\n{long}\n\nX{index} = 1\n"
    factored, used = factor(texts, find_blocks(texts))
    section = render_shared(used, heuristic)
    ids = assign_ids(used)
    assert len(set(ids.values())) == len(used)
    for block in used:
        assert f"### {ids[block]} ·" in section
        assert block.text in section
    named = {
        line.split("block ")[1].split(" ")[0]
        for text in factored.values()
        for line in text.split("\n")
        if "ctxpack: shared block" in line
    }
    assert named  # something was actually substituted
    for block_id in named:
        assert f"### {block_id} ·" in section


def test_factoring_twice_is_a_no_op(heuristic: Tokenizer, repo: Path):
    """Documented behaviour, not an accident.

    The second pass cannot find the block -- the marker is one line, and the
    block's full text no longer exists anywhere -- so it reports nothing and
    leaves the text alone. With a *different* block set this is not guaranteed.
    """
    texts = texts_of(repo)
    factored, used = factor(texts, find_blocks(texts))
    again, again_used = factor(factored, used)
    assert again == factored
    assert again_used == []


def test_re_factoring_does_not_explode(heuristic: Tokenizer, repo: Path):
    """Even if a marker joins a fresh window, output cannot grow without bound."""
    texts = texts_of(repo)
    current = texts
    sizes = [sum(heuristic.count(t).tokens for t in current.values())]
    for _ in range(3):
        current, _used = factor(current, find_blocks(current))
        sizes.append(sum(heuristic.count(t).tokens for t in current.values()))
    assert sizes == sorted(sizes, reverse=True) or sizes[-1] <= sizes[0]
    assert sizes[-1] <= sizes[0]


# -- edge cases --------------------------------------------------------------


def test_empty_input():
    assert find_blocks({}) == []
    assert factor({}, []) == ({}, [])
    assert render_shared([], _tokenizer()) == ""


def test_single_file():
    texts = {"only.py": module(0)}
    assert find_blocks(texts) == []
    assert factor(texts, find_blocks(texts)) == (texts, [])


def test_file_with_no_trailing_newline():
    """A file that does not end in a newline comes back that way.

    ``str.split("\n")`` leaves a trailing empty element for a file that does end
    in one, so rejoining puts it back; a file that does not end in one has no
    such element and must not acquire one.
    """
    header = "\n".join(f"# clause {i} of the notice" for i in range(6)) + "\n"
    texts = {f"m{i}.py": f"{header}\nX{i} = 1" for i in range(4)}
    factored, used = factor(texts, find_blocks(texts))
    assert used
    for path, text in factored.items():
        assert not text.endswith("\n")
        assert text.split("\n")[0] == marker_for(path, "B1", 7)
        assert text.split("\n")[-1] == f"X{path[1]} = 1"


def test_crlf_input_keeps_crlf():
    header = "\n".join(f"# clause {i} of the notice" for i in range(6))
    texts = {
        "win.py": header.replace("\n", "\r\n") + "\r\n\r\nX = 1\r\n",
        "unix.py": header + "\n\nY = 2\n",
    }
    factored, _used = factor(texts, find_blocks(texts))
    assert "\r\n" in factored["win.py"]
    assert "\r\r" not in factored["win.py"]
    assert factored["win.py"].endswith("\r\n")
    assert "\r" not in factored["unix.py"]


def test_unicode_content():
    header = "\n".join(
        [
            "# Ünïcödé çømment wîth àccents and 漢字",
            "# второй комментарий тоже здесь",
            "# שלום עולם כולו כאן",
            "# emojis are fine too: 🎯 🚀 ✨",
            "# a fifth line to reach the minimum",
        ]
    )
    texts = {f"m{i}.py": f"{header}\n\nX{i} = 1\n" for i in range(4)}
    found = find_blocks(texts)
    # Five comment lines plus the blank separator, which is part of the run.
    assert [(b.lines, b.occurrences) for b in found] == [(6, 4)]
    assert found[0].text == header + "\n"
    factored, used = factor(texts, found)
    assert used[0].occurrences == 4
    assert all(header not in text for text in factored.values())
    assert "# Ünïcödé" in render_shared(used, _tokenizer())


def test_file_of_only_blank_lines():
    texts = {
        "blank.py": "\n\n\n\n\n",
        "empty.py": "",
        "a.py": module(0),
        "b.py": module(1),
        "c.py": module(2),
    }
    found = find_blocks(texts)
    assert len(found) == 1
    factored, _used = factor(texts, found)
    assert factored["blank.py"] == "\n\n\n\n\n"
    assert factored["empty.py"] == ""


def test_texts_is_not_mutated():
    texts = {f"m{i}.py": module(i) for i in range(4)}
    before = dict(texts)
    factor(texts, find_blocks(texts))
    assert texts == before


# -- render_shared -----------------------------------------------------------


def test_shared_section_carries_the_warning(heuristic: Tokenizer, repo: Path):
    texts = texts_of(repo)
    factored, used = factor(texts, find_blocks(texts))
    section = render_shared(used, heuristic)
    assert section.startswith(f"## {SHARED_SECTION_TITLE}")
    assert "reading aid, not source" in section
    assert "must not be executed" in section
    assert "must not be executed" not in factored["src/mod_0.py"]
    assert section.endswith("\n")
    # Priced, so a caller can budget for the section it is about to append.
    assert heuristic.count(section).tokens > 0


def test_shared_section_survives_backticks(heuristic: Tokenizer):
    body = "\n".join(f"# ```python {i}```" for i in range(6))
    texts = {f"m{i}.py": f"{body}\n\nX{i} = 1\n" for i in range(3)}
    section = render_shared(factor(texts, find_blocks(texts))[1], heuristic)
    assert "````" in section


def test_render_shared_on_a_hand_built_block(heuristic: Tokenizer):
    block = Block(text="alpha\nbeta\ngamma\ndelta\n", lines=4, files=("x.py", "y.py"),
                  occurrences=7, tokens=12)
    section = render_shared([block], heuristic)
    assert "### B1 · 4 lines" in section
    assert "7 occurrences in 2 files" in section
    assert "```python\nalpha" in section


# -- validation --------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_lines": 0}, "min_lines"),
        ({"min_files": 0}, "min_files"),
        ({"min_lines": 10, "max_lines": 5}, "max_lines"),
    ],
)
def test_bad_find_blocks_arguments(kwargs, message):
    with pytest.raises(CtxpackError, match=message):
        find_blocks({"a.py": module(0)}, **kwargs)


@pytest.mark.parametrize("value", [0, 1, -3])
def test_bad_min_occurrences(value):
    with pytest.raises(CtxpackError, match="min_occurrences"):
        factor({}, [], min_occurrences=value)


# -- performance -------------------------------------------------------------


def test_two_hundred_files_is_fast():
    header = "\n".join(f"// clause {i} of the shared licence notice" for i in range(14))
    texts = {}
    for index in range(200):
        body = "\n".join(
            f"    value_{index}_{step} = compute({step}, alpha, beta)\n"
            f"    emit(value_{index}_{step})"
            for step in range(45)
        )
        texts[f"pkg/mod_{index:03d}.py"] = f"{header}\n\n{body}\n"

    started = time.perf_counter()
    blocks = find_blocks(texts)
    elapsed = time.perf_counter() - started

    assert blocks, "expected the shared header to be found"
    assert elapsed < 1.0, f"find_blocks took {elapsed:.2f}s on 200 files"


def _tokenizer() -> Tokenizer:
    """A counting tokenizer for assertions that do not need a fixture."""
    return Tokenizer(mode="never", calibration=1.0)


# ---------------------------------------------------------------------------
# regressions found by review
# ---------------------------------------------------------------------------


def _shared_header_fixtures(count: int) -> dict[str, str]:
    body = "\n".join(f"# shared licence line {i}" for i in range(12))
    out = {}
    for i in range(count):
        out[f"pkg{i % 5}/mod_{i}.py"] = (
            f"# unique header {i}\n{body}\n\ndef thing_{i}():\n    return {i}\n"
        )
    return out


@pytest.mark.parametrize("ending", ["\n", "\r\n"])
def test_factoring_preserves_the_terminating_newline(ending: str):
    """A factored file must still end the way it started.

    Regression: the run of a block ending at end-of-file absorbed the phantom
    empty element that `split` leaves for a newline-terminated file, and the
    rewrite then consumed it as part of the run. The file came back with no
    final newline -- silent data loss on every file the feature touched.
    """
    body = "\n".join([f"# unique {i}" for i in range(3)])
    shared = "\n".join(
        ["# shared start", "shared one here", "shared two here",
         "shared three here", "shared four here"]
    )
    texts = {}
    for i in range(4):
        lines = [f"# unique {i}-{tag}" for tag in body.splitlines()[:1]]
        lines += [*shared.splitlines(), ""]
        texts[f"f{i}.py"] = ending.join(lines) + ending

    blocks = find_blocks(texts)
    factored, used = factor(texts, blocks, min_occurrences=2)
    assert used, "expected the shared block to be factored"
    for path, text in factored.items():
        assert text.endswith(ending), (
            f"{path} lost its {ending!r} terminator: {text[-40:]!r}"
        )


def test_block_lines_matches_the_rendered_block():
    """`Block.lines` must agree with the text a reader actually sees.

    A block of N source lines joins to N-1 separators, so `text.split("\\n")`
    always yields exactly N elements -- including when the last line is blank
    and the text therefore ends in a newline.
    """
    texts = _shared_header_fixtures(4)
    for block in find_blocks(texts):
        assert block.lines == len(block.text.split("\n")), (
            f"lines={block.lines} but the text accounts for "
            f"{len(block.text.split(chr(10)))}"
        )


def test_occurrence_count_is_independent_of_block_position():
    """A block must be counted the same wherever it sits in a file.

    Before the phantom-line fix, a trailing copy matched only when a real blank
    line happened to follow the first copy, so occurrence counts depended on
    position within the file.
    """
    unit = "\n".join(f"SHARED_LINE_{i} = {i}" for i in range(6))
    first = f"{unit}\n\nONLY_A = 1\n\n{unit}\n"   # copy at line 0, copy at EOF
    second = f"ONLY_B = 2\n\n{unit}\n\n{unit}\n"  # copy mid-file, copy last
    texts = {"a.py": first, "b.py": second}
    blocks = find_blocks(texts)
    assert blocks
    # Two copies in each file, at different positions.
    assert blocks[0].occurrences == 4


def test_locate_is_linear_in_file_count():
    """The anchor search must not degrade quadratically with repo size.

    Regression: the search walked every file and then re-scanned the *global*
    anchor hit list for each one. 800 files took 15.2s where 0.4s suffices,
    and on a repo of generated files that is minutes of stall.
    """
    import time

    # A 4-line pattern repeating every 97 lines maximises anchor collisions,
    # which is what made the old re-scan quadratic.
    texts = {}
    for n in (120, 480):
        texts = {}
        for i in range(n):
            texts[f"f{i}.py"] = "\n".join(
                f"# unique {i} {j}" if j % 97 == 0 else f"    shared_{j % 4} = {j}"
                for j in range(200)
            ) + "\n"
        blocks = find_blocks(texts)
        start = time.perf_counter()
        factor(texts, blocks)
        elapsed = time.perf_counter() - start
        if n == 120:
            first_elapsed = elapsed

    # 4x the files must not cost anywhere near 16x the time.
    assert elapsed < 8 * (first_elapsed + 0.05), (
        f"{first_elapsed:.2f}s -> {elapsed:.2f}s for 4x the files: superlinear"
    )
