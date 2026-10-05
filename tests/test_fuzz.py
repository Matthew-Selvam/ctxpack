"""Fuzzing and property tests for the output layer.

Everything in :mod:`ctxpack.render` is judged on whether it survives an
arbitrary source file, and getting that wrong never raises: a fence that
closes early, XML that will not parse or a marker pointing at nothing all
produce a bundle that *looks* fine and quietly misleads whoever reads it.

So this module does not test the renderers against a handful of payloads
hand-picked once. It builds a corpus of deliberately hostile inputs --
markdown fence breakers, XML/CDATA hazards, the characters XML 1.0 forbids,
paths that are punctuation soup, and a seeded random fuzzer on top -- packs
each one for real, and asserts the invariants that must hold for all of them:

1. the XML renders and parses, always;
2. the JSON renders, loads, and round-trips what was on disk;
3. fenced payloads stay inside their fence, whole;
4. every path that went in comes back out;
5. no renderer ever raises;
6. rendering the same result twice is byte-identical;
7. gitignore patterns compile and match in bounded time.

Every generator is seeded from a name via blake2b, and every failure message
carries that seed plus a repr of the input. To re-fuzz with different random
cases, set ``CTXPACK_FUZZ_SEED`` to any integer::

    CTXPACK_FUZZ_SEED=1234 python -m pytest tests/test_fuzz.py

To confirm a reported failure replays identically, run it twice and diff::

    python -m pytest tests/test_fuzz.py -q > a.txt 2>&1
    python -m pytest tests/test_fuzz.py -q > b.txt 2>&1
    diff a.txt b.txt

``_case("some-label")`` takes a seed derived from the label, so the tests
that do not read the environment are reproducible across machines; the
hand-built payloads (the fence breakers, the character classes, the pattern
list) are fixed constants and do not vary at all.

No hypothesis, no new dependencies. The corpus is bounded to 323 cases so the
module runs in about four seconds, and it stays that way on four Python
versions.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import re
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import pytest

from ctxpack.errors import CtxpackError
from ctxpack.pack import (
    MODES,
    Budget,
    Document,
    ManifestEntry,
    Packer,
    PackResult,
    SharedBlock,
)
from ctxpack.rank import Scored
from ctxpack.render import (
    FORMATS,
    _cdata,
    _label,
    _xml_safe,
    fence,
    language_for,
    render,
)
from ctxpack.tokens import Tokenizer
from ctxpack.walk import Candidate, IgnoreStack, compile_pattern, discover, read_text

#: Deterministic counting, so a failure never depends on whether tiktoken
#: happens to be installed in the environment that ran the suite.
TOKENIZER = Tokenizer(mode="never", calibration=1.0)


# --------------------------------------------------------------------------
# Failure reporting
# --------------------------------------------------------------------------


def seed_for(label: str) -> int:
    """A stable seed for a test name.

    :func:`hash` is salted per interpreter, so it cannot be used for anything
    that has to replay tomorrow.
    """
    return int.from_bytes(hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest(), "big")


def overflow_seed() -> int | None:
    """``CTXPACK_FUZZ_SEED`` as an int, or ``None`` when unset or unusable."""
    raw = os.environ.get("CTXPACK_FUZZ_SEED")
    if raw is None:
        return None
    try:
        return int(raw, 0)
    except ValueError:
        return None


def clip(value: object, limit: int = 400) -> str:
    """A repr that stays readable when the input is a megabyte of backticks."""
    body = repr(value)
    if len(body) <= limit:
        return body
    return f"{body[:limit]}...<+{len(body) - limit} chars>"


def blow_up(test: str, seed: int, payload: object, reason: str) -> None:
    """Fail with enough context to replay the case byte for byte."""
    raise AssertionError(
        f"\n  test:   {test}\n"
        f"  seed:   {seed}  (replay with CTXPACK_FUZZ_SEED={seed})\n"
        f"  input:  {clip(payload)}\n"
        f"  why:    {reason}\n"
    )


def flag(test: str, label: str, seed: int, reason: str) -> None:
    """Same, for a case identified by name rather than by inline payload."""
    raise AssertionError(
        f"\n  test:   {test}\n"
        f"  case:   {label}\n"
        f"  seed:   {seed}  (replay with CTXPACK_FUZZ_SEED={seed})\n"
        f"  why:    {reason}\n"
    )


# --------------------------------------------------------------------------
# Wall-clock guard
# --------------------------------------------------------------------------

_HAVE_ALARM = hasattr(signal, "SIGALRM") and sys.platform != "win32"


class _Deadline(Exception):
    """Internal: a guarded call overran its limit."""


def bounded(fn, *args, limit: float, **kwargs) -> float | None:
    """Call ``fn(*args, **kwargs)``; return elapsed seconds, or ``None`` if it overran.

    Asserting on elapsed time alone would be flaky on a loaded CI box, and
    letting the call run unbounded would hang the suite instead of failing
    it. Bounding it means a genuine backtracking blow-up shows up as a
    failure with a seed attached, not as a cancelled job.
    """
    if not _HAVE_ALARM:
        pytest.skip("wall-clock guard needs SIGALRM")

    def on_alarm(signum, frame):
        raise _Deadline

    import time

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, limit)
    started = time.perf_counter()
    try:
        fn(*args, **kwargs)
    except _Deadline:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    return time.perf_counter() - started


# --------------------------------------------------------------------------
# Hostile vocabulary
#
# Every payload is written with escapes on purpose: a literal control
# character in this file would be unreadable in a traceback and ambiguous to
# a reviewer. Nothing here survives as itself by accident.
# --------------------------------------------------------------------------

#: Things that terminate, or try to terminate, a markdown code fence.
FENCE_BREAKERS: tuple[str, ...] = (
    "```",
    "````",
    "~~~~~",
    "``````",
    "`````````",
    "`",
    "``",
    "```python",
    "```python\nprint(1)\n```",
    "~~~\n~~~\n~~~",
    "   ```",
    "\t```",
    "\t``````",
    "```\n```\n```",
    "text ``` then ```` then ~~~~~",
    "```\nunterminated fence",
    "x\n```",
    "\n```\n",
    "``` ``` ```",
    "a```b```c",
    "<!-- ``` -->",
    "```]]>```",
    "- ```\n- a list item",
    "    ```\n    four spaces of indent",
    "> ```\n> blockquoted",
    "```{.python}",
    "``` \n",
)

#: Sequences with meaning to an XML parser, in every position that matters.
XML_HAZARDS: tuple[str, ...] = (
    "]]>",
    "]]]>",
    "]]]]>",
    "]]]]]]>",
    "<![CDATA[",
    "<![CDATA[]]]]><![CDATA[>",
    "]]>&gt;",
    "<!--",
    "-->",
    "--",
    "<?xml version='1.0'?>",
    "<!DOCTYPE ctxpack [<!ENTITY boom 'v'>]>",
    "<entry path='x'/>",
    "<file path=\"y\">",
    "&",
    "<",
    ">",
    '"',
    "'",
    "&amp;",
    "&lt;",
    "&#x0;",
    "]]",
    "]]>]]>",
    "]]> <![CDATA[ ]]>",
    "<>&\"'",
    "<>&\"'\n]]>",
)

#: Characters XML 1.0 forbids outright. CDATA is not an escape hatch for
#: these, which is what ``_xml_safe`` exists to handle.
FORBIDDEN_CHARS: tuple[str, ...] = (
    "\x00",
    "\x01",
    "\x02",
    "\x03",
    "\x04",
    "\x05",
    "\x06",
    "\x07",
    "\x08",
    "\x0b",
    "\x0c",
    "\x0e",
    "\x0f",
    "\x10",
    "\x11",
    "\x12",
    "\x13",
    "\x14",
    "\x15",
    "\x16",
    "\x17",
    "\x18",
    "\x19",
    "\x1a",
    "\x1b",
    "\x1c",
    "\x1d",
    "\x1e",
    "\x1f",
    "\ufffe",
    "\uffff",
    "\ufdd0",
    "\ufdef",
)

#: The whole C0 range at once, plus the noncharacters, the way a terminal
#: log dump or a Latin-1 file would actually deliver them.
FORBIDDEN_RUNS: tuple[str, ...] = (
    "".join(chr(code) for code in range(0x00, 0x09)),
    "\x0b\x0c",
    "".join(chr(code) for code in range(0x0E, 0x20)),
    "\ufffe\uffff\ufdd0\ufdef",
    "a\x01b\x1fc\x7fd",
    "\x1b[31mred\x1b[0m\x1b]0;title\x07",
)

#: Edge cases for ``_cdata`` itself, where ``]]>`` splitting has to be exact.
CDATA_EDGES: tuple[str, ...] = (
    "",
    "]",
    "]]",
    "]]]",
    ">",
    "a]",
    "x]",
    "x]]",
    "]]>",
    "]]]>",
    "]]]]>",
    "a]b]]>c",
    "]]>]]>]]>",
    "]]]]><![CDATA[>",
    "]]><![CDATA[]]>",
    "<![CDATA[]]]>",
    "\n]]>\n",
    "]]]]]]]]]]>",
    "no hazard at all",
)

#: Shapes a source file takes that are not really source.
CONTENT_SHAPES: tuple[str, ...] = (
    "",
    " ",
    "\n",
    "\n\n\n",
    "\t\t\t",
    "no trailing newline",
    "trailing spaces   \n",
    "crlf\r\nsecond\r\n",
    "mixed\r\nlf\nand\rlf\r",
    "\r",
    "\r\r",
    "x" * 5000,
    "a" * 300 + " " + "b" * 300,
    "\n".join(f"line {i}" for i in range(200)),
    "".join(f"    {'  ' * i}if x{i}:\n" for i in range(40)) + "        " + "pass\n",
    "def f():\n    return {\n" + "".join(f"        {i}: '{i}',\n" for i in range(120)) + "    }\n",
    "\ufeffbom then text",
    "\u2028line separator\u2029paragraph separator\n",
    "\xa0non-breaking spaces\n",
    "\x7fdel",
    "unicode: \u00e9 \u65e5\u672c\u8a9e \U0001f389 \U0010ffff\n",
    "]]>" * 100,
    "]]]]" * 60,
    "`<tag attr=\"value\">`" * 60,
    "#" * 200,
    "\x1b[31mred\x1b[0m",
    "a\rb",
    "tab\tand\tspaces\n",
    "json-ish {\"a\": [1, 2, {\"b\": null}], \"c\": \"]]>\"}\n",
    "'''\n'''\n",
    '"' * 50,
    "'" * 50,
    "*" * 60,
    "=" * 60 + "\n",
)

#: Byte sequences that are not valid UTF-8. ``read_text`` decodes with
#: ``errors="replace"``, so these become U+FFFD rather than raising -- which
#: is itself the property worth pinning.
BYTE_HAZARDS: tuple[tuple[str, bytes], ...] = (
    ("invalid-lead", b"\xff\xfe\xfd"),
    ("valid-accents", b"caf\xc3\xa9\n"),
    ("wtf8-surrogate", b"\xed\xa0\x80"),
    ("wtf8-surrogate-2", b"\xed\xbf\xbf"),
    ("above-max-plane", b"\xf4\x90\x80\x80"),
    ("overlong-nul", b"\xc0\x80"),
    ("lone-continuation", b"\x80"),
    ("line-separators", b"\xe2\x80\xa8\xe2\x80\xa9\n"),
    ("nul-byte", b"X = 1\n\x00\x00binary\n"),
    ("bom-then-junk", b"\xef\xbb\xbf\xff"),
    ("truncated-multibyte", b"ok\n\xc3"),
    ("only-ff", b"\xff" * 40),
    ("utf16-le-bom", b"\xff\xfeh\x00i\x00"),
)

#: File *names*. Not paths: everything here goes one level deep, where the
#: only unrepresentable characters are "/" and NUL.
PATH_HAZARDS: tuple[str, ...] = (
    "plain.py",
    "with space.py",
    "quote'name.py",
    'quote"name.py',
    "amp&and.py",
    "amp&amp;.py",
    "lt<gt>.py",
    "close]bracket.py",
    "open[bracket.py",
    "no-extension",
    "just.a.py",
    "...",
    "....",
    "-leading-dash.py",
    "--flag.py",
    "dash-in-middle-.py",
    "\u65e5\u672c\u8a9e.py",
    "e\u0301.py",
    "emoji\U0001f389.py",
    "tab\tname.py",
    "back`tick.py",
    "three```ticks.py",
    "tick`and`tick.py",
    "trailing`backtick.py",
    "star*.py",
    "q?.py",
    "semi;colon.py",
    "pipe|bar.py",
    "colon:name.py",
    "pct%20.py",
    "dollar$var.py",
    "hash#tag.py",
    "at@sign.py",
    "tilde~name.py",
    "plus+name.py",
    "paren(s).py",
    "caret^name.py",
    "{brace}.py",
    "a" * 120 + ".py",
    "UPPER.PY",
    "x.Py",
    ".hidden.py",
    "trailingdot.",
    "new\nline.py",
    "nl\n```ticks.py",
    "nl\n## Files.py",
    "nl\n\n```",
    "double\n\n```\n",
    "trailing\n",
    "\n",
    "```",
    "````",
    "bom\ufeffname.py",
    "zero\u200bwidth.py",
    "rtl\u202ename.py",
    "not-a-dot.",
    "dot.dot.dot",
)

#: Characters drawn on by the random fuzzer: ordinary text so payloads stay
#: readable, adversarial text so the interesting paths get taken.
NASTY_ALPHABET: tuple[str, ...] = tuple(
    "abcXYZ019 \t\n\r"
    "`~*_?![](){}<>&\"'/\\|;#$%@-_=+,.:"
    "]]><![CDATA[<!---->"
    + "".join(chr(code) for code in range(0x00, 0x20))
    + "".join(chr(code) for code in (0x7F, 0x85, 0x9F, 0xA0, 0x2028, 0x2029))
    + "\ufffd\ufffe\uffff\ufeff\ufdd0"
    + "\u00e9\u65e5\u672c\u8a9e\U0001f389\U0010ffff"
)

_NAME_ALPHABET = "abcXY_-.\u00e9\u65e5`\n\t*?[]<>&\""


def _rand_name(rng: random.Random) -> str:
    """A random legal single-component file name."""
    while True:
        stem = "".join(rng.choice(_NAME_ALPHABET) for _ in range(rng.randrange(1, 12)))
        if stem.strip(".") and "/" not in stem and "\x00" not in stem:
            return stem


def _rand_text(rng: random.Random) -> str:
    length = rng.randrange(0, 260)
    return "".join(rng.choice(NASTY_ALPHABET) for _ in range(length))


def _rand_bytes(rng: random.Random) -> bytes:
    """Random bytes, biased towards ones that are not valid UTF-8."""
    length = rng.randrange(0, 200)
    if rng.random() < 0.5:
        return bytes(rng.randrange(0, 256) for _ in range(length))
    return _rand_text(rng).encode("utf-8", "surrogatepass")


def _rand_files(rng: random.Random) -> tuple[tuple[str, bytes], ...]:
    count = rng.randrange(1, 4)
    files = []
    for _ in range(count):
        name = _rand_name(rng)
        if rng.random() < 0.25:
            name = name + rng.choice(PATH_HAZARDS)
        if rng.random() < 0.4:
            name = "sub/dir/" + name
        data = _rand_bytes(rng)
        files.append((name, data))
    return tuple(files)


# --------------------------------------------------------------------------
# The corpus
# --------------------------------------------------------------------------

BUDGETS = (20_000, 2_000, 1_000, 200_000)
MANIFESTS = ("full", "paths", "none")


@dataclass(frozen=True)
class Case:
    """One hostile project: files to write, plus how to pack them."""

    label: str
    files: tuple[tuple[str, bytes], ...]
    budget: int = 20_000
    mode: str = "balanced"
    manifest: str = "full"

    @property
    def seed(self) -> int:
        return seed_for(self.label)


_CASES_MADE = itertools.count()


def _case(
    label: str,
    files,
    *,
    budget: int | None = None,
    mode: str | None = None,
    manifest: str | None = None,
) -> Case:
    """A case whose flags rotate, so the corpus covers every combination."""
    index = next(_CASES_MADE)
    encoded = tuple(
        (name, data if isinstance(data, bytes) else data.encode("utf-8", "surrogatepass"))
        for name, data in files
    )
    return Case(
        label=label,
        files=encoded,
        budget=budget if budget is not None else BUDGETS[index % len(BUDGETS)],
        mode=mode if mode is not None else MODES[index % len(MODES)],
        manifest=manifest if manifest is not None else MANIFESTS[index % len(MANIFESTS)],
    )


def build_corpus() -> list[Case]:
    """Every hand-designed hostile case, plus the seeded random fuzz."""
    cases: list[Case] = []

    # -- markdown fence breakers, alone and surrounding ordinary text
    for index, payload in enumerate(FENCE_BREAKERS):
        cases.append(
            _case(f"fence/{index}", [("a.py", payload), ("b.md", f"lead\n{payload}\ntail\n")])
        )

    # -- XML and CDATA hazards
    for index, payload in enumerate(XML_HAZARDS):
        cases.append(
            _case(
                f"xml/{index}",
                [("x.py", f"X = 1\n{payload}\n"), ("y.xml", f"<r>{payload}</r>\n")],
            )
        )

    # -- characters XML 1.0 forbids, one per case and in bulk
    for index, char in enumerate(FORBIDDEN_CHARS):
        cases.append(
            _case(
                f"forbidden/{index}",
                [("c0.py", f"X = 1\n{char}\n"), ("c0.md", f"text {char} text\n")],
            )
        )
    for index, run in enumerate(FORBIDDEN_RUNS):
        cases.append(_case(f"forbidden-run/{index}", [("run.py", f"X = 1\n{run}\n")]))

    # -- CDATA splitting edge cases, driven straight through the renderer
    for index, payload in enumerate(CDATA_EDGES):
        cases.append(_case(f"cdata/{index}", [("edge.py", payload), ("edge.xml", payload)]))

    # -- content shapes
    for index, payload in enumerate(CONTENT_SHAPES):
        cases.append(
            _case(
                f"shape/{index}",
                [
                    ("shape.py", payload),
                    ("shape.md", payload),
                    ("shape.txt", payload),
                ],
            )
        )

    # -- byte sequences that are not valid UTF-8
    for label, payload in BYTE_HAZARDS:
        cases.append(_case(f"bytes/{label}", [("bytes.py", payload), ("bytes.txt", payload)]))

    # -- one hostile name per case, with an ordinary companion so there is
    #    always at least one file to compare against
    for index, name in enumerate(PATH_HAZARDS):
        cases.append(
            _case(
                f"path/{index}",
                [(name, b"X = 1\n"), ("README.md", b"# fine\n")],
            )
        )

    # -- hostile name *and* hostile body together
    for index, name in enumerate(PATH_HAZARDS[:20]):
        cases.append(
            _case(
                f"path-body/{index}",
                [(name, "X = ]]> 1\n```\n" + "\x01\x02")],
            )
        )

    # -- pairs of hazards, where one format's escaping can undo another's
    for index, (left, right) in enumerate(
        zip(FENCE_BREAKERS, XML_HAZARDS, strict=False)
    ):
        cases.append(_case(f"mix/{index}", [("mixed.py", left + right), ("mixed.md", right)]))

    # -- the seeded random fuzzer
    forced = overflow_seed()
    rng = random.Random(seed_for("random-corpus") if forced is None else forced)
    for index in range(60):
        label = f"random/{index}" if forced is None else f"random/{index}/seed-{forced}"
        cases.append(_case(label, _rand_files(rng)))

    return cases


CORPUS = build_corpus()


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> list[tuple[Case, PackResult]]:
    """Every case written to disk and packed for real, once per session."""
    root = tmp_path_factory.mktemp("fuzz")
    tokenizer = Tokenizer(mode="never", calibration=1.0)
    built: list[tuple[Case, PackResult]] = []
    for index, case in enumerate(CORPUS):
        case_dir = root / f"c{index:04d}"
        case_dir.mkdir()
        for name, data in case.files:
            target = case_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        packer = Packer(
            tokenizer,
            Budget(total=case.budget, manifest=case.manifest),
            mode=case.mode,
        )
        built.append((case, packer.pack(discover(case_dir))))
    return built


# --------------------------------------------------------------------------
# A minimal code-fence scanner

_FENCE_RE = re.compile(r" {0,3}(`{3,}|~{3,})(.*)$")


def fence_regions(markdown: str) -> list[tuple[int, int, str]]:
    """``(open_line, close_line, body)`` for each fenced block in ``markdown``.

    A block opened by ``N`` backticks is closed only by a line whose backtick
    run is at least ``N`` long with nothing after it -- which is the rule
    :func:`ctxpack.render.fence` relies on when it escalates past the longest
    run in the payload. A block that never closes is reported with a close
    line past the end of the document, because an unterminated fence is a
    defect, not a region.
    """
    regions: list[tuple[int, int, str]] = []
    opened_at: int | None = None
    open_char = ""
    open_len = 0
    body: list[str] = []
    for index, line in enumerate(markdown.split("\n")):
        match = _FENCE_RE.match(line.rstrip("\r"))
        if opened_at is None:
            if match is None:
                continue
            ticks, rest = match.group(1), match.group(2)
            if ticks[0] == "`" and "`" in rest:
                continue  # an inline code span, not a fence
            opened_at, open_char, open_len = index, ticks[0], len(ticks)
            body = []
            continue
        if match is not None:
            ticks, rest = match.group(1), match.group(2)
            if (
                ticks[0] == open_char
                and len(ticks) >= open_len
                and not rest.strip()
            ):
                regions.append((opened_at, index, "\n".join(body)))
                opened_at = None
                continue
        body.append(line)
    if opened_at is not None:
        regions.append((opened_at, len(markdown.split("\n")), "\n".join(body)))
    return regions


def _index_was_trimmed(result: PackResult) -> bool:
    """True when the index was cut to fit, so it no longer lists every file.

    Detected from the marker ``pack._trim_manifest`` appends rather than by
    recomputing its cap arithmetic.
    """
    return "more files (not listed)" in result.manifest_text


def _kept_prefix_of(on_disk: str, reported: str) -> bool:
    """True when ``reported`` is the file, or a line-aligned prefix of it.

    ``_truncate`` keeps whole lines and appends a marker saying how many it
    dropped, so a reported body must never disagree with the file anywhere
    before the marker.
    """
    kept = 0
    for left, right in zip(on_disk, reported, strict=False):
        if left != right:
            break
        kept += 1
    if on_disk[:kept] != reported[:kept]:
        return False
    return reported[kept:] == "" or reported[kept:].startswith(TRUNCATION_MARKER)


#: What ``pack._truncate`` appends in place of the lines it dropped.
TRUNCATION_MARKER = "\n... [ctxpack truncated: "


def _xml_line_endings(text: str) -> str:
    """Apply the line-ending normalisation every XML parser performs.

    Literal CR and CRLF in element content become LF. That is XML 1.0
    section 2.11, not something a renderer can escape around, so content
    fidelity for a CRLF file is asserted against the normalised form.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _xml_carries(node, text: str) -> bool:
    """True when a parsed ``<file>``/``<block>`` still holds all of ``text``.

    ``_render_xml`` puts each CDATA section on its own line for
    readability, so the parsed text is the payload plus the whitespace around
    it. Anything else -- a dropped character, a reordered one, a leaked
    ``]>`` -- shows up as a payload that is no longer an exact suffix.
    """
    expected = _xml_line_endings(_xml_safe(text))
    got = node.text or ""
    if expected not in got:
        return False
    at = got.index(expected)
    return not (got[:at] + got[at + len(expected) :]).strip()


# --------------------------------------------------------------------------
# Properties 1, 2, 5, 6 -- parse, round-trip, never raise, deterministic
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS)
def test_xml_always_parses(corpus, fmt):
    """Property 1: the XML renderer never emits an unparseable document."""
    for case, result in corpus:
        try:
            text = render(result, "xml")
        except Exception as exc:
            flag("test_xml_always_parses", case.label, case.seed, f"render raised {exc!r}")
        try:
            ElementTree.fromstring(text)
        except ElementTree.ParseError as exc:
            flag(
                "test_xml_always_parses",
                case.label,
                case.seed,
                f"xml did not parse: {exc}\n  document: {clip(text)}",
            )


def test_json_round_trips_what_was_written(corpus):
    """Property 2: json loads, and its content is what is on disk."""
    for case, result in corpus:
        try:
            payload = json.loads(render(result, "json"))
        except ValueError as exc:
            flag("test_json_round_trips_what_was_written", case.label, case.seed, f"{exc!r}")
            continue
        reported = payload["files"]
        if len(reported) != len(result.documents):
            flag(
                "test_json_round_trips_what_was_written",
                case.label,
                case.seed,
                f"{len(reported)} files reported, {len(result.documents)} packed",
            )
        for entry, doc in zip(reported, result.documents, strict=True):
            where = f"{doc.path!r}"
            if entry["path"] != doc.path:
                flag(
                    "test_json_round_trips_what_was_written",
                    case.label,
                    case.seed,
                    f"path {where}: reported {entry['path']!r}",
                )
            if entry["content"] != doc.text:
                flag(
                    "test_json_round_trips_what_was_written",
                    case.label,
                    case.seed,
                    f"content {where}: {clip(entry['content'])} != {clip(doc.text)}",
                )
            if entry["language"] != language_for(doc.path):
                flag(
                    "test_json_round_trips_what_was_written",
                    case.label,
                    case.seed,
                    f"language {where}: {entry['language']!r}",
                )
            if entry["truncated"] != doc.truncated:
                flag(
                    "test_json_round_trips_what_was_written",
                    case.label,
                    case.seed,
                    f"truncated {where}: {entry['truncated']!r}",
                )
            on_disk = read_text(doc.candidate)
            if on_disk is None:
                flag(
                    "test_json_round_trips_what_was_written",
                    case.label,
                    case.seed,
                    f"{where}: packed file no longer reads as text",
                )
            elif not _kept_prefix_of(on_disk, doc.text):
                flag(
                    "test_json_round_trips_what_was_written",
                    case.label,
                    case.seed,
                    f"{where}: reported content is not the file, or the file with "
                    f"a truncation marker. file: {clip(on_disk)}\n"
                    f"  reported: {clip(doc.text)}",
                )
            elif bool(doc.truncated) != (doc.text != on_disk):
                flag(
                    "test_json_round_trips_what_was_written",
                    case.label,
                    case.seed,
                    f"{where}: truncated={doc.truncated} but content "
                    f"{'differs from' if doc.text != on_disk else 'matches'} the file",
                )
        if payload["index"] != [
            {
                "path": entry.path,
                "tokens": entry.tokens,
                "bytes": entry.size,
                "estimated": entry.estimated,
            }
            for entry in result.manifest
        ]:
            flag(
                "test_json_round_trips_what_was_written",
                case.label,
                case.seed,
                "the json index does not match the manifest",
            )


def test_render_never_raises_and_is_deterministic(corpus):
    """Properties 5 and 6: no renderer raises; two renders are identical."""
    formats = (*FORMATS, "pdf", "", "XML", " xml", "\x00", "md", "tree ")
    for case, result in corpus:
        for fmt in formats:
            try:
                first = render(result, fmt)
            except CtxpackError:
                continue  # documented: unknown formats are a controlled error
            except Exception as exc:
                flag(
                    "test_render_never_raises_and_is_deterministic",
                    case.label,
                    case.seed,
                    f"render({fmt!r}) raised {type(exc).__name__}: {exc}",
                )
            if fmt not in FORMATS:
                continue
            if not first.strip():
                flag(
                    "test_render_never_raises_and_is_deterministic",
                    case.label,
                    case.seed,
                    f"{fmt} rendered nothing",
                )
            second = render(result, fmt)
            if first != second:
                flag(
                    "test_render_never_raises_and_is_deterministic",
                    case.label,
                    case.seed,
                    f"{fmt} is not byte-identical across two renders",
                )
            if not first.endswith("\n"):
                flag(
                    "test_render_never_raises_and_is_deterministic",
                    case.label,
                    case.seed,
                    f"{fmt} does not end with a newline",
                )


# --------------------------------------------------------------------------
# Properties 3 and 4 -- nothing swallowed, nothing lost
# --------------------------------------------------------------------------


def test_markdown_fences_still_contain_their_payload(corpus):
    """Property 3: the index and every file body sit inside one fence, whole."""
    for case, result in corpus:
        text = render(result, "markdown")
        regions = fence_regions(text)

        if result.manifest_text:
            held = [
                (opened, closed, body)
                for opened, closed, body in regions
                if body.strip("\n") == result.manifest_text.strip("\n")
            ]
            if len(held) != 1:
                flag(
                    "test_markdown_fences_still_contain_their_payload",
                    case.label,
                    case.seed,
                    "the index is not inside exactly one fence; something in a\n"
                    f"  path closed it early. manifest: {clip(result.manifest_text)}\n"
                    f"  fence bodies seen: {[clip(b) for _, _, b in regions]}",
                )

        for doc in result.documents:
            body = doc.text.rstrip("\n")
            if not any(region.rstrip("\n") == body for _, _, region in regions):
                flag(
                    "test_markdown_fences_still_contain_their_payload",
                    case.label,
                    case.seed,
                    f"no intact fence holds the body of {doc.path!r}.\n"
                    f"  body: {clip(body)}\n"
                    f"  fence bodies seen: {[clip(b) for _, _, b in regions]}",
                )

        for block in result.shared_blocks:
            body = block.text.rstrip("\n")
            if not any(region.rstrip("\n") == body for _, _, region in regions):
                flag(
                    "test_markdown_fences_still_contain_their_payload",
                    case.label,
                    case.seed,
                    f"no intact fence holds shared block {block.id!r}",
                )

        unclosed = [opened for opened, closed, _ in regions if closed <= opened]
        if unclosed:
            flag(
                "test_markdown_fences_still_contain_their_payload",
                case.label,
                case.seed,
                f"fence opened at line {unclosed} never closes",
            )


def test_fence_escalation_is_sound():
    """``fence`` picks a run longer than anything in its payload."""
    rng = random.Random(seed_for("fence-escalation"))
    payloads = list(FENCE_BREAKERS) + [_rand_text(rng) for _ in range(200)]
    payloads.extend("x" * count for count in range(0, 40))
    for payload in payloads:
        for language in ("", "python", "markdown", "js"):
            wrapped = fence(payload, language)
            seed = seed_for("fence-escalation")
            opened, _, rest = wrapped.partition("\n")
            ticks = opened[: len(opened) - len(language)]
            if opened != ticks + language or len(ticks) < 3:
                blow_up(
                    "test_fence_escalation_is_sound",
                    seed,
                    payload,
                    f"opening fence is {opened!r} for language {language!r}",
                )
            body, _, closed = rest.rpartition("\n")
            if closed != ticks:
                blow_up(
                    "test_fence_escalation_is_sound",
                    seed,
                    payload,
                    f"closing fence {closed!r} does not match opening {ticks!r}",
                )
            if body != payload:
                blow_up(
                    "test_fence_escalation_is_sound",
                    seed,
                    payload,
                    "the payload came back out of the fence changed",
                )
            longest = max(
                (len(run) for run in re.findall(r"`+", payload)),
                default=0,
            )
            if len(ticks) <= longest:
                blow_up(
                    "test_fence_escalation_is_sound",
                    seed,
                    payload,
                    f"a {len(ticks)}-tick fence cannot contain a {longest}-tick run",
                )


def test_every_path_that_went_in_comes_back_out(corpus):
    """Property 4: a packed file is named in every format that lists files.

    Checked against the *parsed* structure rather than the rendered text:
    a path containing a quote is escaped in the xml and written escaped in
    the json, so a substring search would flag correct output.
    """
    for case, result in corpus:
        wanted = [doc.path for doc in result.documents]
        markdown = render(result, "markdown")
        seen = {
            # _render_markdown writes each heading as ``` ### N. `path` ```,
            # so that exact span is what "names the file" means here. Parsing
            # the heading with a regex instead would be testing my parser
            # rather than the renderer, and a path containing a newline or a
            # backtick breaks it.
            "markdown": [
                # Headings render the path through `_label`, which escapes
                # newlines and neutralises backticks so a filename cannot end
                # the heading or its code span. So the rendered form, not the
                # raw path, is what "names the file" means for markdown.
                path for path in wanted if f"`{_label(path)}`" in markdown
            ],
            "json": [entry["path"] for entry in json.loads(render(result, "json"))["index"]],
            "xml": _paths_in_xml(render(result, "xml")),
            "tree": _paths_in_tree(render(result, "tree"), result),
        }
        for fmt, found in seen.items():
            for path in wanted:
                # _xml_safe strips what XML cannot represent, so a path
                # carrying such a character legitimately differs there.
                # See the notes at the bottom of this module.
                expected = _xml_safe(path) if fmt == "xml" else path
                if expected not in found:
                    flag(
                        "test_every_path_that_went_in_comes_back_out",
                        case.label,
                        case.seed,
                        f"{fmt} output never mentions packed file {path!r}; "
                        f"it mentions {sorted(found)[:6]}",
                    )


def _paths_in_xml(text: str) -> list[str]:
    """Paths named in the xml bundle, from both its index and its files."""
    root = ElementTree.fromstring(text)
    return [node.get("path") for node in root.iter("entry") if node.get("path")] + [
        node.get("path") for node in root.iter("file") if node.get("path")
    ]


def _paths_in_tree(text: str, result: PackResult) -> list[str]:
    """Paths the tree bundle lists.

    The tree format *is* the index, so with ``--manifest none`` it has
    nothing to print, and a trimmed index has stopped listing files by
    design. In both cases the paths cannot be recovered from it, which is a
    deliberate consequence of those options rather than a rendering fault --
    see the notes at the bottom of this module.
    """
    if not result.manifest_text or _index_was_trimmed(result):
        # Neither case can be checked: with --manifest none there is no index,
        # and a trimmed index has stopped listing files by design. Return the
        # packed paths so the assertion passes vacuously rather than failing
        # on an intentional omission. See the notes at the bottom.
        return [doc.path for doc in result.documents]
    # The index rows are "<tokens>  <bytes>  <path>", but a path may itself
    # contain the separator or a newline, so search the whole text for each
    # path rather than splitting it into rows.
    return [entry.path for entry in result.manifest if entry.path in text]


def test_cdata_round_trips_its_own_edge_cases():
    """``_cdata`` splits ``]]>`` without losing or gaining a character."""
    seed = seed_for("cdata-edges")
    payloads = list(CDATA_EDGES)
    rng = random.Random(seed)
    alphabet = "]]]><!CDATA[abc&\"'\n "
    payloads.extend(
        "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 24)))
        for _ in range(200)
    )
    for payload in payloads:
        expected = _xml_line_endings(_xml_safe(payload))
        try:
            node = ElementTree.fromstring(f"<r>{_cdata(payload)}</r>")
        except ElementTree.ParseError as exc:
            blow_up("test_cdata_round_trips_its_own_edge_cases", seed, payload, f"{exc!r}")
            continue
        got = node.text or ""
        if got != expected:
            blow_up(
                "test_cdata_round_trips_its_own_edge_cases",
                seed,
                payload,
                f"parsed back as {clip(got)}, expected {clip(expected)}",
            )


def test_xml_content_is_the_document_modulo_forbidden_characters(corpus):
    """XML loses only the characters XML 1.0 cannot carry, and nothing else."""
    for case, result in corpus:
        try:
            root = ElementTree.fromstring(render(result, "xml"))
        except ElementTree.ParseError as exc:
            flag("test_xml_content_is_the_document", case.label, case.seed, f"{exc!r}")
            continue
        nodes = list(root.iter("file"))
        if len(nodes) != len(result.documents):
            flag(
                "test_xml_content_is_the_document_modulo_forbidden_characters",
                case.label,
                case.seed,
                f"{len(nodes)} <file> elements for {len(result.documents)} documents",
            )
        for node, doc in zip(nodes, result.documents, strict=True):
            if not _xml_carries(node, doc.text):
                flag(
                    "test_xml_content_is_the_document_modulo_forbidden_characters",
                    case.label,
                    case.seed,
                    f"{doc.path!r}: <file> parses as {clip(node.text or '')}, but the "
                    f"document is {clip(_xml_line_endings(_xml_safe(doc.text)))}",
                )
        blocks = list(root.iter("block"))
        for node, block in zip(blocks, result.shared_blocks, strict=True):
            if not _xml_carries(node, block.text):
                flag(
                    "test_xml_content_is_the_document_modulo_forbidden_characters",
                    case.label,
                    case.seed,
                    f"shared block {block.id!r}: {clip(node.text or '')}",
                )


# --------------------------------------------------------------------------
# Hand-built shapes the CLI produces but a plain pack does not
# --------------------------------------------------------------------------


def synthetic_result(
    documents: tuple[tuple[str, str], ...],
    *,
    blocks: tuple[SharedBlock, ...] = (),
    notes: tuple[str, ...] = (),
    root: str = "/synthetic/root",
) -> PackResult:
    """A PackResult built directly, bypassing ``discover``.

    ``walk.read_text`` decodes with ``errors="replace"``, so a lone surrogate
    can never reach the renderers from a real file: invalid bytes become
    U+FFFD. Testing surrogates therefore means building the documents here
    instead. See the notes at the bottom of this module.
    """
    built = []
    for name, text in documents:
        candidate = Candidate(
            path=name,
            abs_path=Path(root) / name,
            size=len(text.encode("utf-8", "surrogatepass")),
            is_binary=False,
        )
        count = TOKENIZER.count(text)
        built.append(
            Document(
                candidate=candidate,
                scored=Scored(candidate=candidate, score=1.0, signals=()),
                text=text,
                tokens=count.tokens,
                lines=text.count("\n") + 1,
            )
        )
    manifest = [
        ManifestEntry(doc.path, doc.tokens, doc.candidate.size) for doc in built
    ]
    manifest_text = "\n".join(
        f"  {entry.tokens:>9,}  {entry.size:>10,}  {entry.path}" for entry in manifest
    )
    return PackResult(
        root=Path(root),
        documents=built,
        manifest=manifest,
        manifest_text=manifest_text,
        manifest_tokens=TOKENIZER.count(manifest_text).tokens,
        budget=20_000,
        header_reserve=300,
        mode="balanced",
        encoding="o200k_base",
        method="estimate",
        calibration=1.0,
        discovered=len(built),
        scanned=len(built),
        unscanned=0,
        notes=list(notes),
        shared_blocks=list(blocks),
    )


SURROGATE_PAYLOADS: tuple[str, ...] = (
    "\ud800",
    "\udfff",
    "\ud83d",
    "\udcff",
    "a\ud800b",
    "\ud800\ud800",
    "lone\n\ud800\nsurrogate",
    "```\n\ud800\n```",
    "]]>\ud800",
    "\ud800" * 50,
)


def test_lone_surrogates_survive_every_renderer():
    """The reason ``_xml_safe`` exists: a surrogate cannot be encoded at all."""
    seed = seed_for("surrogates")
    for payload in SURROGATE_PAYLOADS:
        result = synthetic_result((("lone.py", f"X = 1\n{payload}\n"),))
        for fmt in FORMATS:
            try:
                text = render(result, fmt)
            except Exception as exc:
                blow_up(
                    "test_lone_surrogates_survive_every_renderer",
                    seed,
                    payload,
                    f"{fmt} raised {type(exc).__name__}: {exc}",
                )
            if fmt == "xml":
                try:
                    node = ElementTree.fromstring(text)
                except ElementTree.ParseError as exc:
                    blow_up(
                        "test_lone_surrogates_survive_every_renderer",
                        seed,
                        payload,
                        f"xml did not parse: {exc}",
                    )
                    continue
                kept = [child.text or "" for child in node.iter("file")]
                if any("\ud800" <= ch <= "\udfff" for ch in "".join(kept)):
                    blow_up(
                        "test_lone_surrogates_survive_every_renderer",
                        seed,
                        payload,
                        "a lone surrogate reached the parsed xml",
                    )
                if "X = 1" not in "".join(kept):
                    blow_up(
                        "test_lone_surrogates_survive_every_renderer",
                        seed,
                        payload,
                        "the rest of the file was lost with the surrogate",
                    )
            if fmt == "json":
                try:
                    loaded = json.loads(text)
                except ValueError as exc:
                    blow_up(
                        "test_lone_surrogates_survive_every_renderer",
                        seed,
                        payload,
                        f"json did not load: {exc!r}",
                    )
                    continue
                reported = loaded["files"][0]["content"]
                if reported != result.documents[0].text:
                    blow_up(
                        "test_lone_surrogates_survive_every_renderer",
                        seed,
                        payload,
                        f"json changed the content: {clip(reported)}",
                    )


def test_hostile_metadata_and_shared_blocks():
    """Ids, notes and shared-block bodies are escaped like everything else."""
    seed = seed_for("metadata")
    hostile_id = 'block-1"\'<&>`x` \ud800 ]'
    result = synthetic_result(
        (("a.py", "X = 1\n"), ("b.md", "]]>\n```\n")),
        blocks=(
            SharedBlock(id=hostile_id, lines=1, occurrences=2, text="]]>\n```\n\x01"),
            SharedBlock(id="<block/>", lines=0, occurrences=1, text=""),
        ),
        notes=(
            "a note with ``` and ]]> and <x> and \x01",
            "a note\nwith a newline",
            "--",
        ),
        root='/root "quoted" & <angled>',
    )
    for fmt in FORMATS:
        try:
            render(result, fmt)
        except Exception as exc:  # noqa: PERF203 - the loop is four formats long
            blow_up(
                "test_hostile_metadata_and_shared_blocks",
                seed,
                hostile_id,
                f"{fmt}: {exc!r}",
            )
    try:
        ElementTree.fromstring(render(result, "xml"))
    except ElementTree.ParseError as exc:
        blow_up("test_hostile_metadata_and_shared_blocks", seed, hostile_id, f"xml: {exc!r}")
    try:
        json.loads(render(result, "json"))
    except ValueError as exc:
        blow_up("test_hostile_metadata_and_shared_blocks", seed, hostile_id, f"json: {exc!r}")
    # The shared blocks must be individually identifiable, so the escaping
    # has to be reversible rather than merely well-formed.
    root = ElementTree.fromstring(render(result, "xml"))
    ids = [node.get("id") for node in root.iter("block")]
    if ids != [_xml_safe(hostile_id), "<block/>"]:
        blow_up("test_hostile_metadata_and_shared_blocks", seed, hostile_id, f"ids: {ids!r}")


# --------------------------------------------------------------------------
# Property 7 -- gitignore patterns
# --------------------------------------------------------------------------

#: Patterns whose *shape* is the point: adjacent unbounded quantifiers, long
#: ``**`` chains, an unbalanced class, a trailing backslash, unicode.
PATTERN_SEEDS: tuple[str, ...] = (
    "*",
    "**",
    "***",
    "?",
    "??",
    "[",
    "[!",
    "[]",
    "[]]",
    "[[",
    "[a-z]",
    "[!a-z]",
    "[^a-z]",
    "[a-]",
    "[-a]",
    "[\\]]",
    "[z-a]",
    "[\\d--z]",
    "\\",
    "a\\",
    "\\a",
    "!",
    "!a",
    "/",
    "//",
    "a/",
    "/a",
    "/*",
    "*/",
    "**/",
    "/**",
    "**a",
    "a**",
    "**/**",
    "**/**/**",
    "**/**/**/**/**/x",
    "a/**/b",
    "a/**/b/**/c/**/d",
    "(**)",
    "(*)",
    "*a*a*a*",
    "*a*a*a*a*a*a*a*a*b",
    "*?",
    "*??",
    "*?*?*?*?*?*?",
    "*a?b*c",
    ".",
    "..",
    "...",
    "*.py",
    "*.",
    "*..",
    "#comment",
    "",
    "   ",
    "  # indented comment",
    "!",
    "!!a",
    "\u00e9*.py",
    "\u65e5\u672c\u8a9e/**",
    "\U0001f389.py",
    "a" * 200,
    "*" * 40,
    "?" * 40,
    "[" * 40,
    "\\" * 40,
    "**" * 30,
)

#: Patterns that make the matcher do real work, and so must not be allowed
#: to take it out on the context window.
BACKTRACKING_PATTERNS: tuple[str, ...] = (
    "*" * 10 + "b",
    "*" * 12 + "b",
    "*" * 14 + "b",
    "*" * 16,
    "*" * 16 + "b",
    "*" * 24,
    "*a" * 8 + "*ab",
    "*a" * 6 + "*ab",
    "*" * 12 + "?",
    "*?" * 12 + "a",
    "*" * 8 + "*" * 8 + "b",
    "a*" * 8 + "b",
    "*a*a*a*a*a*a*a*a*a*a*a*b",
    "/" + "*" * 30,
    "*" * 30 + "/b",
)

#: A path that is long, has no "/" (so no glob can anchor early) and cannot
#: match any of the patterns above, which is the case that forces the
#: matcher to explore every way it could have split the string.
HOSTILE_SUBJECT = "a" * 200 + "\nb"

#: Generous on purpose: every pattern here costs microseconds once it is
#: linear, and hours if it is not, so the threshold is nowhere near a
#: source of flakes.
PATTERN_TIME_LIMIT = 1.0


def _compile_or_flag(pattern: str, seed: int, test: str):
    """:func:`compile_pattern`, with any exception turned into a seeded failure."""
    try:
        return compile_pattern(pattern)
    except Exception as exc:
        blow_up(test, seed, pattern, f"compile_pattern raised {type(exc).__name__}: {exc}")


def _match_or_flag(rule, subject: str, pattern: str, seed: int, test: str) -> None:
    """One ``rule.matches(subject)``, same treatment."""
    try:
        rule.matches(subject)
    except Exception as exc:
        blow_up(test, seed, pattern, f"matches({clip(subject)}) raised {exc!r}")

IGNORE_SUBJECTS: tuple[str, ...] = (
    "",
    "a",
    "a.py",
    "src/ctxpack/render.py",
    "a" * 200,
    "a" * 200 + "\nb",
    "\u65e5\u672c\u8a9e/\u30d5\u30a1\u30a4\u30eb.py",
    "with space/file.py",
    "quote'file\".py",
    "brack[et]s.py",
    "back`tick.py",
    "new\nline.py",
    "`````",
    "]]>",
    "--",
    "/absolute/path.py",
    "../escape.py",
    "./relative.py",
    "x" * 400 + "/" + "y" * 400,
)


def test_compile_pattern_never_raises():
    """Property 7a: no gitignore line, however malformed, crashes the compiler."""
    seed = seed_for("compile-pattern")
    patterns = list(PATTERN_SEEDS)
    rng = random.Random(seed)
    alphabet = "*?[]!\\/.^$-abz\u00e9 \t"
    patterns.extend(
        "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 40)))
        for _ in range(600)
    )

    for pattern in patterns:
        rule = _compile_or_flag(pattern, seed, "test_compile_pattern_never_raises")
        if rule is None:
            continue
        for subject in IGNORE_SUBJECTS:
            _match_or_flag(rule, subject, pattern, seed, "test_compile_pattern_never_raises")


def test_pattern_matching_is_bounded():
    """Property 7b: a hostile pattern cannot make matching take exponential time.

    The patterns here are adjacent unbounded quantifiers -- ``*`` translated
    to ``[^/]*``, or ``**`` to ``.*``, with nothing between them. A subject
    that cannot match forces the engine to try every way of splitting it
    between those quantifiers, which is what turns a one-line ``.gitignore``
    into an unbounded hang.
    """
    seed = seed_for("backtracking")
    for pattern in BACKTRACKING_PATTERNS:
        rule = compile_pattern(pattern)
        if rule is None:
            blow_up("test_pattern_matching_is_bounded", seed, pattern, "compiled to None")
        elapsed = bounded(rule.matches, HOSTILE_SUBJECT, limit=PATTERN_TIME_LIMIT)
        if elapsed is None:
            blow_up(
                "test_pattern_matching_is_bounded",
                seed,
                pattern,
                f"matching a {len(HOSTILE_SUBJECT)}-char path took longer than "
                f"{PATTERN_TIME_LIMIT}s",
            )


def test_ignore_stack_is_bounded_and_never_raises():
    """Property 7c: the same, through the nested-rule stack discovery uses."""
    seed = seed_for("ignore-stack")
    here = "test_ignore_stack_is_bounded_and_never_raises"
    patterns = list(BACKTRACKING_PATTERNS)
    rules = [
        rule
        for pattern in (*patterns, *PATTERN_SEEDS)
        if (rule := _compile_or_flag(pattern, seed, here)) is not None
    ]
    if not rules:
        blow_up(here, seed, "PATTERN_SEEDS", "no rules compiled at all")

    for pattern in patterns:
        rule = _compile_or_flag(pattern, seed, here)
        if rule is None:
            continue
        stack = IgnoreStack([rule])
        stack.push("sub/dir", rules[:20])
        stack.push("sub/dir/deeper", rules)
        for subject in IGNORE_SUBJECTS:
            try:
                elapsed = bounded(stack.ignored, subject, False, limit=PATTERN_TIME_LIMIT)
            except Exception as exc:
                blow_up(
                    "test_ignore_stack_is_bounded_and_never_raises",
                    seed,
                    pattern,
                    f"ignored({clip(subject)}) raised {type(exc).__name__}: {exc}",
                )
            if elapsed is None:
                blow_up(
                    "test_ignore_stack_is_bounded_and_never_raises",
                    seed,
                    pattern,
                    f"{len(rules)} rules against {clip(subject)} took longer than "
                    f"{PATTERN_TIME_LIMIT}s",
                )
        stack.pop()
        stack.pop()
        # Back to the base layer, which holds only the rule it was constructed
        # with -- not the rules that were pushed on top of it.
        if stack.count() != 1:
            blow_up(here, seed, pattern, "pop() ate the base layer")
        stack.pop()


def test_a_star_run_in_gitignore_does_not_stall_discovery(tmp_path):
    """The same defect, reached the way a user would reach it.

    ``ctxpack .`` compiles every ``.gitignore`` it finds and matches it
    against every path in the repository, so a pattern that backtracks
    exponentially does not need an unusual file -- just an ordinary deep path
    and a line in an ignore file.
    """
    seed = seed_for("discover-stall")
    (tmp_path / ".gitignore").write_text("*" * 16 + "b\n", encoding="utf-8")
    deep = tmp_path
    for index in range(6):
        deep = deep / f"directory_name_{index:02d}"
    deep.mkdir(parents=True)
    (deep / "target.py").write_text("X = 1\n", encoding="utf-8")

    elapsed = bounded(discover, tmp_path, limit=2.0)
    if elapsed is None:
        blow_up(
            "test_a_star_run_in_gitignore_does_not_stall_discovery",
            seed,
            "*" * 16 + "b",
            "discover() did not finish within 2s on a "
            f"{len(str(deep / 'target.py'))}-character path",
        )
    found = [candidate.path for candidate in discover(tmp_path).files]
    if "target.py" not in {Path(path).name for path in found}:
        blow_up(
            "test_a_star_run_in_gitignore_does_not_stall_discovery",
            seed,
            "*" * 16 + "b",
            f"the file went missing: {found}",
        )


# --------------------------------------------------------------------------
# Generated on purpose, but deliberately *not* asserted on
#
# Listed so the gaps are visible rather than forgotten:
#
# 0. The index fence under a path containing a newline and backticks. The
#    assertion is *in* the suite (test_markdown_fences_still_contain_their_
#    payload) and it currently fails: see the report. Left asserted rather
#    than skipped, because a passing test would be a lie.
#
# 1. Lone surrogates reaching the renderers from a real file. Generated and
#    impossible to reach: ``walk.read_text`` decodes with
#    ``errors="replace"``, so invalid bytes become U+FFFD and no surrogate
#    survives. Asserted instead on hand-built Documents, in
#    ``test_lone_surrogates_survive_every_renderer``.
# 2. NUL bytes inside a packed document. Same reason: ``walk`` sniffs the
#    first 8 KiB for NUL and drops the file as binary before any renderer
#    sees it, so a document cannot hold one. Asserted on disk, in
#    ``BYTE_HAZARDS`` ("nul-byte"), as "the bundle still renders".
# 3. XML byte round-tripping of CRLF. XML 1.0 section 2.11 requires parsers
#    to normalise literal CR and CRLF in content to LF, so an XML consumer
#    cannot get CRLF back no matter what the renderer emits. Asserted
#    against the normalised form instead, via ``_xml_line_endings``.
# 4. Paths containing characters XML cannot represent (a control byte in a
#    file name). ``_xml_safe`` strips them, so the xml index legitimately
#    spells such a path differently from markdown and json. Asserted against
#    ``_xml_safe(path)`` for xml and against the raw path elsewhere.
# 5. The U+2028 / U+2029 line separators changing ``Document.lines``.
#    ``str.splitlines`` counts them, ``text.count("\\n")`` does not, so the
#    reported line count for such a file is approximate. That is a counting
#    question in pack.py, not a rendering one, and no renderer consumes
#    ``lines``, so it is left alone rather than asserted into existence.
# 6. Markdown inline code spans. A name containing a backtick can leave the
#    inline span in a ``### 1. `name``` heading unterminated, and an
#    unterminated span swallows the following lines as literal text. No
#    assertion: telling "unbalanced span" from "balanced span" needs a real
#    CommonMark implementation, and the file body and its fence stay intact
#    either way. Reported instead of tested.
# --------------------------------------------------------------------------
