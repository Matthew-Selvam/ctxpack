"""Structure instead of bodies.

ctxpack normally spends its whole budget on full file contents, which is the
right default and the wrong answer to a large class of questions. "What does
this project expose?", "where is auth handled?", "what are the entrypoints?",
"which module defines ``Session``?" -- none of those need a single function
body. They need signatures.

An outline is exactly that: every declaration, its signature, and its first
docstring line, with the bodies left out. Measured against ctxpack's own source,
it runs 4x to 14x cheaper with a median around 6x -- so a budget that fit three
full modules fits twenty outlines. That is the trade: breadth instead of depth,
and you convert breadth back into depth by spending real tokens on the files
that turn out to matter.

The ratio is not a constant, and the variance is the interesting part. It is a
function of how much body a file has: a 700-line module with fifty small
functions compresses well, a five-line module with one class does not compress
at all because there is nothing to remove. Do not expect 6x from a file that was
never going to give it.

Decisions that look wrong but are not:

* **Line numbers survive, and so does source order.** An outline is only useful
  if you can get from it back to the source, and it is only diff-stable if
  adding a blank line at the top of a file does not reshuffle everything below.
  Entries are therefore emitted in declaration order with real 1-based line
  numbers, never sorted, grouped by kind, or deduplicated.

* **Failures degrade, they never raise.** :func:`ast.parse` rejects files that
  are truncated mid-edit, generated with a syntax error, or written in a dialect
  it does not know. An outline is an optimisation over reading the whole file, so
  it must never be the thing that stops a pack: bad AST falls back to a regex
  scan, an unrecognised extension falls back to a placeholder, and an empty file
  falls back to a placeholder too. An empty list is indistinguishable from
  "nothing found", and "nothing found" silently drops a file from the bundle.

* **Python is parsed, JavaScript is guessed.** The stdlib ships a real Python
  parser and nothing at all for JS/TS, and ctxpack has no dependencies.
  :func:`js_outline` is line-oriented regex matching: correct on ordinary code,
  confused by braces inside string literals. It says so in its docstring instead
  of pretending otherwise. Only two languages, deliberately: a wrong outline is
  worse than none, because it looks authoritative.

* **Docstrings are kept; assigned values mostly are not.** The first line of a
  docstring answers "what is this for", which is the question agents actually
  ask, so it is worth its tokens. An assigned *value* is another matter -- a
  module-level dict literal can be a thousand lines -- so values are included
  only when they occupy one source line and render short. That single rule is
  what keeps a locale file or a constants table from turning its outline into a
  second copy of itself.

  The exception is ``__all__``, a list of bare names. Counting it would be
  cheaper and useless, so it is printed.

* **No comment alignment.** Lines could be padded so trailing comments line up
  in a column. That costs tokens on every line and makes a one-character rename
  rewrite the whole file's whitespace, which is the opposite of diff-stable.
  Two spaces before the ``#`` and no more.
"""

from __future__ import annotations

import ast
import copy
import re
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, replace

from .errors import CtxpackError
from .tokens import estimate_tokens

__all__ = [
    "JS_EXTS",
    "KINDS",
    "OUTLINE_EXTS",
    "PYTHON_EXTS",
    "Outline",
    "estimate_ratio",
    "js_outline",
    "outline_for",
    "outline_text",
    "outline_tokens",
    "python_outline",
    "render_outline",
    "unstructured",
]

#: Extensions routed to :func:`python_outline`.
PYTHON_EXTS = frozenset({".py", ".pyi"})

#: Extensions routed to :func:`js_outline`. ``.vue``/``.svelte`` are excluded on
#: purpose: their script blocks sit inside templates, and regexing a template
#: produces confident nonsense.
JS_EXTS = frozenset({".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"})

#: Every extension with a dedicated extractor. A missing entry is deliberate
#: rather than an oversight: Go, Rust and the rest would each need their own
#: hand-written scanner, and a wrong outline is worse than none because it looks
#: authoritative.
OUTLINE_EXTS = PYTHON_EXTS | JS_EXTS

#: Every value ``Outline.kind`` can take, for callers that validate their own.
KINDS = (
    "class",
    "function",
    "method",
    "const",
    "interface",
    "type",
    "enum",
    "import",
    "section",
    "other",
)

#: Leading keyword rendered before each kind. Methods get none: inside a class
#: the bare name is already unambiguous, and the keyword would be a wasted token
#: on the most numerous lines in the outline.
_KIND_PREFIX = {
    "class": "class ",
    "function": "function ",
    "method": "",
    "const": "const ",
    "interface": "interface ",
    "type": "type ",
    "enum": "enum ",
    "import": "",
    "section": "",
    "other": "",
}

#: Flags rendered *before* the name, because they change how the declaration is
#: called: ``async find_by_email(...)``, ``static create(...)``, ``get size()``.
_LEADING_FLAGS = ("async", "static", "getter", "setter")

#: Human-readable flag text. Decorators and flags share this table because they
#: share the trailing comment: one marker list, one rule, no guessing which of
#: the two a reader is looking at.
_FLAG_LABELS = {
    "abstract": "abstract",
    "async": "async",
    "dataclass": "dataclass",
    "attrs": "attrs",
    "namedtuple": "namedtuple",
    "typeddict": "typeddict",
    "protocol": "protocol",
    "enum": "enum",
    "overload": "overload",
    "property": "property",
    "staticmethod": "staticmethod",
    "classmethod": "classmethod",
    "final": "final",
    "export": "export",
    "default": "default export",
    "static": "static",
    "getter": "get",
    "setter": "set",
    "stub": "stub",
    "abstractclass": "abstract class",
    "declare": "declare",
}

#: Decorator names that describe a class *shape* rather than a behaviour. Kept
#: as flags because ``@dataclass`` tells a reader more about a class than its
#: base list does; every other decorator is rendered the same way as these, as a
#: bare name in the marker list.
_SHAPE_DECORATORS = {
    "dataclass": "dataclass",
    "attrs": "attrs",
    "attr": "attrs",
    "define": "attrs",
    "namedtuple": "namedtuple",
    "typeddict": "typeddict",
    "runtime_checkable": "protocol",
    "final": "final",
}

#: Base class names meaning "this class is an enum", matched on the last
#: component only, so ``class Status(str, enum.Enum)`` still counts.
_ENUM_BASES = frozenset(
    {
        "Enum",
        "EnumMeta",
        "EnumType",
        "Flag",
        "IntEnum",
        "IntFlag",
        "ReprEnum",
        "StrEnum",
        "StrFlag",
    }
)

#: Substrings identifying a guard worth descending into. ``TYPE_CHECKING`` holds
#: the forward-referenced protocols and dataclasses that answer "what is this
#: module's contract"; version guards hide real definitions often enough to be
#: worth the extra lines.
_SECTION_TESTS = (
    "TYPE_CHECKING",
    "TYPEGUARD",
    "MYPY",
    "PYRIGHT",
    "VERSION",
    "PY3",
    "TYPING",
)

#: Longest assigned value kept inline. The guard against an outline quietly
#: becoming the file it is summarising.
_MAX_VALUE = 48

#: Longest signature or type-alias body kept on one line.
_MAX_DETAIL = 96

#: Longest trailing comment, so one enormous docstring cannot dominate a render.
_MAX_COMMENT = 72

#: Longest ``__all__``-style name list rendered rather than counted.
_MAX_NAMES = 120

#: Placeholder text for a recognised-but-elided value.
_ELIDED = "..."

#: Extensions that are declarations by definition, so a body is always noise.
_STUB_EXTS = (".pyi", ".d.ts")

#: ``ast.Match`` exists from Python 3.10, which is this package's floor, but the
#: name is resolved at runtime rather than imported at module scope so the module
#: still loads if that ever stops being true.
_MATCH_STMT = getattr(ast, "Match", None)

#: Maximum brace-nesting depth followed in JS/TS before giving up.
#:
#: Rendering recurses per nesting level, and 1200 nested `class C {` blocks --
#: trivial to write, trivial to hit with minified or generated input -- blew the
#: interpreter's recursion limit with a bare ``RecursionError``. This module's
#: contract is that it never raises, so past the limit the file is reported as
#: unstructured instead, which is honest: no useful outline of 1200-deep nesting
#: exists anyway.
MAX_NESTING_DEPTH = 200


def _source_lines(text: str) -> list[str]:
    """Split into physical lines for line numbering, on newlines only.

    ``str.splitlines()`` also breaks on form feed, ``\\x0b``, ``\\x1c``, NEL and
    U+2028 -- all of which are ordinary *characters* a source file can contain.
    Each one shifts every subsequent line number by one, so a single form feed
    silently misreports the file's whole layout, which for a navigation tool is
    worse than not reporting it. ``boiler._split_lines`` already refuses this
    hazard; the two modules now agree.

    The trailing empty element from a final newline is dropped, since it is not
    a line and would make a file one line longer than it is.
    """
    parts = (text or "").split("\n")
    if parts and parts[-1] == "":
        parts.pop()
    return parts


@dataclass(frozen=True)
class Outline:
    """One declaration, without its body.

    ``children`` holds nested declarations: methods under their class, a nested
    function under its enclosing function, statements under a ``section``
    guard. ``line`` is 1-based and refers to the original source, never to the
    render. ``detail`` is whatever completes the name into a readable signature
    -- ``(a, b: int) -> R`` for a function, ``(Base)`` for a class,
    ``= {...}`` for a constant.
    """

    kind: str
    name: str
    line: int = 0
    detail: str = ""
    children: tuple[Outline, ...] = ()
    doc: str = ""
    decorators: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in _KIND_PREFIX:
            raise CtxpackError(
                f"unknown outline kind {self.kind!r}; expected one of "
                + ", ".join(KINDS)
            )
        # Coerce, do not merely annotate. Building an Outline from a list of
        # children is the natural thing to do and every internal caller did it,
        # so without this the frozen dataclass would hold a mutable list and
        # `frozen=True` would be a promise the type did not keep. Coercing here
        # means `children` is a tuple whatever the caller passed.
        if not isinstance(self.children, tuple):
            object.__setattr__(self, "children", tuple(self.children))

    @property
    def signature(self) -> str:
        """``name`` plus ``detail``: the part that reads like source."""
        return f"{self.name}{self.detail}" if self.detail else self.name

    @property
    def markers(self) -> str:
        """Decorators and flags as one comma-separated list, in source order.

        Bare, never ``@``-prefixed: the ``@`` is a character of pure noise
        repeated on every decorated entry, and a reader can tell a marker from a
        value without it.
        """
        seen: list[str] = []
        for item in (*self.decorators, *self.flags):
            label = _FLAG_LABELS.get(item, item)
            if label and label not in seen:
                seen.append(label)
        return ", ".join(seen)

    def walk(self) -> Iterator[Outline]:
        """Yield this node then its descendants, depth first, in source order."""
        yield self
        for child in self.children:
            yield from child.walk()

    def find(self, name: str) -> Outline | None:
        """First node named ``name``, searching depth first in source order."""
        for node in self.walk():
            if node.name == name:
                return node
        return None


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _prefix(item: Outline) -> str:
    lead = "".join(f"{_FLAG_LABELS[f]} " for f in item.flags if f in _LEADING_FLAGS)
    return lead + _KIND_PREFIX[item.kind]


def _render_entry(item: Outline, *, indent: str, depth: int) -> str:
    line = f"{indent * depth}{_prefix(item)}{item.signature}"
    tail = item.markers
    if item.doc:
        tail = f"{tail} - {item.doc}" if tail else item.doc
    if tail:
        tail = _cap(tail, _MAX_COMMENT)
        line = f"{line}  # {tail}"
    return line.rstrip()


def render_outline(outlines: list[Outline], *, indent: str = "  ") -> str:
    """Flatten outlines to one line per declaration, two-space nesting.

    Stable by construction: every line derives from a single declaration, entries
    stay in source order with their original line numbers, and nothing is sorted
    or aligned. Rendering the same file twice is byte-identical, which is what
    makes an outline safe to commit or diff.

    The trailing comment carries markers (decorators, ``dataclass``, ``async``)
    and then the first line of the docstring, in that order, because a reader
    wants to know *what shape* a thing is before they want to know what it says.
    """
    if not indent or indent.strip():
        # Rejecting a non-whitespace indent matters more than it looks: the
        # indent is the only nesting signal in the render, so a caller passing
        # `-->` would produce something that reads as structure it does not have.
        raise CtxpackError(
            "indent must be one or more whitespace characters, got "
            f"{indent!r}"
        )
    lines: list[str] = []
    for item in outlines:
        lines.extend(_render_tree(item, indent=indent))
    return "\n".join(lines)


def _render_tree(item: Outline, *, indent: str, depth: int = 0) -> list[str]:
    lines = [_render_entry(item, indent=indent, depth=depth)]
    for child in item.children:
        lines.extend(_render_tree(child, indent=indent, depth=depth + 1))
    return lines


def _cap(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - len(_ELIDED)].rstrip() + _ELIDED


# --------------------------------------------------------------------------
# Placeholder
# --------------------------------------------------------------------------


def unstructured(text: str, *, reason: str = "no outline extractor") -> list[Outline]:
    """One entry standing in for a file that could not be structured.

    Never returns an empty list, not even for an empty file. ``[]`` means "no
    declarations", which is indistinguishable from "empty file" and from "the
    parser gave up" -- and all three would let a file disappear from a bundle
    without a trace. A visible placeholder costs a few tokens and keeps the file
    present and explained.
    """
    count = text.count("\n") + 1 if text else 0
    size = f"{count} lines" if count else "empty"
    return [Outline(kind="other", name="(unstructured)", line=1, detail=f" {reason}; {size}")]


# --------------------------------------------------------------------------
# Python
# --------------------------------------------------------------------------


def _first_line(text: str, limit: int = _MAX_COMMENT) -> str:
    """First non-empty line of a docstring, collapsed onto one line and capped."""
    for raw in _source_lines(text):
        line = " ".join(raw.split())
        if line:
            return _cap(line, limit)
    return ""


def _docstring(node: ast.AST) -> str:
    try:
        return _first_line(ast.get_docstring(node) or "")
    except Exception:  # pragma: no cover - get_docstring is total in practice
        return ""


def _unparse(node: ast.AST | None) -> str:
    """One-line source for an AST node, or ``""`` if it cannot be rendered."""
    if node is None:
        return ""
    try:
        return " ".join(ast.unparse(node).split())
    except Exception:
        # A node the unparser cannot render must not cost us the whole entry.
        # An empty detail degrades to a name-only line, which is still useful.
        return ""


def _balance(text: str, open_ch: str, close_ch: str) -> int:
    """Index of the close bracket matching the first ``open_ch``, or ``-1``."""
    depth = 0
    for i, ch in enumerate(text):
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """``(a, b: int = 1) -> R`` on one line, however the source formatted it.

    ``ast.unparse`` on a body-less copy is the right tool: it re-renders
    annotations, defaults and return types from the AST instead of slicing text,
    so a signature spread over nine lines with a comment between the parameters
    comes back correctly joined. The manual reconstruction below is the fallback
    for the rare node it refuses, and it reads the AST fields directly.
    """
    # A shallow copy rather than a hand-built FunctionDef: the constructor grew
    # a ``type_params`` field in 3.12 and the keyword set is version-dependent,
    # while ``copy.copy`` works the same on every supported version and cannot
    # drift out of sync with the interpreter.
    stub = copy.copy(node)
    stub.body = [ast.Pass()]
    stub.decorator_list = []
    try:
        rendered = ast.unparse(stub)
    except Exception:
        rendered = ""
    text = rendered.split("\n", 1)[0] if rendered else ""
    if text:
        paren = text.find("(")
        close = _balance(text[paren:], "(", ")") if paren != -1 else -1
        if paren != -1 and close != -1:
            params = _space_defaults(text[paren : paren + close + 1])
            rest = text[paren + close + 1 :].strip()
            detail = params
            if rest.startswith("->"):
                returns = rest[2:].strip().rstrip(":").strip()
                if returns:
                    detail += f" -> {returns}"
            return _cap(detail, _MAX_DETAIL)
    return _manual_signature(node)


def _space_defaults(params: str) -> str:
    """``a=1`` becomes ``a = 1``, without touching ``=`` inside a string.

    ``ast.unparse`` omits the spaces PEP 8 wants around a keyword-only default,
    which makes every rendered signature look machine-printed next to the
    source it came from. A regex cannot do this safely -- ``f(x="a=b")`` would
    be corrupted -- so the scan tracks quotes, which is the only place a bare
    ``=`` can appear that is not a default assignment.
    """
    out: list[str] = []
    quote = ""
    index = 0
    while index < len(params):
        char = params[index]
        if quote:
            out.append(char)
            if char == "\\":
                out.append(params[index + 1 : index + 2])
                index += 2
                continue
            if char == quote:
                quote = ""
            index += 1
            continue
        if char in "\"'":
            quote = char
            out.append(char)
            index += 1
            continue
        if char == "=" and params[index + 1 : index + 2] != "=":
            while out and out[-1] in " \t":
                out.pop()
            out.append(" = ")
            index += 1
            while params[index : index + 1] == " ":
                index += 1
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _manual_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Fallback when ``ast.unparse`` refuses: read the AST fields directly."""
    a = node.args
    parts: list[str] = []
    positional = [*a.posonlyargs, *a.args]
    pad = [None] * (len(positional) - len(a.defaults))
    for arg, default in zip(positional, [*pad, *a.defaults], strict=True):
        parts.append(_arg(arg, default))
    if a.vararg is not None:
        parts.append("*" + _arg(a.vararg, None))
    elif a.kwonlyargs:
        parts.append("*")
    for arg, default in zip(a.kwonlyargs, a.kw_defaults, strict=True):
        parts.append(_arg(arg, default))
    if a.kwarg is not None:
        parts.append("**" + _arg(a.kwarg, None))
    detail = "(" + ", ".join(parts) + ")"
    if node.returns is not None:
        returns = _unparse(node.returns)
        if returns:
            detail += f" -> {returns}"
    return _cap(detail, _MAX_DETAIL)


def _arg(arg: ast.arg, default: ast.expr | None) -> str:
    text = arg.arg
    if arg.annotation is not None:
        ann = _unparse(arg.annotation)
        if ann:
            text += f": {ann}"
    if default is not None:
        value = _unparse(default)
        if value:
            text += " = " + _cap(value, _MAX_VALUE)
    return text


def _decorator_names(node: ast.AST) -> tuple[str, ...]:
    """Bare names of a node's decorators, outermost first.

    Bare and last-component-only: ``@app.get("/users")`` becomes ``get``. A
    decorator's arguments are a route string or a name, not structure, and
    rendering them costs tokens on every entry they appear on.
    """
    names: list[str] = []
    for dec in getattr(node, "decorator_list", []):
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Attribute):
            names.append(target.attr)
    return tuple(names)


def _enum_flag(bases: list[ast.expr]) -> str:
    for base in bases:
        target = base.value if isinstance(base, ast.Subscript) else base
        if isinstance(target, ast.Name):
            last = target.id
        elif isinstance(target, ast.Attribute):
            last = target.attr
        else:
            continue
        if last in _ENUM_BASES:
            return "enum"
    return ""


def _function_outline(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    kind: str = "function",
) -> Outline:
    names = _decorator_names(node)
    flags = ["async"] if isinstance(node, ast.AsyncFunctionDef) else []
    flags += [n for n in names if n in ("property", "staticmethod", "classmethod",
                                       "overload", "final", "abstractmethod")]
    return Outline(
        kind=kind,
        name=node.name,
        line=node.lineno,
        detail=_signature(node),
        children=_nested_declarations(node.body),
        doc=_docstring(node),
        decorators=names,
        flags=tuple(dict.fromkeys(flags)),
    )


def _class_outline(node: ast.ClassDef) -> Outline:
    names = _decorator_names(node)
    flags = list(dict.fromkeys(
        _SHAPE_DECORATORS[n] for n in names if n in _SHAPE_DECORATORS
    ))
    enum = _enum_flag(node.bases)
    if enum:
        flags.insert(0, enum)
    bases = [_unparse(b) for b in node.bases]
    bases += [f"{kw.arg}={_unparse(kw.value)}" for kw in node.keywords if kw.arg]
    return Outline(
        kind="class",
        name=node.name,
        line=node.lineno,
        detail="(" + ", ".join(bases) + ")" if bases else "",
        # `in_class` is what files a function under its class as a `method`
        # rather than a `function`. The distinction is load-bearing for a
        # renderer -- a method is drawn with no keyword, because the class line
        # above it already said what it is -- and it is also the honest reading:
        # a `def` in a class body is a method, never a nested closure.
        children=_python_body(node.body, in_class=True),
        doc=_docstring(node),
        decorators=names,
        flags=tuple(flags),
    )


def _const_detail(node: ast.expr | None, annotation: ast.expr | None = None) -> str:
    """``: ann = value`` for a short, single-line assignment.

    The one place a body could leak into an outline, so it is fenced twice: the
    expression must occupy a single source line *and* render short. A 900-line
    dict of locale strings becomes ``= ...`` rather than the file's most
    expensive line.
    """
    detail = ""
    if annotation is not None:
        ann = _unparse(annotation)
        if ann:
            detail += f": {_cap(ann, _MAX_DETAIL)}"
    if node is None:
        return detail
    names = _name_list(node)
    if names is not None:
        return detail + " = " + names
    end = getattr(node, "end_lineno", None)
    if end is not None and end > getattr(node, "lineno", 0):
        return detail + " = " + _ELIDED
    value = _unparse(node)
    if not value:
        return detail
    if len(value) > _MAX_VALUE:
        return detail + " = " + _ELIDED
    return detail + " = " + value


def _name_list(node: ast.expr) -> str | None:
    """``[a, b, c]`` for a flat collection of names, else ``None``.

    ``__all__`` and ``TypeVar``-style alias lists are pure names, and printing
    them is exactly the "what does this module expose" answer. Counting them
    instead (``= <30 items>``) would be cheaper and useless, so the list is
    printed up to a character cap and elided past it.
    """
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return None
    names = []
    for element in node.elts:
        if isinstance(element, ast.Constant) and isinstance(element.value, str):
            names.append(element.value)
        else:
            return None
    if not names:
        return None
    rendered = "[" + ", ".join(names) + "]"
    return _cap(rendered, _MAX_NAMES)


def _assigned_names(target: ast.expr) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, (ast.Tuple, ast.List)):
        return tuple(t.id for t in target.elts if isinstance(t, ast.Name))
    if isinstance(target, ast.Attribute):
        return (target.attr,)
    if isinstance(target, ast.Starred):
        return _assigned_names(target.value)
    return ()


def _assignment_outlines(stmt: ast.Assign | ast.AnnAssign) -> list[Outline]:
    annotation = stmt.annotation if isinstance(stmt, ast.AnnAssign) else None
    targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
    detail = _const_detail(stmt.value, annotation)
    out: list[Outline] = []
    for target in targets:
        out.extend(
            Outline(kind="const", name=name, line=stmt.lineno, detail=detail)
            for name in _assigned_names(target)
        )
    return out


def _nested_declarations(body: list[ast.stmt]) -> tuple[Outline, ...]:
    """Only ``def``/``class`` found directly inside a function body.

    A closure is part of a function's contract and hiding it makes the outline
    lie, so nested declarations are kept. Everything else in the body is not:
    those are local variables and intermediate steps, and including them is what
    would turn an outline back into a body.
    """
    out: list[Outline] = []
    for stmt in body:
        if isinstance(stmt, ast.ClassDef):
            out.append(_class_outline(stmt))
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(_function_outline(stmt))
    return tuple(out)


def _python_body(body: list[ast.stmt], *, in_class: bool = False) -> list[Outline]:
    """Outline one block of statements. Never descends into a function body.

    Imports are handled here rather than only at module level because a class
    body or a ``TYPE_CHECKING`` block can import too, and an outline that
    silently drops those imports is inconsistent about which imports count.

    ``in_class`` decides whether a ``def`` here is a method or a function. It
    does not carry into nested blocks: a ``def`` inside an ``if`` inside a class
    body is still a method, and the flag says so.
    """
    out: list[Outline] = []
    run: list[str] = []
    run_line = 0

    def flush() -> None:
        nonlocal run, run_line
        if run:
            out.append(
                Outline("import", _cap("import " + ", ".join(run), _MAX_DETAIL), run_line)
            )
            run = []
            run_line = 0

    for stmt in body:
        if isinstance(stmt, ast.Import):
            if not run:
                run_line = stmt.lineno
            run.extend(
                a.name + (f" as {a.asname}" if a.asname else "") for a in stmt.names
            )
            continue
        flush()
        if isinstance(stmt, ast.ImportFrom):
            module = "." * stmt.level + (stmt.module or "")
            names = ", ".join(
                a.name + (f" as {a.asname}" if a.asname else "") for a in stmt.names
            )
            out.append(
                Outline(
                    "import",
                    _cap(f"from {module} import {names}", _MAX_DETAIL),
                    stmt.lineno,
                )
            )
        elif isinstance(stmt, ast.ClassDef):
            out.append(_class_outline(stmt))
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(
                _function_outline(stmt, kind="method" if in_class else "function")
            )
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            out.extend(_assignment_outlines(stmt))
        elif isinstance(stmt, ast.If):
            section = _section_outline(stmt)
            if section is not None:
                out.append(section)
            else:
                # A version guard or a `try: from x import y except ImportError`
                # still holds real definitions. Recurse without inventing a
                # header: a `section` is a claim that its children only exist
                # under a condition, and this one is not worth claiming.
                out.extend(_python_body(stmt.body, in_class=in_class))
                out.extend(_python_body(stmt.orelse, in_class=in_class))
        elif isinstance(stmt, ast.Try):
            out.extend(_python_body(stmt.body, in_class=in_class))
            for handler in stmt.handlers:
                out.extend(_python_body(handler.body, in_class=in_class))
            out.extend(_python_body(stmt.orelse, in_class=in_class))
            out.extend(_python_body(stmt.finalbody, in_class=in_class))
        elif isinstance(stmt, (ast.With, ast.For, ast.While)):
            out.extend(_python_body(stmt.body, in_class=in_class))
        elif _MATCH_STMT is not None and isinstance(stmt, _MATCH_STMT):
            # `match`/`case` is the one compound statement Python 3.10 added, and
            # omitting it means every declaration inside a match arm disappears
            # from the outline. For a navigation aid that is the worst possible
            # failure: the reader is told those symbols do not exist. They very
            # much do.
            for case in stmt.cases:
                out.extend(_python_body(case.body, in_class=in_class))
    flush()
    return out


def _section_outline(node: ast.If) -> Outline | None:
    """Wrap a type-checking or version guard as a ``section``.

    Descending without a marker would silently promote type-only declarations to
    real ones, which is worse than dropping them: an agent would believe the
    class exists at runtime. The header keeps the caveat visible exactly where it
    applies, and ``kind="section"`` is what tells a renderer not to print a
    keyword in front of it.
    """
    test = _unparse(node.test)
    if not any(marker in test.upper() for marker in _SECTION_TESTS):
        return None
    entries = _python_body(node.body, in_class=True)
    if not entries:
        return None
    return Outline(kind="section", name=f"if {test}:", line=node.lineno,
                   children=tuple(entries))


def _parse_quietly(text: str) -> ast.Module:
    """``ast.parse`` without the ``SyntaxWarning`` chatter.

    The compiler emits a ``SyntaxWarning`` for things like an unescaped backslash
    in a string or a stray ``\\`` in a docstring, and then raises the
    ``SyntaxError`` that got us here. Under ``-W error`` -- or any host that
    promoted warnings to errors -- that warning would escape a function whose
    entire contract is "never raise on bad input". Nothing is lost by silencing
    it: the file has failed to parse either way.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(text)


def python_outline(text: str, *, path: str = "") -> list[Outline]:
    """Structure of a Python file: functions, classes, constants, imports.

    Uses the stdlib ``ast``, so line numbers and signatures are exact rather than
    guessed and a multi-line signature collapses to one line. Decorators are
    kept as bare names, dataclasses and enums are flagged, ``__all__`` renders as
    a name list, ``TYPE_CHECKING`` blocks become ``section`` entries with their
    contents indented underneath, and ``async def`` is marked. Function bodies
    are never descended -- except for a nested ``def``/``class``, which is part
    of the enclosing function's contract.

    A syntax error falls back to the line scanner in :func:`_python_fallback`,
    which recovers the common declarations from a truncated file. It never raises.
    """
    text = text or ""
    if not text.strip():
        return unstructured(text, reason="empty file")
    try:
        tree = _parse_quietly(text)
    except Exception:
        # Overwhelmingly a file being edited right now or a template with
        # placeholders in it: ordinary input, not a ctxpack failure. A hostile
        # input (deep nesting, a stray NUL) lands here too, which is the point.
        scanned = _python_fallback(text)
        # Count every node, not the top-level ones: a truncated file usually
        # recovers exactly one class holding every method that was readable,
        # and that is a useful outline, not an empty one.
        if sum(1 for entry in scanned for _ in entry.walk()) > 1:
            return scanned
        return unstructured(text, reason="syntax error, no declarations recovered")
    entries: list[Outline] = []
    doc = _docstring(tree)
    if doc:
        entries.append(Outline(kind="other", name=_cap(doc, _MAX_DETAIL), line=1))
    try:
        entries.extend(_python_body(tree.body))
    except RecursionError:  # pragma: no cover - pathological nesting
        return unstructured(text, reason="nesting too deep to outline")
    except Exception:  # pragma: no cover - defensive
        return unstructured(text, reason="unexpected parse failure")
    if not entries or (len(entries) == 1 and entries[0].kind == "other"):
        # Parsed cleanly but there is nothing in it -- a licence header, a
        # ``# type: ignore`` file, a comment-only module. Still a placeholder,
        # for the same reason as everywhere else.
        return unstructured(text, reason="no declarations found")
    if any(path.endswith(ext) for ext in _STUB_EXTS):
        entries = [
            e if "stub" in e.flags else replace(e, flags=(*e.flags, "stub"))
            for e in entries
        ]
    return entries


# --------------------------------------------------------------------------
# Python fallback: line-oriented, for files ast.parse rejects
# --------------------------------------------------------------------------


_PY_DECORATOR = re.compile(r"^@([A-Za-z_][\w.]*)")
_PY_DEF = re.compile(r"^(?P<async>async\s+)?def\s+(?P<name>[A-Za-z_]\w*)")
_PY_CLASS = re.compile(r"^class\s+(?P<name>[A-Za-z_]\w*)")
_PY_IMPORT = re.compile(r"^(?:from\s+(?P<module>[.\w]+)\s+import|import\s+)(?P<names>.+)$")
_PY_CONST = re.compile(r"^(?P<name>[A-Za-z_]\w*)\s*(?::[^=]+)?\s*=")


def _python_fallback(text: str) -> list[Outline]:
    """Best-effort outline of Python that would not parse. Heuristic.

    Used for truncated files, templates and hostile input. Class membership is
    decided by indentation rather than braces, which is the one thing a regex
    scanner can get right about Python; everything else is a single-line pattern
    match. Names are right far more often than signatures, and a name is enough
    to answer "is this in here".
    """
    out: list[Outline] = []
    # (indent, children so far) rather than a half-built Outline: ``Outline`` is
    # frozen, so a class cannot gain children as its body is read. It is
    # materialised by `_close` when the dedent arrives.
    stack: list[tuple[int, str, int, list[Outline], tuple[str, ...]]] = []
    decorator = ""
    for lineno, raw in enumerate(_source_lines(text), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        while stack and indent <= stack[-1][0]:
            _close(stack, out)
        match = _PY_DECORATOR.match(stripped)
        if match:
            # Bare last component, matching the AST path: `@app.get("/x")` is a
            # route string in disguise, and routes are not structure.
            decorator = match[1].rsplit(".", 1)[-1]
            continue
        match = _PY_DEF.match(stripped)
        if match:
            _attach(
                Outline(
                    "method" if stack else "function",
                    match["name"],
                    lineno,
                    detail=_py_params(stripped),
                    flags=("async",) if match["async"] else (),
                    decorators=(decorator,) if decorator else (),
                ),
                stack,
                out,
            )
            decorator = ""
            continue
        match = _PY_CLASS.match(stripped)
        if match:
            stack.append(
                (
                    indent,
                    match["name"],
                    lineno,
                    [],
                    (decorator,) if decorator else (),
                )
            )
            decorator = ""
            continue
        match = _PY_IMPORT.match(stripped)
        if match:
            module = match["module"]
            names = match["names"].strip()
            rendered = f"from {module} import {names}" if module else f"import {names}"
            _attach(Outline("import", _cap(rendered, _MAX_DETAIL), lineno), stack, out)
            continue
        match = _PY_CONST.match(stripped)
        if match and match["name"][:1].isupper():
            _attach(Outline("const", match["name"], lineno), stack, out)
            continue
        # Any other line means the decorator we were holding belonged to
        # something the patterns did not recognise, so drop it rather than
        # attaching it to the next declaration.
        decorator = ""
    while stack:
        _close(stack, out)
    return out


def _attach(entry: Outline, stack: list, out: list[Outline]) -> None:
    """Nest ``entry`` under the innermost open class, else append at top level."""
    if stack:
        stack[-1][3].append(entry)
    else:
        out.append(entry)


def _close(stack: list, out: list[Outline]) -> None:
    _, name, lineno, children, decorators = stack.pop()
    node = Outline(
        "class", name, lineno, children=tuple(children), decorators=decorators
    )
    if stack:
        stack[-1][3].append(node)
    else:
        out.append(node)


def _py_params(stripped: str) -> str:
    paren = stripped.find("(")
    if paren == -1:
        return ""
    close = _balance(stripped[paren:], "(", ")")
    if close == -1:
        # Truncated mid-signature: keep what is there. A half signature is still
        # navigable; nothing is not.
        return " " + _cap(stripped[paren:], _MAX_DETAIL)
    return " " + _cap(stripped[paren : paren + close + 1], _MAX_DETAIL)


def _py_bases(stripped: str) -> str:
    paren = stripped.find("(")
    if paren == -1:
        return ""
    close = _balance(stripped[paren:], "(", ")")
    if close == -1:
        return ""
    return _cap(stripped[paren : paren + close + 1], _MAX_DETAIL)


# --------------------------------------------------------------------------
# JavaScript / TypeScript -- regex, on purpose
# --------------------------------------------------------------------------

_JS_FUNCTION = re.compile(
    r"^(?P<export>export\s+)?(?P<default>default\s+)?(?P<async>async\s+)?"
    r"function\s*\*?\s*(?P<name>[A-Za-z_$][\w$]*)?\s*(?:<[^()]*>\s*)?\("
)
#: ``const name = ...`` where the initialiser is a function: an arrow (parens or
#: bare identifier), a ``function`` expression, or an ``async`` variant of
#: either. Three shapes, one pattern, because a second pattern would be a
#: second place to get ``async`` and the type annotation wrong.
_JS_FUNCTION_VALUE = re.compile(
    r"^(?P<export>export\s+)?(?P<default>default\s+)?"
    r"(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)"
    r"(?:\s*:\s*[^=]+?)?\s*=\s*(?P<async>async\s+)?"
    r"(?:function\s*\*?\s*(?:[A-Za-z_$][\w$]*)?\s*\((?P<fnparams>[^()]*)\)"
    r"|(?P<params>\([^()]*\)|[A-Za-z_$][\w$]*)\s*(?::[^=]*?)?=>)"
)
#: Any other ``const``/``let``/``var``. Module-level constants are the JS
#: equivalent of Python's, and an outline that shows one language's constants
#: but not the other's is harder to read than one that shows neither.
_JS_CONST = re.compile(
    r"^(?P<export>export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)"
    r"\s*(?::\s*(?P<ann>[^=]+?))?\s*=\s*(?P<value>[^;]*)"
)
_JS_CLASS = re.compile(
    r"^(?P<export>export\s+)?(?P<default>default\s+)?(?P<abstract>abstract\s+)?"
    r"class\s+(?P<name>[A-Za-z_$][\w$]*)\s*(?P<rest>[^{]*)"
)
_JS_INTERFACE = re.compile(
    r"^(?P<export>export\s+)?(?P<declare>declare\s+)?"
    r"interface\s+(?P<name>[A-Za-z_$][\w$]*)\s*(?P<rest>[^{]*)"
)
_JS_ENUM = re.compile(
    r"^(?P<export>export\s+)?(?:const\s+)?enum\s+(?P<name>[A-Za-z_$][\w$]*)"
)
_JS_TYPE = re.compile(
    r"^(?P<export>export\s+)?(?P<declare>declare\s+)?type\s+"
    r"(?P<name>[A-Za-z_$][\w$]*)\s*(?:<[^=]*>)?\s*=\s*(?P<body>.+?);?\s*$"
)
_JS_MEMBER = re.compile(
    r"^(?P<mods>(?:public\s+|private\s+|protected\s+|readonly\s+|static\s+|"
    r"abstract\s+|override\s+|async\s+|get\s+|set\s+)*)"
    r"(?P<priv>#)?(?P<name>[A-Za-z_$][\w$]*)\s*(?:<[^>(]*>)?\s*\("
)
#: A class field holding an arrow function: ``onClick = () => {...}``. Has no
#: ``const`` keyword, which is why it needs its own pattern rather than a
#: variant of :data:`_JS_FUNCTION_VALUE`.
_JS_FIELD_FUNCTION = re.compile(
    r"^(?P<mods>(?:public\s+|private\s+|protected\s+|readonly\s+|static\s+|"
    r"abstract\s+|override\s+|async\s+|declare\s+)*)"
    r"(?P<priv>#)?(?P<name>[A-Za-z_$][\w$]*)\s*(?::\s*[^=]+?)?\s*=\s*"
    r"(?P<async>async\s+)?(?:function\s*\*?\s*(?:[A-Za-z_$][\w$]*)?\s*)?\("
)
#: A plain field of a class body, an interface or an enum member:
#: ``id: string``, ``name?: string``, ``Active = 1``, ``Red,``.
#:
#: The value is non-greedy and the tail is anchored, so a comma *inside* the
#: value -- ``new Map<string, User>()`` -- is not mistaken for the field
#: separator. A line carrying two fields (``a: string, b: number``) does not
#: match at all, and is dropped rather than half-reported: a field outline that
#: silently showed one of the two would be worse than one that showed neither,
#: because the reader cannot tell which happened.
_JS_FIELD = re.compile(
    r"^(?P<mods>(?:readonly\s+|static\s+|declare\s+|public\s+|private\s+|"
    r"protected\s+|abstract\s+)*)"
    r"(?P<priv>#)?(?P<name>[A-Za-z_$][\w$]*)\s*(?P<ann>\??\s*:\s*[^;,=]+)?"
    r"\s*(?P<value>=[^;]*?)?\s*[,;]?\s*$"
)
#: Words that look like a method call at the start of a line but are not
#: declarations. Without this, ``if (x) {`` becomes a method named ``if``.
_JS_NOT_DECLARATIONS = frozenset(
    {
        "await", "catch", "do", "else", "export", "for", "function", "if",
        "import", "new", "return", "super", "switch", "throw", "typeof",
        "while", "yield",
    }
)
_JS_DEFAULT = re.compile(r"^export\s+default\s+(?P<body>.+?);?\s*$")

#: Kinds whose brace-delimited body is worth descending into. Interfaces and
#: enums are here for the same reason classes are: an interface's fields and an
#: enum's members are the contract, and an outline that shows the type but not
#: its shape answers only half of "what does this expose?".
_JS_BODIES = frozenset({"class", "interface", "enum"})

#: Modifier keyword to flag. One table so a modifier added to a pattern is one
#: entry here rather than three separate ``if mod == ...`` chains.
_JS_MOD_FLAGS = {
    "abstract": "abstract",
    "async": "async",
    "declare": "declare",
    "get": "getter",
    "private": "",
    "protected": "",
    "public": "",
    "override": "final",
    "readonly": "",
    "set": "setter",
    "static": "static",
}


@dataclass
class _Frame:
    """An open JS block, and the children found directly inside it.

    Mutable because ``Outline`` is frozen: children are collected here and the
    tree is rebuilt when the frame closes. ``open_depth`` is the brace depth the
    frame's *contents* live at, so a nested function's locals are recognised as
    deeper than the class body and can be ignored.
    """

    entry: Outline
    children: list[Outline]
    open_depth: int
    parent: list[Outline]
    index: int


def js_outline(text: str) -> list[Outline]:
    """Structure of a JS/TS file, matched one line at a time with regexes.

    HEURISTIC, and it has to be: the stdlib ships a Python parser and no parser
    for JavaScript, and ctxpack has no dependencies. What it gets right is what
    an outline needs -- exported and plain functions, arrow and ``function``
    expression consts, classes with their methods (static, getter and setter
    markers included), ``interface``, ``type`` aliases, ``enum``, imports, and
    ``export default`` in each of its forms. Comments are blanked first, so a
    commented-out old implementation does not double the outline.

    What it gets wrong: class bodies are closed by counting ``{`` and ``}``, so
    a brace inside a string or a regex literal desynchronises the nesting and
    misplaces later entries; a declaration whose keyword is not at the start of
    a line can be missed; and TypeScript overload signatures collapse onto one
    another. Output stays in source order and stays good enough to navigate,
    which is the whole promise of an outline. Nothing here raises, at any depth,
    on any input.
    """
    text = text or ""
    if not text.strip():
        return unstructured(text, reason="empty file")
    # Two views of the same lines. `_strip_js_comments` blanks comment *bodies*
    # so a commented-out declaration cannot be matched; `_js_docs` needs the
    # comment text itself. Both preserve the line count exactly, so line numbers
    # mean the same thing in each.
    lines = _source_lines(_strip_js_comments(text))
    docs = _js_docs(_source_lines(text))
    out: list[Outline] = []
    stack: list[_Frame] = []
    depth = 0
    doc = ""
    index = 0
    while index < len(lines):
        lineno = index + 1
        raw = lines[index]
        stripped = raw.strip()
        if not stripped or stripped.startswith(("/*", "//")):
            index += 1
            continue
        # A signature spread over several lines is the single most common way
        # JS source defeats a line-oriented scanner: the regex matches the first
        # line, then sees an unclosed paren and has nothing useful to render.
        #
        # The join is attempted *after* a match, not before, and only when the
        # match's own detail came out truncated. Deciding it the other way round
        # -- speculatively joining anything with an open bracket -- swallowed the
        # closing brace of an enum and silently ate the rest of the file. Paying
        # one extra regex on the rare multiline signature is the cheap direction.
        consumed = 0
        if lineno in docs and not out and not stack:
            # A JSDoc block before the first declaration is the module's own
            # summary -- the counterpart of a Python module docstring -- and gets
            # an entry of its own because it is usually the single most useful
            # line in the file. It is numbered at the comment's first line, so
            # jumping to it lands on the documentation rather than past it.
            out.append(
                Outline(
                    kind="other",
                    name=_cap(docs[lineno][1], _MAX_DETAIL),
                    line=docs[lineno][0],
                )
            )
            doc = ""
        elif lineno in docs:
            doc = docs[lineno][1]
        # "Directly inside a class body" is the only nesting an outline claims.
        # Two levels down is a method's locals, and a local `const` is not part
        # of the class's contract.
        in_class_body = bool(stack) and depth == stack[-1].open_depth + 1
        try:
            entry, doc = _js_entry(stripped, lineno, doc, inside=in_class_body)
            if (
                entry is not None
                and (depth == 0 or in_class_body)
                and _needs_continuation(stripped, entry.detail)
            ):
                # Only ever join a declaration that will be kept. Joining one
                # inside a function body cannot help -- it is about to be
                # discarded -- but it eats the lines after it, and those are
                # declarations that were going to be kept.
                joined, consumed = _join_continuation(lines, index)
                if consumed:
                    stripped = joined
                    entry, doc = _js_entry(joined, lineno, doc, inside=in_class_body)
        except Exception:  # pragma: no cover - defensive
            entry, doc = None, doc
        if entry is not None and (depth == 0 or in_class_body):
            # A declaration found inside a function body is a local, and putting
            # it at the top level would be a lie about where it lives. Dropping
            # it is the same call Python makes, for the same reason.
            parent = stack[-1].children if in_class_body else out
            parent.append(entry)
            if entry.kind in _JS_BODIES and _opens_block(stripped):
                if len(stack) >= MAX_NESTING_DEPTH:
                    # Past the limit: stop descending rather than recursing until
                    # the interpreter gives up. A flat outline of pathological
                    # input beats a RecursionError, and beats nothing at all.
                    return unstructured(
                        stripped, reason="nesting too deep to outline"
                    )
                stack.append(_Frame(entry, [], depth, parent, len(parent) - 1))
        depth += stripped.count("{") - stripped.count("}")
        if depth < 0:
            # Unbalanced braces mean the nesting counter has lost track. Closing
            # everything is the honest response: a flat outline is still useful,
            # a wrongly nested one is worse than useless.
            depth = 0
            stack.clear()
        while stack and depth <= stack[-1].open_depth:
            _close_frame(stack)
        # `consumed` skips the continuation lines the join already absorbed.
        # Without it they are re-scanned as declarations of their own, and a
        # parameter named `id` becomes a top-level entry.
        index += 1 + consumed
    while stack:
        _close_frame(stack)
    if not out:
        return unstructured(text, reason="no declarations found")
    return out


def _close_frame(stack: list[_Frame]) -> None:
    """Freeze a finished frame's children back into its entry, in its parent."""
    frame = stack.pop()
    frame.parent[frame.index] = replace(frame.entry, children=tuple(frame.children))


#: How many lines a signature may be spread over before we stop joining. Twelve
#: covers every hand-written signature in practice; past that the construct is
#: data, not code.
_MAX_JOIN_LINES = 12


def _needs_continuation(source: str, detail: str) -> bool:
    """True when a declaration ran past the end of its source line.

    Both halves matter. The *source* line ending in an operator (``=``, ``|``,
    ``=>``) is how a broken ``const`` chain announces itself, and there the
    rendered detail is legitimately empty. The *detail* carrying an unbalanced
    ``(`` is how a broken parameter list announces itself. Either alone would
    miss half the cases; together they cover both without ever firing on a
    declaration that was captured whole, which is what keeps the retry from
    eating a closing brace.
    """
    # A trailing comma is deliberately NOT a signal. It is how interface fields
    # and enum members end, so treating it as a continuation joined every member
    # to the next line and swallowed the closing brace of the body. The operators
    # that actually break a line across lines are listed here; commas never do.
    if source.rstrip().endswith(("=", "|", "&", "=>", "+")):
        return True
    if _paren_depth(source) > 0:
        return True
    return bool(detail) and detail.count("(") > detail.count(")")


def _join_continuation(lines: list[str], start: int) -> tuple[str, int]:
    """Join a declaration's continuation lines into one, and report how many.

    Joins while parentheses are still open, or while the line ends in an operator
    (``=``, ``|``, ``&``, ``=>``, which is how a long ``const`` chain breaks).
    Returns ``(joined_text, lines_consumed)``; consumed is zero when the first
    line was already self-contained.

    Only reached once a declaration has already matched *and* produced a
    truncated detail, so it cannot fire on a statement inside a function body.

    Only parentheses are counted, not angle brackets. ``<`` and ``>`` are
    ambiguous in JavaScript -- generic parameters, arrow functions and the
    comparison operators all look the same -- and miscounting them once cost a
    real file three declarations, which is exactly the failure this module
    promises cannot happen.
    """
    first = lines[start].strip()
    if _paren_depth(first) <= 0 and not _ends_mid_expression(first):
        return first, 0
    parts = [first]
    last = start
    depth = _paren_depth(first)
    for offset in range(1, _MAX_JOIN_LINES):
        if start + offset >= len(lines):
            break
        parts.append(lines[start + offset].strip())
        last = start + offset
        depth += _paren_depth(parts[-1])
        if depth <= 0 and not _ends_mid_expression(parts[-1]):
            break
    return " ".join(parts), last - start


def _paren_depth(text: str) -> int:
    """Net parenthesis depth of a line, counting every bracket on it.

    Deliberately counts the whole line rather than stopping at the first ``{``.
    Stopping there looks more precise and is not: ``fetch(`${BASE}/${id}`)``
    contains a ``{`` inside a template literal, so cutting at it leaves a lone
    ``(`` and the line looks like a signature that continues -- which is how a
    one-line ``await fetch(...)`` inside a function body swallowed the next
    eleven lines of the file.

    Not counting ``{}`` at all is deliberate too: a declaration line that opens
    a body still has balanced parentheses, and one that does not is not a
    declaration.
    """
    return text.count("(") - text.count(")")


def _ends_mid_expression(text: str) -> bool:
    """True when a line clearly continues onto the next one."""
    stripped = text.rstrip()
    if not stripped or stripped.endswith((";", "{")):
        return False
    if stripped.endswith(("=", "|", "&", "+", ".", "=>")):
        return True
    return _paren_depth(stripped) > 0


def _opens_block(stripped: str) -> bool:
    return stripped.count("{") > 0


def _js_entry(
    stripped: str, lineno: int, doc: str, *, inside: bool
) -> tuple[Outline | None, str]:
    """Match one line against the declaration patterns, in priority order.

    Comment lines never reach here: :func:`_js_docs` has already removed them
    from consideration, which keeps "this line declares something" a question
    about declarations only.
    """
    if stripped.startswith("import"):
        if stripped.startswith("import(") or stripped.startswith("import."):
            return None, doc  # dynamic import, not a declaration
        return (
            Outline("import", _cap(stripped.rstrip(";"), _MAX_DETAIL), lineno),
            "",
        )

    for pattern, kind in ((_JS_CLASS, "class"), (_JS_INTERFACE, "interface"),
                          (_JS_ENUM, "enum")):
        match = pattern.match(stripped)
        if not match:
            continue
        # Read through groupdict because the three patterns do not share a
        # group set: `enum Name {` has nothing after the name at all.
        rest = _cap(" ".join(match.groupdict().get("rest", "").split()), _MAX_DETAIL)
        detail = ""
        if kind in ("class", "interface") and rest.startswith("extends "):
            # `extends Base` is the inheritance question. `implements X` is not
            # included: it is a promise about shape rather than reuse, and the
            # interface body below already shows the shape.
            detail = f"({rest[8:].split(' implements')[0].strip()})"
        return (
            Outline(kind, match["name"], lineno, detail=detail, doc=doc,
                    flags=_js_flags(match, stripped)),
            "",
        )

    match = _JS_TYPE.match(stripped)
    if match:
        body = _cap(match["body"], _MAX_DETAIL - 3)
        return (
            Outline("type", match["name"], lineno, detail=f" = {body}", doc=doc,
                    flags=_js_flags(match, stripped)),
            "",
        )

    match = _JS_FUNCTION.match(stripped)
    if match:
        # An anonymous `export default function () {}` has no name to show, and
        # "default" is the name the module system will actually import it under.
        name = match["name"] or "default"
        return (
            Outline("function", name, lineno, detail=_js_params(stripped),
                    doc=doc, flags=_js_flags(match, stripped)),
            "",
        )

    match = _JS_FUNCTION_VALUE.match(stripped)
    if match:
        # An arrow assigned to a class field is a method in every way that
        # matters to a reader, so it is filed as one.
        raw = match["params"]
        if raw is None:
            params = "(" + (match["fnparams"] or "") + ")"
        else:
            params = raw if raw.startswith("(") else f"({raw})"
        # The return annotation sits *before* the `=>`, so the arrow's own
        # position is what `_js_returns` needs -- not where the match ended.
        arrow_at = stripped.rfind("=>", match.start(), match.end())
        detail = params + _js_returns(stripped, arrow_at)
        return (
            Outline("method" if inside else "function", match["name"], lineno,
                    detail=_cap(detail, _MAX_DETAIL), doc=doc,
                    flags=_js_flags(match, stripped)),
            "",
        )

    match = _JS_CONST.match(stripped)
    if match and not stripped.rstrip(";").endswith(("=>", "{")):
        detail = ""
        if match["ann"]:
            detail += f": {_cap(match['ann'], _MAX_DETAIL)}"
        raw = match["value"].strip().rstrip(";")
        if raw:
            # Trailing operators survive here because the value was cut off at a
            # line break and `_join_continuation` has not run yet. Showing them
            # is what makes `_detail_is_cut_off` able to notice the truncation,
            # and the retry replaces the whole detail anyway.
            detail += f" = {_cap(raw, _MAX_VALUE)}" if len(raw) <= _MAX_VALUE else (
                " = " + _ELIDED
            )
        return (
            Outline("const", match["name"], lineno, detail=detail, doc=doc,
                    flags=_js_flags(match, stripped)),
            "",
        )

    if inside:
        match = _JS_MEMBER.match(stripped)
        if match and match["name"] not in _JS_NOT_DECLARATIONS:
            flags = [
                flag
                for mod in (match["mods"] or "").split()
                for flag in [_JS_MOD_FLAGS.get(mod, "")]
            ]
            return (
                Outline("method", ("#" if match["priv"] else "") + match["name"],
                        lineno, detail=_js_params(stripped), doc=doc,
                        flags=tuple(flag for flag in flags if flag)),
                "",
            )
        match = _JS_FIELD_FUNCTION.match(stripped)
        if match and match["name"] not in _JS_NOT_DECLARATIONS:
            mods = (match["mods"] or "").split()
            flags = [f for m in mods for f in [_JS_MOD_FLAGS.get(m, "")]]
            if match["async"]:
                flags.append("async")
            return (
                Outline("method", ("#" if match["priv"] else "") + match["name"],
                        lineno, detail=_js_params(stripped), doc=doc,
                        flags=tuple(f for f in flags if f)),
                "",
            )
        match = _JS_FIELD.match(stripped)
        if match and match["name"] not in _JS_NOT_DECLARATIONS:
            detail = ""
            if match["ann"]:
                detail = ": " + _cap(match["ann"].lstrip("? ").lstrip(": "), _MAX_DETAIL)
            value = (match["value"] or "").lstrip("= ").strip()
            if value:
                detail += " = " + (_cap(value, _MAX_VALUE)
                                   if len(value) < _MAX_VALUE else _ELIDED)
            return (
                Outline("const", ("#" if match["priv"] else "") + match["name"],
                        lineno, detail=detail, doc=doc),
                "",
            )

    match = _JS_DEFAULT.match(stripped)
    if match:
        return (
            Outline("other", "default export", lineno,
                    detail=" = " + _cap(match["body"], _MAX_VALUE),
                    flags=("default",)),
            "",
        )
    return None, doc


def _js_returns(stripped: str, arrow_at: int) -> str:
    """``: Promise<User | null>`` from the annotation before an arrow's ``=>``.

    The counterpart to the return half of :func:`_js_params`, for the arrow form
    where the parameters were matched by the regex rather than by balancing
    parentheses. ``arrow_at`` is the index of the ``=>``, which anchors the
    annotation to the right place; everything after the parameters' closing
    paren and before the arrow is the return type, if there is one.

    Bounded by the arrow rather than by ``{`` or ``;`` because those can appear
    *inside* a return type -- ``() => { a: 1 }`` is a valid annotation -- while
    the arrow always ends it.
    """
    if arrow_at == -1:
        return ""
    head = stripped[:arrow_at].rstrip()
    paren = head.rfind(")")
    if paren == -1:
        return ""
    tail = head[paren + 1 :].strip()
    if not tail.startswith(":"):
        return ""
    return f": {tail[1:].strip()}"


def _js_flags(match: re.Match[str], stripped: str) -> tuple[str, ...]:
    """Flags a declaration carries, from its match and its raw text.

    Read through ``groupdict`` rather than by index: the patterns do not all
    define the same groups, and a shared ``.get`` is the only version of this
    that does not need a per-pattern branch to stay correct when one is edited.
    """
    groups = match.groupdict()
    flags: list[str] = []
    if groups.get("async"):
        flags.append("async")
    if groups.get("abstract"):
        flags.append("abstractclass")
    if groups.get("declare"):
        flags.append("declare")
    if groups.get("export") or stripped.startswith("export"):
        flags.append("export")
    if groups.get("default") or stripped.startswith("export default"):
        flags.append("default")
    return tuple(flags)


def _js_params(stripped: str, start: int = 0) -> str:
    """``(a, b): Ret`` -- parameters and the return annotation that follows them.

    The return type is included because in TypeScript it *is* half the contract,
    and it is nearly free: it is a handful of tokens sitting on a line that was
    already being spent. For Python the equivalent is the ``-> R`` that
    :func:`_signature` pulls off the AST.

    Multi-line parameter lists are routine in TypeScript and would otherwise
    swallow the rest of the file, so this cuts at the closing paren -- or at the
    detail cap when the list runs past the end of the line. A partial signature
    is honest and still navigable.
    """
    paren = stripped.find("(", start)
    if paren == -1:
        return ""
    close = _balance(stripped[paren:], "(", ")")
    if close == -1:
        return _cap(stripped[paren:], _MAX_DETAIL)
    detail = stripped[paren : paren + close + 1]
    rest = stripped[paren + close + 1 :].lstrip()
    if not rest.startswith(":"):
        return _cap(detail, _MAX_DETAIL)
    # Stop at the body brace or the statement semicolon, whichever comes first:
    # everything past it is implementation, which is the one thing excluded here.
    stop = len(rest)
    for marker in ("{", ";"):
        found = rest.find(marker)
        if found != -1:
            stop = min(stop, found)
    returns = rest[1:stop].strip()
    if not returns:
        return _cap(detail, _MAX_DETAIL)
    return _cap(f"{detail}: {returns}", _MAX_DETAIL)


def _js_docs(lines: list[str]) -> dict[int, tuple[int, str]]:
    """Map each declaration's line number to ``(comment_line, its first line)``.

    Keyed on the *declaration's* line, not the comment's. That is the whole
    point of doing this in a pre-pass: a JSDoc block spans several lines, so
    matching line by line would attach the text to whichever line happened to be
    scanned next -- which is often an import, or nothing at all. The comment's
    own line number comes along so the module-summary entry can point at the
    documentation rather than at the first import.

    Only the first meaningful line of a block is kept, and ``@param``-style tag
    lines are skipped. A JSDoc block with full tag documentation runs to forty
    lines, and an outline that reproduces it has stopped being an outline.

    Two kinds of comment are deliberately *not* collected: ``//`` comments above
    a declaration, which are far more often a note to the next person
    (``// eslint-disable``, ``// @ts-ignore``, ``// FIXME``) than documentation;
    and plain ``/* */`` blocks, which are explanatory asides. Only ``/** */``
    counts as documentation, because that is the convention the language set and
    because a rule that fires on every comment is a rule that cannot be trusted.
    """
    docs: dict[int, tuple[int, str]] = {}
    pending: tuple[int, str] = (0, "")
    in_doc = False
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if in_doc:
            # Inside a multi-line JSDoc block. The first line that reads as
            # prose wins; the rest (`@param`, `@returns`, more `*` rows) is
            # dropped, which is where most of the tokens are.
            text = stripped.lstrip("*").strip()
            if not pending[1] and not text.startswith("@"):
                pending = (pending[0], _first_line(text))
            if "*/" in stripped:
                in_doc = False
            continue
        if stripped.startswith("/**"):
            in_doc = "*/" not in stripped
            body = stripped[3:].split("*/")[0]
            text = body.strip().lstrip("*").strip()
            pending = (lineno, "" if text.startswith("@") else _first_line(text))
            continue
        if stripped.startswith("/*"):
            continue  # a plain block comment is a note, not documentation
        if stripped.startswith("//"):
            # A `//` between a JSDoc block and its declaration severs the link,
            # which matches how a reader reads it: the note is about the next
            # thing, not this one.
            pending = (0, "")
            continue
        if not stripped:
            continue
        if pending[1]:
            docs[lineno] = pending
            pending = (0, "")
    return docs


def _strip_js_comments(text: str) -> str:
    """Blank out comments, preserving line structure exactly.

    Line structure matters: entries are numbered against the returned text, so
    if this dropped a line, every line number below it would be wrong. Comment
    characters become spaces rather than being removed, for the same reason.

    String literals are left alone deliberately. Removing them is a real parse
    and a real source of bugs, and the cost of not doing it is one mis-nested
    brace in a file whose strings contain no braces -- rare enough that the
    ``depth = 0`` reset handles it when it happens.
    """
    out: list[str] = []
    in_block = False
    for line in _source_lines(text):
        result: list[str] = []
        index = 0
        quote = ""
        while index < len(line):
            if in_block:
                end = line.find("*/", index)
                if end == -1:
                    result.append(" " * (len(line) - index))
                    index = len(line)
                else:
                    result.append(" " * (end + 2 - index))
                    index = end + 2
                    in_block = False
                continue
            pair = line[index : index + 2]
            if not quote and pair == "/*":
                in_block = True
                result.append("  ")
                index += 2
                continue
            if not quote and pair == "//":
                result.append(" " * (len(line) - index))
                index = len(line)
                continue
            char = line[index]
            result.append(char)
            if not quote and char in "\"'`":
                quote = char
            elif quote and char == quote and line[index - 1 : index] != "\\":
                quote = ""
            index += 1
        out.append("".join(result))
    return "\n".join(out)


# --------------------------------------------------------------------------
# Dispatch and costing
# --------------------------------------------------------------------------

_JS_SNIFF = re.compile(
    r"^\s*(?:export|import)\s|=>|^\s*(?:const|let|var)\s+\w+\s*=|function\s*\(",
    re.MULTILINE,
)


def outline_for(path: str, text: str, *, kind: str = "") -> list[Outline]:
    """Outline ``text``, choosing an extractor from ``path``'s extension.

    ``kind`` overrides the extension (``"python"``, ``"js"``, ``"other"``) for
    callers that already know the language -- a fenced block inside a markdown
    file, say. An unknown extension is not an error: it yields the placeholder,
    because a file ctxpack cannot outline is still a file worth listing.

    Raises :class:`CtxpackError` only for a ``kind`` this module does not
    implement, which is a caller bug rather than a user input problem.
    """
    text = text or ""
    ext = _ext(path)
    kind = kind or _kind_for(ext)
    if kind == "python":
        return python_outline(text, path=path)
    if kind == "js":
        return js_outline(text)
    if kind in ("other", "auto", ""):
        return unstructured(text, reason=f"no extractor for {ext or path}")
    raise CtxpackError(f"unknown outline kind {kind!r}; use python, js or other")


def _kind_for(ext: str) -> str:
    if ext in PYTHON_EXTS:
        return "python"
    if ext in JS_EXTS:
        return "js"
    return "other"


def _ext(path: str) -> str:
    base = path.rsplit("/", 1)[-1]
    dot = base.rfind(".")
    return base[dot:].lower() if dot > 0 else ""


def outline_text(path: str, text: str) -> str:
    """Rendered outline of one file, for callers that just want the string."""
    return render_outline(outline_for(path, text))


def outline_tokens(text: str, *, path: str = "") -> int:
    """Tokens the outline of ``text`` costs, priced by the real estimator.

    Not an estimate of an estimate: the outline is rendered and the resulting
    string is priced, so this number can be compared directly against what a
    packer would spend on the file. :func:`~ctxpack.tokens.estimate_tokens` is
    used rather than a ``Tokenizer`` on purpose -- a count that changed depending
    on whether tiktoken happened to be installed would make the compression ratio
    irreproducible between machines.

    Without ``path`` the language is sniffed, which is a guess; pass the path when
    you have it, because getting it wrong turns a 7x ratio into a 1x one.
    """
    kind = _kind_for(_ext(path)) if path else _sniff(text)
    return estimate_tokens(render_outline(outline_for(path, text, kind=kind)))


def _sniff(text: str) -> str:
    """Guess the language from markers only JS/TS has.

    Deliberately one-directional. A Python file containing the word ``export``
    in a docstring would be misfiled as JavaScript, whereas a JavaScript file
    that fails every pattern here would be misfiled as Python -- and the second
    mistake still produces an outline, because the patterns overlap enough.
    """
    return "js" if _JS_SNIFF.search(text or "") else "python"


def estimate_ratio(text: str, *, path: str = "") -> float:
    """How many times cheaper the outline is than the file it summarises.

    Original tokens divided by outline tokens. Above 1 means the outline paid
    for itself; on real source this lands between 5x and 20x, and below 3x is a
    warning sign that bodies or long values are leaking in. Returns ``0.0``
    rather than dividing by zero for an empty file, so a caller can report a
    ratio without a guard of its own.
    """
    text = text or ""
    if not text.strip():
        return 0.0
    outline = outline_tokens(text, path=path)
    if outline <= 0:
        return 0.0
    return estimate_tokens(text) / outline
