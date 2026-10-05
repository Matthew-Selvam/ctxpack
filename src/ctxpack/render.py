"""Turning a :class:`~ctxpack.pack.PackResult` into something an LLM can read.

Four formats, because the consumer varies: ``markdown`` for pasting into a chat,
``xml`` for structure-sensitive models, ``json`` for programmatic use, and
``tree`` when you want the map of a repo at the cheapest possible price.

The fiddly parts of this module are all about not corrupting content. Source
files contain triple backticks, ``]]>``, ``&``, and stray control characters
copied out of terminal logs; all of them will silently break a naive renderer,
so fences escalate, CDATA sections get split, attributes are escaped with the
stdlib, and characters XML cannot represent are dropped -- see ``_xml_safe``,
which exists because CDATA is not an escape hatch for them.

Every format here is verified to parse against hostile payloads by
``scripts/bench.py``, and by tests that feed the renderers exactly this
material.
"""

from __future__ import annotations

import json
import re
from xml.sax.saxutils import escape as xml_escape
from xml.sax.saxutils import quoteattr

from .errors import CtxpackError
from .pack import Document, PackResult

__all__ = ["FORMATS", "fence", "language_for", "render"]

FORMATS = ("markdown", "xml", "json", "tree")

#: Extension to markdown fence language, for syntax highlighting.
LANGUAGES: dict[str, str] = {
    ".py": "python", ".pyi": "python", ".ts": "typescript", ".tsx": "tsx",
    ".js": "javascript", ".jsx": "jsx", ".mjs": "javascript", ".cjs": "javascript",
    ".go": "go", ".rs": "rust", ".java": "java", ".kt": "kotlin", ".rb": "ruby",
    ".php": "php", ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp",
    ".cxx": "cpp", ".hpp": "cpp", ".cs": "csharp", ".swift": "swift",
    ".scala": "scala", ".sql": "sql", ".sh": "shell", ".bash": "shell",
    ".zsh": "shell", ".lua": "lua", ".dart": "dart", ".ex": "elixir",
    ".exs": "elixir", ".graphql": "graphql", ".gql": "graphql",
    ".proto": "protobuf", ".vue": "vue", ".svelte": "svelte",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".xml": "xml", ".html": "html", ".css": "css", ".scss": "scss",
    ".md": "markdown", ".mdx": "markdown", ".rst": "rst", ".diff": "diff",
    ".patch": "diff", ".tf": "hcl", ".zig": "zig", ".hs": "haskell",
}

_FENCE_RE = re.compile(r"`{3,}", re.MULTILINE)

#: Printed once, above the hoisted blocks. A factored bundle is a reading aid,
#: not a compilable artefact: the files carry markers in place of shared text.
#: Saying so here is cheaper than having a reader wonder why the code no longer
#: parses.
_SHARED_WARNING = (
    "> The files below have had repeated blocks replaced with references to this\n"
    "> section, to spend fewer tokens on text that appeared many times over.\n"
    "> The result is for reading, not for compiling."
)


def language_for(path: str) -> str:
    dot = path.rfind(".")
    if dot < 0:
        return ""
    return LANGUAGES.get(path[dot:].lower(), "")


def fence(text: str, language: str = "") -> str:
    """Wrap ``text`` in a backtick fence long enough to survive its content.

    A markdown file containing ``` would otherwise end the block early and
    spill the rest of the bundle into prose.
    """
    longest = 0
    for match in _FENCE_RE.finditer(text):
        longest = max(longest, len(match.group(0)))
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{language}\n{text}\n{ticks}"


def _xml_safe(text: str) -> str:
    """Remove characters XML 1.0 cannot represent, at any nesting depth.

    CDATA is not an escape hatch for these: XML 1.0 restricts the character set
    itself, so a file containing a stray ``\\x01`` -- a control code from a
    binary-ish log, a terminal escape pasted into a comment -- makes the whole
    document unparseable even though it is "inside" a CDATA section.

    Removed rather than replaced: these characters are invisible in every
    renderer, so keeping a visible placeholder would corrupt the source text
    the model is trying to read, while dropping them loses nothing a reader
    could have seen. Lone surrogates are handled the same way, since they
    cannot be UTF-8 encoded at all.
    """
    return "".join(
        char
        for char in text
        if (
            # Tab, LF, CR, and the printable ranges.
            char in "\t\n\r"
            or "\x20" <= char <= "\ud7ff"
            or "\ue000" <= char <= "\ufffd"
            or "\U00010000" <= char <= "\U0010ffff"
        )
        and not ("\ud800" <= char <= "\udfff")
    )


def _cdata(text: str) -> str:
    """Wrap in CDATA, splitting any ``]]>`` the content contains."""
    text = _xml_safe(text)
    if "]]>" in text:
        text = text.replace("]]>", "]]]]><![CDATA[>")
    return f"<![CDATA[{text}]]>"


def _human(n: int) -> str:
    return f"{n:,}"


def _short_path(result: PackResult) -> str:
    return str(result.root)


def _stats_line(result: PackResult) -> str:
    method = "exact" if result.method == "exact" else "estimate"
    return (
        f"{_human(result.budget)} tokens budget · {_human(len(result.documents))} "
        f"of {_human(result.discovered)} files · {result.mode} · "
        f"{result.encoding} ({method})"
    )


def _outcome(result: PackResult) -> list[str]:
    """Honest one-liners about what was dropped, collapsed or cut."""
    out: list[str]
    out = [
        f"index: {_human(len(result.manifest))} entries, "
        f"{_human(result.manifest_tokens)} tokens"
    ]
    if result.duplicates:
        worst = max(d.similarity for d in result.duplicates)
        out.append(
            f"collapsed {_human(len(result.duplicates))} near-duplicate files "
            f"(up to {worst:.0%} similar)"
        )
    truncated = result.truncated
    if truncated:
        saved = sum(d.saved for d in truncated)
        out.append(
            f"truncated {_human(len(truncated))} file(s), saving ~{_human(saved)} tokens"
        )
    return out


def _label(path: str) -> str:
    """Make a path safe to drop into a one-line markdown construct.

    Filenames can contain newlines on macOS and Linux, and a raw one inside a
    heading silently restructures the document: the heading ends early and the
    remainder of the path becomes body text. Escaped rather than stripped, so the
    reader can still tell what the file is called.
    """
    out = path.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n")
    return out.replace("`", "'")


def _document_block(doc: Document, index: int) -> str:
    language = language_for(doc.path)
    body = fence(doc.text.rstrip("\n"), language)
    flag = " *(truncated)*" if doc.truncated else ""
    return (
        f"### {index}. `{_label(doc.path)}`{flag}\n\n"
        f"{_human(doc.tokens)} tokens · {_human(doc.lines)} lines\n\n"
        f"{body}\n"
    )


def _render_markdown(result: PackResult) -> str:
    parts: list[str] = [
        "# ctxpack bundle\n",
        f"`{_short_path(result)}`\n",
        f"{_stats_line(result)}\n",
    ]
    parts.extend(f"- {line}" for line in _outcome(result))
    if result.notes:
        parts.append("")
        parts.extend(f"- note: {note}" for note in result.notes)
    parts.append("")

    if result.manifest_text:
        parts.append("## Index\n")
        # Through `fence()`, not a hardcoded ``` pair. The index interpolates
        # file paths, and a path containing a newline plus backticks would emit
        # a line-leading fence that closes this block early and reopens a new one,
        # swallowing the `## Files` heading into a code block. A filename really
        # can contain a newline on macOS and Linux.
        parts.append(fence(result.manifest_text))

    if result.documents:
        parts.append("## Files\n")
        parts.extend(
            _document_block(doc, index)
            for index, doc in enumerate(result.documents, start=1)
        )

    if result.shared_blocks:
        parts.append("## Shared blocks\n")
        parts.append(_SHARED_WARNING)
        parts.append("")
        for block in result.shared_blocks:
            parts.append(f"### {block.id} — {block.lines} lines\n")
            parts.append(fence(block.text.rstrip("\n"), ""))
            parts.append("")

    return "\n".join(parts)


def _render_tree(result: PackResult) -> str:
    parts = [
        f"root: {_short_path(result)}",
        _stats_line(result),
        *_outcome(result),
    ]
    if result.shared_blocks:
        parts.append(
            f"shared blocks: {len(result.shared_blocks)} "
            f"(bodies hoisted; see the markdown bundle)"
        )
    if result.manifest_text:
        parts.append("")
        parts.append(result.manifest_text)
    return "\n".join(parts) + "\n"


def _attr(value: str) -> str:
    """Quote an XML attribute value, after making it representable at all."""
    return quoteattr(_xml_safe(value))


def _render_xml(result: PackResult) -> str:
    root = _attr(str(result.root))
    out = ['<?xml version="1.0" encoding="UTF-8"?>']
    out.append(
        "<ctxpack"
        f" root={root}"
        f' budget="{result.budget}"'
        f' used="{result.accounted}"'
        f' files="{len(result.documents)}"'
        f' discovered="{result.discovered}"'
        f' encoding={_attr(result.encoding)}'
        f' method={_attr(result.method)}'
        f' mode={_attr(result.mode)}'
        ">"
    )
    out.append(
        f'  <summary>{xml_escape("; ".join(_outcome(result)))}</summary>'
    )
    if result.manifest_text:
        out.append("  <index>")
        for entry in result.manifest:
            note = ' estimated="true"' if entry.estimated else ""
            out.append(
                f'    <entry path={_attr(entry.path)}'
                f' tokens="{entry.tokens}" bytes="{entry.size}"{note}/>'
            )
        out.append("  </index>")
    out.append("  <files>")
    for doc in result.documents:
        attrs = (
            f'    <file path={_attr(doc.path)}'
            f' tokens="{doc.tokens}" lines="{doc.lines}"'
            f' language={_attr(language_for(doc.path))}'
            + (' truncated="true"' if doc.truncated else "")
            + ">"
        )
        out.append(attrs)
        out.append(_cdata(doc.text))
        out.append("    </file>")
    out.append("  </files>")
    if result.shared_blocks:
        out.append("  <shared>")
        for block in result.shared_blocks:
            out.append(
                f'    <block id={_attr(block.id)} lines="{block.lines}"'
                f' occurrences="{block.occurrences}">'
            )
            out.append(_cdata(block.text))
            out.append("    </block>")
        out.append("  </shared>")
    out.append("</ctxpack>")
    return "\n".join(out) + "\n"


def _render_json(result: PackResult) -> str:
    payload = {
        "root": str(result.root),
        "budget": result.budget,
        "accounted": result.accounted,
        "encoding": result.encoding,
        "method": result.method,
        "calibration": result.calibration,
        "mode": result.mode,
        "discovered": result.discovered,
        "scanned": result.scanned,
        "unscanned": result.unscanned,
        "ignored": result.ignored,
        "summary": _outcome(result),
        "notes": result.notes,
        "index": [
            {
                "path": e.path,
                "tokens": e.tokens,
                "bytes": e.size,
                "estimated": e.estimated,
            }
            for e in result.manifest
        ],
        "shared_blocks": [
            {
                "id": block.id,
                "lines": block.lines,
                "occurrences": block.occurrences,
                "text": block.text,
            }
            for block in result.shared_blocks
        ],
        "files": [
            {
                "path": doc.path,
                "tokens": doc.tokens,
                "lines": doc.lines,
                "bytes": doc.candidate.size,
                "language": language_for(doc.path),
                "truncated": doc.truncated,
                "original_tokens": doc.original_tokens,
                "score": doc.scored.score,
                "content": doc.text,
            }
            for doc in result.documents
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


_RENDERERS = {
    "markdown": _render_markdown,
    "xml": _render_xml,
    "json": _render_json,
    "tree": _render_tree,
}


def render(result: PackResult, fmt: str = "markdown") -> str:
    """Render a pack result. Raises :class:`CtxpackError` on unknown formats."""
    try:
        renderer = _RENDERERS[fmt]
    except KeyError:
        raise CtxpackError(
            f"unknown format {fmt!r}; choose one of " + ", ".join(FORMATS)
        ) from None
    return renderer(result)
