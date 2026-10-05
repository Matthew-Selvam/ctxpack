"""Turning a :class:`~ctxpack.pack.PackResult` into something an LLM can read.

Four formats, because the consumer varies: ``markdown`` for pasting into a chat,
``xml`` for structure-sensitive models, ``json`` for programmatic use, and
``tree`` when you want the map of a repo at the cheapest possible price.

The fiddly parts of this module are all about not corrupting content. Source
files contain triple backticks, ``]]>``, and ``&``; all three will silently
break a naive renderer, so fences escalate, CDATA sections get split, and
attributes are escaped with the stdlib.
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


def _cdata(text: str) -> str:
    """Wrap in CDATA, splitting any ``]]>`` the content contains."""
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


def _document_block(doc: Document, index: int) -> str:
    language = language_for(doc.path)
    body = fence(doc.text.rstrip("\n"), language)
    flag = " *(truncated)*" if doc.truncated else ""
    return (
        f"### {index}. `{doc.path}`{flag}\n\n"
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
        parts.append("```")
        parts.append(result.manifest_text)
        parts.append("```\n")

    if result.documents:
        parts.append("## Files\n")
        parts.extend(
            _document_block(doc, index)
            for index, doc in enumerate(result.documents, start=1)
        )

    return "\n".join(parts)


def _render_tree(result: PackResult) -> str:
    parts = [
        f"root: {_short_path(result)}",
        _stats_line(result),
        *_outcome(result),
    ]
    if result.manifest_text:
        parts.append("")
        parts.append(result.manifest_text)
    return "\n".join(parts) + "\n"


def _render_xml(result: PackResult) -> str:
    root = quoteattr(str(result.root))
    out = ['<?xml version="1.0" encoding="UTF-8"?>']
    out.append(
        "<ctxpack"
        f" root={root}"
        f' budget="{result.budget}"'
        f' used="{result.accounted}"'
        f' files="{len(result.documents)}"'
        f' discovered="{result.discovered}"'
        f' encoding={quoteattr(result.encoding)}'
        f' method={quoteattr(result.method)}'
        f' mode={quoteattr(result.mode)}'
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
                f'    <entry path={quoteattr(entry.path)}'
                f' tokens="{entry.tokens}" bytes="{entry.size}"{note}/>'
            )
        out.append("  </index>")
    out.append("  <files>")
    for doc in result.documents:
        attrs = (
            f'    <file path={quoteattr(doc.path)}'
            f' tokens="{doc.tokens}" lines="{doc.lines}"'
            f' language={quoteattr(language_for(doc.path))}'
            + (' truncated="true"' if doc.truncated else "")
            + ">"
        )
        out.append(attrs)
        out.append(_cdata(doc.text))
        out.append("    </file>")
    out.append("  </files>")
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
