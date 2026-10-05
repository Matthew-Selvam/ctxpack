"""Shared fixtures.

Tests run against ``src/`` without requiring an install, so ``pytest`` works
in a fresh checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ctxpack.tokens import Tokenizer  # noqa: E402


@pytest.fixture(scope="session")
def heuristic() -> Tokenizer:
    """Deterministic counting: no calibration, no tiktoken."""
    return Tokenizer(mode="never", calibration=1.0)


@pytest.fixture
def tiktoken_available() -> bool:
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        return False
    return True


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "needs_tiktoken: test needs tiktoken to compare against"
    )


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A small repository with the shapes that make packing hard.

    Includes a README, a nested source tree, a changelog, a binary file, a
    lockfile, an ignored build directory, near-duplicate files and one file
    large enough to force truncation.
    """
    root = tmp_path / "project"
    (root / "src" / "app").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "build").mkdir()
    (root / "locales").mkdir()

    (root / "README.md").write_text(
        "# Sample\n\nA sample project used by the ctxpack tests.\n", encoding="utf-8"
    )
    (root / "CHANGELOG.md").write_text(
        "# Changelog\n\n" + "\n".join(f"- {i}. did a thing" for i in range(60)),
        encoding="utf-8",
    )
    (root / "package-lock.json").write_text(
        '{"lockfileVersion": 3, "packages": {}}\n', encoding="utf-8"
    )
    (root / ".gitignore").write_text("build/\n*.log\n!keep.log\n", encoding="utf-8")
    (root / "debug.log").write_text("noise\n", encoding="utf-8")
    (root / "keep.log").write_text("noise\n", encoding="utf-8")
    (root / "build" / "out.js").write_text("var x = 1;\n", encoding="utf-8")

    (root / "src" / "app" / "main.py").write_text(
        "def main() -> None:\n    print('hello')\n", encoding="utf-8"
    )
    (root / "src" / "app" / "index.ts").write_text(
        "export const version = '1.0.0';\n", encoding="utf-8"
    )
    (root / "src" / "app" / "types.py").write_text(
        "from dataclasses import dataclass\n\n\n@dataclass\nclass Config:\n    name: str\n",
        encoding="utf-8",
    )
    (root / "docs" / "guide.md").write_text(
        "# Guide\n\n" + "\n".join(f"Step {i}: do the thing properly." for i in range(40)),
        encoding="utf-8",
    )

    # Big enough to be cut at a low budget.
    (root / "src" / "app" / "big.py").write_text(
        "\n".join(
            f"def function_number_{i}(argument_one, argument_two):\n"
            f"    return argument_one + argument_two + {i}\n"
            for i in range(400)
        ),
        encoding="utf-8",
    )

    # Two near-identical files in different directories: the duplicate case.
    body = "\n".join(
        f"    this is a distinctive line of source code number {i} with padding"
        for i in range(40)
    )
    (root / "locales" / "en.py").write_text(
        "MESSAGES = {\n" + body + "\n}\n", encoding="utf-8"
    )
    (root / "locales" / "fr.py").write_text(
        "MESSAGES = {\n" + body + "\n}\n", encoding="utf-8"
    )

    # Binary, must be skipped.
    (root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

    return root


@pytest.fixture
def nested_project(tmp_path: Path) -> Path:
    """A tree where each directory carries its own .gitignore."""
    root = tmp_path / "nested"
    (root / "a" / "deep").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    (root / "a" / "deep" / ".gitignore").write_text("secret.md\n", encoding="utf-8")
    (root / "a" / "deep" / "secret.md").write_text("hidden\n", encoding="utf-8")
    (root / "a" / "deep" / "visible.md").write_text("shown\n", encoding="utf-8")
    (root / "a" / "top.md").write_text("top level\n", encoding="utf-8")
    (root / "b" / "thing.md").write_text("thing\n", encoding="utf-8")
    return root
