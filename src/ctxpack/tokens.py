r"""Token accounting.

ctxpack needs to answer "how many tokens is this?" constantly, and it needs to
answer it in environments where nothing is installed -- a CI container, a
sandboxed agent runtime, an air-gapped box. So there are two paths.

``exact``
    Hand the text to :mod:`tiktoken` when it is importable. Ground truth for
    the OpenAI BPE families, which is still what nearly every model on the
    market uses or clones.

``estimate``
    A dependency-free approximation: a splitter that mirrors the *shape* of the
    cl100k/o200k split patterns, a per-piece-class cost table, and one global
    calibration factor.

The estimate is consistently **high** on real-world text -- an early version
overshot by 37% at the median, wasting a third of the budget on nothing.
Pretending a fixed accuracy would be dishonest, so ctxpack ships the tooling
instead: the seven constants in :data:`PARAMS` were fitted against 895 real
files from nine languages (see ``scripts/fit_params.py``), and ``ctxpack
calibrate`` re-derives a single scalar multiplier on whatever corpus it is
pointed at. See :func:`calibration_factor` and :func:`mean_abs_pct_error`.

The splitter below is a transcription of the regex in ``tiktoken``'s
``_educational.py``, re-expressed with ``[^\W\d_]`` for ``\p{L}`` because the
stdlib ``re`` module has no Unicode property escapes. The difference is
immaterial for token *counts``.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .errors import CtxpackError

__all__ = [
    "DEFAULT_CALIBRATION",
    "KNOWN_ENCODINGS",
    "PARAMS",
    "Count",
    "Tokenizer",
    "calibration_factor",
    "count_from_costs",
    "estimate_tokens",
    "load_calibration",
    "mean_abs_pct_error",
    "piece_costs",
    "save_calibration",
]

#: Encodings ctxpack knows how to name. Actual encoding only matters on the
#: exact path; the heuristic is encoding-agnostic by construction.
KNOWN_ENCODINGS: tuple[str, ...] = (
    "o200k_base",
    "cl100k_base",
    "p50k_base",
    "r50k_base",
    "gpt2",
)

#: Environment variable pointing at an explicit calibration file.
CALIBRATION_ENV = "CTXPACK_CALIBRATION"

#: Multiplier applied to every heuristic count. See module docstring.
#: ``ctxpack calibrate --write`` rewrites this on your machine.
DEFAULT_CALIBRATION = 1.0

# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

_SPLIT_RE = re.compile(
    # English contractions stay glued to the following text, as in cl100k.
    r"'(?:[sdmt]|ll|ve|re)"
    # An optional leading space is absorbed by the word that follows it.
    r"| ?[^\W\d_]+"
    r"| ?\d+"
    r"| ?[^\s\w]+"
    r"|_+"
    r"|\s+",
    re.UNICODE,
)

_RE_LETTERS = re.compile(r"^ ?[^\W\d_]+$")
_RE_DIGITS = re.compile(r"^ ?\d+$")
_RE_PUNCT = re.compile(r"^ ?[^\s\w]+$")
_RE_UNDERSCORE = re.compile(r"^_+$")
_RE_WS = re.compile(r"^\s+$")

#: Fitted cost model. Keys are piece classes; values are ``(chars_per_token,
#: free_prefix)``. ``free_prefix`` is the number of characters each piece gets
#: for free before the ratio starts charging -- for letters that is the
#: always-single-token short-word case.
#:
#: Fitted by ``scripts/fit_params.py`` against tiktoken 0.14 ``o200k_base``.
PARAMS: dict[str, tuple[float, float]] = {
    "letters": (3.615, 4.0),
    "digits": (3.0, 3.0),
    "punct": (1.215, 2.0),
    "ws": (4.626, 3.005),
    "mixed": (3.5, 0.0),
    "underscore": (3.0, 0.0),
    "nonascii": (1.727, 0.0),
}

# Classes are integers so the hot loop can use a list rather than a dict.
_LETTERS, _DIGITS, _PUNCT, _WS, _MIXED, _UNDERSCORE, _NONASCII = range(7)
_CLASS_KEYS = (
    "letters",
    "digits",
    "punct",
    "ws",
    "mixed",
    "underscore",
    "nonascii",
)
CLASS_LETTERS = _LETTERS
CLASS_DIGITS = _DIGITS
CLASS_PUNCT = _PUNCT
CLASS_WS = _WS
CLASS_MIXED = _MIXED
CLASS_UNDERSCORE = _UNDERSCORE
CLASS_NONASCII = _NONASCII


def _classify(piece: str) -> int:
    # Non-ASCII is checked first and wins outright. An accented Latin word is
    # rare enough in the training corpus that BPE shreds it: "Ünïcödé" costs
    # real tokens, but pricing it as ordinary letters makes it look free.
    # Emoji and CJK land here too. `str.isascii` is a C-level scan, so this is
    # cheaper than it looks.
    if not piece.isascii():
        return _NONASCII
    if _RE_UNDERSCORE.match(piece):
        return _UNDERSCORE
    if _RE_LETTERS.match(piece):
        return _LETTERS
    if _RE_DIGITS.match(piece):
        return _DIGITS
    if _RE_PUNCT.match(piece):
        return _PUNCT
    if _RE_WS.match(piece):
        return _WS
    return _MIXED


def piece_costs(text: str) -> list[tuple[int, int]]:
    """Decompose ``text`` into ``(class, length)`` pairs.

    Exposed separately from :func:`estimate_tokens` so the fitting script can
    tokenise a corpus once and then evaluate candidate cost models cheaply.
    """
    return [(_classify(p), len(p)) for p in _SPLIT_RE.findall(text)]


def count_from_costs(
    costs: list[tuple[int, int]],
    params: dict[str, tuple[float, float]] | None = None,
) -> float:
    """Total token cost for pre-classified pieces. Returns a float.

    Rounding happens once, at the very end, rather than per piece -- an early
    version rounded every piece and systematically overshot.
    """
    table = params or PARAMS
    ratios = [table[k][0] for k in _CLASS_KEYS]
    prefixes = [table[k][1] for k in _CLASS_KEYS]
    total = 0.0
    for cls, n in costs:
        span = n - prefixes[cls]
        # No per-piece floor, deliberately.
        #
        # It is tempting to charge every piece at least one token, since BPE
        # cannot encode "hello" as zero tokens. That is physically true and
        # measurably wrong: the cl100k/o200k split pattern *over-segments*
        # relative to real BPE merges, because adjacent pieces get merged back
        # together. Punctuation-dense code splits into more pieces than it has
        # tokens -- a small Python method yields 22 pre-tokens and 16 real
        # ones.
        #
        # With a floor every cost is pinned at or above the piece count, so the
        # model can never come back down to the true value. Measured holdout
        # error went from 8% to 18%, with every fitted ratio jammed against its
        # upper bound trying to compensate.
        #
        # So this is not a piece-counting model at all. It is a per-class
        # characters-per-token estimator whose free prefix absorbs short
        # pieces, which is what the data actually supports. The cost is worse
        # *relative* accuracy on very short strings ("hello world" scores 1
        # against a true 2); the absolute error stays near one token, which is
        # irrelevant when pricing files of thousands.
        total += 1.0 if span <= 0 else span / ratios[cls]
    return total


def estimate_tokens(
    text: str,
    *,
    calibration: float = 1.0,
    params: dict[str, tuple[float, float]] | None = None,
) -> int:
    """Approximate the token count of ``text`` with no dependencies."""
    if not text:
        return 0
    total = count_from_costs(piece_costs(text), params)
    if calibration != 1.0:
        total *= calibration
    return max(1, math.ceil(total))


# --------------------------------------------------------------------------
# Exact path
# --------------------------------------------------------------------------


def _load_tiktoken(encoding: str):
    try:
        import tiktoken  # type: ignore import-not-found
    except Exception:
        return None
    try:
        return tiktoken.get_encoding(encoding)
    except Exception:
        return None


@dataclass(frozen=True)
class Count:
    """A token count plus provenance, so callers can tell how much to trust it."""

    tokens: int
    chars: int
    method: str  # "exact" | "estimate"
    encoding: str

    @property
    def exact(self) -> bool:
        return self.method == "exact"

    @property
    def chars_per_token(self) -> float:
        if not self.tokens:
            return 0.0
        return self.chars / self.tokens

    def __int__(self) -> int:
        return self.tokens


class Tokenizer:
    """Counts tokens, preferring ground truth and degrading gracefully.

    ``mode`` is one of:

    ``auto``
        Use :mod:`tiktoken` when it is installed, else the heuristic.
    ``exact``
        Require :mod:`tiktoken`; raise if it is missing.
    ``never``
        Always use the heuristic. Useful for benchmarking the heuristic.
    """

    def __init__(
        self,
        encoding: str = "o200k_base",
        *,
        mode: str = "auto",
        calibration: float | None = None,
    ) -> None:
        if encoding not in KNOWN_ENCODINGS:
            raise CtxpackError(
                f"unknown encoding {encoding!r}; choose one of "
                + ", ".join(KNOWN_ENCODINGS)
            )
        if mode not in ("auto", "exact", "never"):
            raise CtxpackError(f"unknown token mode {mode!r}; use auto, exact or never")

        self.encoding = encoding
        self.mode = mode
        self.calibration = (
            load_calibration() if calibration is None else float(calibration)
        )
        self._impl = None
        self._resolved = False

        if mode == "exact" and not self._tiktoken_available():
            raise CtxpackError(
                "exact token counting requested but tiktoken is not installed; "
                "run `pip install 'ctxpack[exact]'` or pass --exact never"
            )

    # -- internals ---------------------------------------------------------

    def _tiktoken_available(self) -> bool:
        return _load_tiktoken(self.encoding) is not None

    def _resolve(self) -> None:
        if self._resolved:
            return
        self._resolved = True
        self._impl = None if self.mode == "never" else _load_tiktoken(self.encoding)

    @property
    def method(self) -> str:
        """``"exact"`` or ``"estimate"``, decided once and cached."""
        self._resolve()
        return "exact" if self._impl is not None else "estimate"

    @property
    def exact(self) -> bool:
        return self.method == "exact"

    # -- public API --------------------------------------------------------

    def count(self, text: str) -> Count:
        self._resolve()
        chars = len(text)
        if self._impl is not None:
            # ``disallowed_special=()`` keeps source files that happen to
            # contain ``<|endoftext|>`` from raising instead of counting.
            n = len(self._impl.encode(text, disallowed_special=()))
            return Count(n, chars, "exact", self.encoding)
        return Count(
            estimate_tokens(text, calibration=self.calibration),
            chars,
            "estimate",
            self.encoding,
        )

    def count_many(self, texts: Iterable[str]) -> list[Count]:
        return [self.count(t) for t in texts]


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def calibration_factor(pairs: Sequence[tuple[int, int]]) -> float:
    """Least-squares multiplier mapping heuristic counts onto exact counts.

    ``pairs`` is a sequence of ``(heuristic, exact)`` counts. Minimising
    ``sum((f*h - e)**2)`` gives ``f = sum(h*e) / sum(h*h)``, which is the
    right estimator to use because it minimises total squared error rather
    than letting a handful of enormous files dominate the mean.
    """
    num = sum(h * e for h, e in pairs)
    den = sum(h * h for h, e in pairs)
    if den == 0:
        return 1.0
    return num / den


def mean_abs_pct_error(pairs: Sequence[tuple[int, int]]) -> float:
    """Mean absolute percentage error of the *raw* heuristic against exact."""
    if not pairs:
        return 0.0
    total = 0.0
    for h, e in pairs:
        if e:
            total += abs(h - e) / e
    return total / len(pairs)


def calibration_path() -> Path:
    """Where the calibration file lives, honouring ``CTXPACK_CALIBRATION``."""
    override = os.environ.get(CALIBRATION_ENV)
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base).expanduser() / "ctxpack" / "calibration.json"


def load_calibration() -> float:
    """Read the persisted calibration factor, or fall back to the default."""
    path = calibration_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return DEFAULT_CALIBRATION
    factor = payload.get("factor")
    if isinstance(factor, (int, float)) and factor > 0:
        return float(factor)
    return DEFAULT_CALIBRATION


def save_calibration(
    factor: float,
    *,
    samples: int = 0,
    mean_abs_pct: float = 0.0,
    encoding: str = "",
    path: Path | None = None,
) -> Path:
    """Persist ``factor`` so later runs are calibrated out of the box."""
    target = path or calibration_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "factor": round(factor, 6),
                "samples": samples,
                "mean_abs_pct_error": round(mean_abs_pct, 6),
                "encoding": encoding,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return target
