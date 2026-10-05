"""Token accounting: the heuristic, the exact path, and calibration."""

from __future__ import annotations

import pytest

from ctxpack.errors import CtxpackError
from ctxpack.tokens import (
    _CLASS_KEYS,
    DEFAULT_CALIBRATION,
    PARAMS,
    Tokenizer,
    calibration_factor,
    count_from_costs,
    estimate_tokens,
    load_calibration,
    mean_abs_pct_error,
    piece_costs,
    save_calibration,
)

#: File-sized samples, spanning the shapes ctxpack actually prices.
SAMPLES = [
    "",
    "def main() -> None:\n    print('hello')\n",
    "# Heading\n\nSome *markdown* prose with `code` and a [link](https://x.dev).\n",
    '{"a": 1, "b": [true, false, null], "c": "a string with spaces"}',
    "SELECT id, name FROM users WHERE created_at > '2024-01-01' ORDER BY name;",
    "https://example.com/some/really/long/path/that/keeps/going?q=1&r=2#frag",
    "\n".join(
        f"    indented_line_number_{i} = compute(value_{i}, other)"
        for i in range(200)
    ),
    "\n".join(
        f"class Widget{i}:\n    def render(self):\n        return f'<div />'"
        for i in range(60)
    ),
    "\n".join(f"def f{i}(a, b):\n    return {{'k': a, 'v': b}}" for i in range(40)),
]

#: Deliberately tiny strings, used for the absolute-error assertion rather than
#: the relative one -- see the note in ``count_from_costs``.
SHORT_SAMPLES = [
    "hello world",
    "x = 1",
    "# comment",
    "def f(): pass",
    "Ünïcödé",
]


def test_empty_text_is_zero_tokens():
    assert estimate_tokens("") == 0


def test_never_zero_for_non_empty():
    assert estimate_tokens("x") >= 1
    assert estimate_tokens(" ") >= 1


def test_monotonic_in_length():
    short = estimate_tokens("hello world")
    long = estimate_tokens("hello world " * 20)
    assert long > short


def test_calibration_scales_linearly():
    base = estimate_tokens("def main() -> None:\n    print('hi')\n")
    doubled = estimate_tokens(
        "def main() -> None:\n    print('hi')\n", calibration=2.0
    )
    assert doubled >= base * 1.9


def test_non_ascii_costs_more_than_ascii():
    ascii_text = "a" * 40
    cjk_text = "字" * 40
    assert estimate_tokens(cjk_text) > estimate_tokens(ascii_text)


def test_piece_costs_agree_with_estimate():
    text = "class A:\n    def b(self): return 1 + 2\n"
    costs = piece_costs(text)
    direct = count_from_costs(costs)
    assert int(direct) <= estimate_tokens(text, calibration=1.0)


def test_params_are_sane():
    """Every fitted constant must stay in a physically possible range."""
    for key, (ratio, prefix) in PARAMS.items():
        assert 0.3 <= ratio <= 20, f"{key} ratio {ratio} is not plausible"
        assert 0 <= prefix <= 8, f"{key} prefix {prefix} is not plausible"
        # A free prefix may exceed the ratio -- that just means short pieces are
        # cheap and long ones are not. What must not happen is a negative or
        # unbounded prefix.


def test_every_param_class_is_known():
    assert set(PARAMS) == set(_CLASS_KEYS)


@pytest.mark.parametrize("text", SAMPLES)
def test_heuristic_within_25_percent_of_tiktoken(text, tiktoken_available):
    """Accuracy on realistic, file-sized text.

    Very short strings are a known weak spot of a character-mass model -- see
    the note in ``count_from_costs`` -- so relative accuracy is asserted on
    inputs the size of files, which is what packing actually does.
    """
    if not tiktoken_available:
        pytest.skip("tiktoken is not installed")
    import tiktoken

    if not text:
        return
    enc = tiktoken.get_encoding("o200k_base")
    exact = len(enc.encode(text, disallowed_special=()))
    estimate = estimate_tokens(text)
    error = abs(estimate - exact) / exact
    assert error < 0.25, f"{error:.1%} off for {text[:40]!r}: {estimate} vs {exact}"


@pytest.mark.parametrize("text", SHORT_SAMPLES)
def test_absolute_error_on_short_strings_is_small(text, tiktoken_available):
    """Short strings may be proportionally off, but never much in absolute terms."""
    if not tiktoken_available:
        pytest.skip("tiktoken is not installed")
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
    exact = len(enc.encode(text, disallowed_special=()))
    estimate = estimate_tokens(text)
    assert abs(estimate - exact) <= 2, f"{text!r}: {estimate} vs {exact}"


def test_non_ascii_is_priced_separately():
    """Accented text must not be treated as cheap ASCII letters."""
    latin = estimate_tokens("Ünïcödé" * 10)
    ascii_equivalent = estimate_tokens("Unicode" * 10)
    assert latin > ascii_equivalent


def test_unknown_encoding_rejected():
    with pytest.raises(CtxpackError, match="unknown encoding"):
        Tokenizer("not-a-real-encoding")


def test_unknown_mode_rejected():
    with pytest.raises(CtxpackError, match="unknown token mode"):
        Tokenizer("o200k_base", mode="sometimes")


def test_exact_mode_requires_tiktoken(tiktoken_available):
    if tiktoken_available:
        assert Tokenizer("o200k_base", mode="exact").method == "exact"
    else:
        with pytest.raises(CtxpackError, match="tiktoken"):
            Tokenizer("o200k_base", mode="exact")


def test_never_mode_ignores_tiktoken(tiktoken_available):
    assert Tokenizer(mode="never").method == "estimate"


def test_count_metadata():
    tokenizer = Tokenizer(mode="never")
    result = tokenizer.count("hello world")
    assert result.chars == len("hello world")
    assert result.tokens > 0
    assert 0 < result.chars_per_token < 100
    assert not result.exact
    assert int(result) == result.tokens


def test_count_handles_special_token_text(tiktoken_available):
    """Source files sometimes contain <|endoftext|>; that must not raise."""
    tokenizer = Tokenizer(mode="auto")
    result = tokenizer.count("before <|endoftext|> after")
    assert result.tokens > 0


# -- calibration -------------------------------------------------------------


def test_calibration_factor_is_least_squares():
    pairs = [(10, 20), (20, 40), (30, 60)]
    assert calibration_factor(pairs) == pytest.approx(2.0)


def test_calibration_factor_handles_zero_input():
    assert calibration_factor([]) == 1.0
    assert calibration_factor([(0, 0)]) == 1.0


def test_mean_abs_pct_error():
    assert mean_abs_pct_error([(100, 100)]) == 0.0
    assert mean_abs_pct_error([(110, 100), (90, 100)]) == pytest.approx(0.1)


def test_scaling_reduces_error():
    pairs = [(100, 110), (200, 220), (50, 55)]
    factor = calibration_factor(pairs)
    scaled = [(round(h * factor), e) for h, e in pairs]
    assert mean_abs_pct_error(scaled) < mean_abs_pct_error(pairs)


def test_calibration_roundtrip(tmp_path, monkeypatch):
    path = tmp_path / "calibration.json"
    save_calibration(1.25, samples=10, mean_abs_pct=0.04, encoding="o200k_base", path=path)
    assert path.exists()

    monkeypatch.setenv("CTXPACK_CALIBRATION", str(path))
    assert load_calibration() == pytest.approx(1.25)


def test_missing_calibration_file_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("CTXPACK_CALIBRATION", str(tmp_path / "nope.json"))
    assert load_calibration() == DEFAULT_CALIBRATION


def test_corrupt_calibration_file_falls_back(tmp_path, monkeypatch):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("CTXPACK_CALIBRATION", str(path))
    assert load_calibration() == DEFAULT_CALIBRATION


def test_calibration_applies_to_tokenizer(tmp_path, monkeypatch):
    path = tmp_path / "cal.json"
    save_calibration(2.0, path=path)
    monkeypatch.setenv("CTXPACK_CALIBRATION", str(path))
    tokenizer = Tokenizer(mode="never")
    assert tokenizer.calibration == pytest.approx(2.0)
