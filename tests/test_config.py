"""Repository config files: discovery, validation, and precedence.

The precedence tests are the reason this file is longer than the feature looks:
argparse fills defaults for every flag, so the interesting question is never
"what is the value" but "did the user ask for this at all".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from ctxpack import config as cfg
from ctxpack.config import (
    UNSET,
    Config,
    describe,
    discover_config,
    load_config,
    load_or_empty,
    merge,
    passed_dests,
    resolve_patterns,
    unknown_keys_warning,
)
from ctxpack.errors import CtxpackError

toml = pytest.importorskip("tomllib", reason="TOML config tests need 3.11+")


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# -- discovery ---------------------------------------------------------------


def test_discover_finds_in_start_directory(tmp_path: Path):
    write(tmp_path / "ctxpack.toml", "budget = 1\n")
    assert discover_config(tmp_path) == tmp_path / "ctxpack.toml"


def test_discover_walks_upward(tmp_path: Path):
    write(tmp_path / "ctxpack.toml", "budget = 1\n")
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    assert discover_config(deep) == tmp_path / "ctxpack.toml"


def test_discover_from_a_file_uses_its_directory(tmp_path: Path):
    write(tmp_path / "ctxpack.toml", "budget = 1\n")
    (tmp_path / "src").mkdir()
    target = write(tmp_path / "src" / "main.py", "x = 1\n")
    assert discover_config(target) == tmp_path / "ctxpack.toml"


def test_discover_prefers_the_nearest_file(tmp_path: Path):
    write(tmp_path / "ctxpack.toml", "budget = 1\n")
    inner = tmp_path / "pkg"
    inner.mkdir()
    write(inner / ".ctxpack.toml", "budget = 2\n")
    assert discover_config(inner) == inner / ".ctxpack.toml"


def test_discover_stops_at_the_git_root(tmp_path: Path):
    """The important one: a config *above* the work tree must not be found.

    Otherwise the same command packs differently depending on where the
    checkout lives, including via a file in $HOME that no commit mentions.
    """
    write(tmp_path / "ctxpack.toml", "budget = 1\n")
    (tmp_path / ".git").mkdir()
    # The repo root is *below* the config, so the walk hits .git first.
    repo = tmp_path / "work"
    (repo / ".git").mkdir(parents=True)
    (repo / "src").mkdir()
    assert discover_config(repo / "src") is None


def test_discover_checks_the_stopping_directory_first(tmp_path: Path):
    """.git ends the walk, but a config in that same directory still counts."""
    (tmp_path / ".git").mkdir()
    write(tmp_path / "ctxpack.toml", "budget = 1\n")
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    assert discover_config(repo / "src") == tmp_path / "ctxpack.toml"


def test_discover_returns_none_when_absent(tmp_path: Path):
    # A .git keeps the search from escaping tmp_path, which is not something a
    # test should depend on.
    (tmp_path / ".git").mkdir()
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    assert discover_config(deep) is None


def test_discover_is_cwd_independent(tmp_path: Path, monkeypatch):
    write(tmp_path / "ctxpack.toml", "budget = 1\n")
    deep = tmp_path / "a"
    deep.mkdir()
    monkeypatch.chdir(tmp_path)
    assert discover_config(Path("a")) == tmp_path / "ctxpack.toml"


# -- parsing -----------------------------------------------------------------


def test_full_config_parses(tmp_path: Path):
    path = write(
        tmp_path / "ctxpack.toml",
        """
# Recorded so the reasoning survives the person who set it.
budget = 60000
format = "xml"
mode = "coverage"
manifest = "paths"
encoding = "cl100k_base"
exact = "never"
reach_weight = 4.0
outline = true
factor_shared = true
diff = "main...HEAD"
per_dir_frac = 0.5
dedupe_threshold = 0.9
exclude = ["build/", "**/*.min.js"]
include = ["src/**"]
""",
    )
    loaded = load_config(path)
    assert loaded.budget == 60000
    assert loaded.format == "xml"
    assert loaded.mode == "coverage"
    assert loaded.manifest == "paths"
    assert loaded.encoding == "cl100k_base"
    assert loaded.exact == "never"
    assert loaded.reach_weight == 4.0
    assert loaded.outline is True
    assert loaded.factor_shared is True
    assert loaded.diff == "main...HEAD"
    assert loaded.per_dir_frac == 0.5
    assert loaded.dedupe_threshold == 0.9
    assert loaded.exclude == ("build/", "**/*.min.js")
    assert loaded.include == ("src/**",)
    assert loaded.source == path
    assert loaded.unknown_keys == ()
    assert loaded.base == tmp_path


def test_defaults_fill_the_gaps(tmp_path: Path):
    loaded = load_config(write(tmp_path / "ctxpack.toml", "budget = 1000\n"))
    assert loaded.budget == 1000
    assert loaded.mode is None  # "no opinion", not "balanced"
    assert loaded.format is None
    assert loaded.outline is None
    assert loaded.exclude == ()
    assert loaded.unknown_keys == ()


def test_table_form(tmp_path: Path):
    path = write(
        tmp_path / "ctxpack.toml", '[ctxpack]\nbudget = 2000\nmode = "depth"\n'
    )
    loaded = load_config(path)
    assert loaded.budget == 2000
    assert loaded.mode == "depth"


def test_tool_table_form(tmp_path: Path):
    path = write(
        tmp_path / "ctxpack.toml",
        '[tool]\n\n[tool.ctxpack]\nbudget = 3000\n',
    )
    assert load_config(path).budget == 3000


def test_json_is_accepted(tmp_path: Path):
    path = tmp_path / "ctxpack.json"
    path.write_text(json.dumps({"budget": 4000, "mode": "depth"}), encoding="utf-8")
    loaded = load_config(path)
    assert loaded.budget == 4000
    assert loaded.mode == "depth"


def test_json_rejects_a_non_object(tmp_path: Path):
    path = tmp_path / "ctxpack.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(CtxpackError) as exc:
        load_config(path)
    assert str(path) in str(exc.value)
    assert "top level" in str(exc.value)


# -- edge cases --------------------------------------------------------------


def test_empty_file(tmp_path: Path):
    loaded = load_config(write(tmp_path / "ctxpack.toml", ""))
    assert loaded.is_empty()
    assert loaded.source == tmp_path / "ctxpack.toml"


def test_comments_only_file(tmp_path: Path):
    loaded = load_config(
        write(tmp_path / "ctxpack.toml", "# 60k because the window is 64k\n")
    )
    assert loaded.is_empty()
    assert loaded.budget is None


def test_empty_json_is_allowed(tmp_path: Path):
    assert load_config(write(tmp_path / "ctxpack.json", "\n  \n")).is_empty()


def test_missing_file_names_the_file(tmp_path: Path):
    missing = tmp_path / "ctxpack.toml"
    with pytest.raises(CtxpackError) as exc:
        load_config(missing)
    assert str(missing) in str(exc.value)
    assert "no such config file" in str(exc.value)


def test_directory_instead_of_file(tmp_path: Path):
    target = tmp_path / "ctxpack.toml"
    target.mkdir()
    with pytest.raises(CtxpackError) as exc:
        load_config(target)
    assert str(target) in str(exc.value)
    assert "directory" in str(exc.value)


def test_no_config_never_raises():
    assert load_or_empty(None).is_empty()
    assert load_or_empty(None).source is None


def test_base_is_none_for_an_empty_config():
    assert Config().base is None


# -- validation --------------------------------------------------------------

BAD_CASES = [
    ('mode = "sideways"', "mode", "sideways"),
    ("budget = \"60000\"", "budget", "whole number"),
    ("budget = 32.5", "budget", "whole number"),
    ("budget = true", "budget", "whole number"),
    ("budget = 0", "budget", "at least 1"),
    ("format = \"pdf\"", "format", "pdf"),
    ("manifest = \"everything\"", "manifest", "everything"),
    ("encoding = \"klingon\"", "encoding", "klingon"),
    ("exact = \"perhaps\"", "exact", "perhaps"),
    ("outline = \"yes\"", "outline", "true or false"),
    ("outline = 1", "outline", "true or false"),
    ("factor_shared = 0", "factor_shared", "true or false"),
    ("exclude = \"build/\"", "exclude", "list of strings"),
    ("exclude = [1, 2]", "exclude", "only strings"),
    ("include = 7", "include", "list of strings"),
    ("per_dir_frac = 0", "per_dir_frac", "(0, 1]"),
    ("per_dir_frac = 1.5", "per_dir_frac", "(0, 1]"),
    ("per_dir_frac = \"half\"", "per_dir_frac", "must be a number"),
    ("dedupe_threshold = 2.0", "dedupe_threshold", "(0, 1]"),
    ("reach_weight = -1", "reach_weight", "at least 0"),
    ("diff = 12", "diff", "must be a string"),
]


@pytest.mark.parametrize(("body", "key", "needle"), BAD_CASES)
def test_bad_values(tmp_path: Path, body: str, key: str, needle: str):
    path = write(tmp_path / "ctxpack.toml", f"{body}\n")
    with pytest.raises(CtxpackError) as exc:
        load_config(path)
    message = str(exc.value)
    assert str(path) in message
    assert key in message
    assert needle in message


def test_malformed_toml_names_the_file(tmp_path: Path):
    path = write(tmp_path / "ctxpack.toml", "budget = = 1\n")
    with pytest.raises(CtxpackError) as exc:
        load_config(path)
    assert str(path) in str(exc.value)
    assert "malformed TOML" in str(exc.value)


def test_malformed_json_reports_position(tmp_path: Path):
    path = write(tmp_path / "ctxpack.json", '{"budget": }')
    with pytest.raises(CtxpackError) as exc:
        load_config(path)
    assert str(path) in str(exc.value)
    assert "line 1" in str(exc.value)


def test_table_that_is_not_a_table(tmp_path: Path):
    path = write(tmp_path / "ctxpack.toml", "ctxpack = 3\n")
    with pytest.raises(CtxpackError) as exc:
        load_config(path)
    assert str(path) in str(exc.value)
    assert "ctxpack" in str(exc.value)


def test_non_utf8_config(tmp_path: Path):
    path = tmp_path / "ctxpack.toml"
    path.write_bytes(b'budget = 1\nmode = "\xff\xfe"\n')
    with pytest.raises(CtxpackError) as exc:
        load_config(path)
    assert str(path) in str(exc.value)
    assert "UTF-8" in str(exc.value)


# -- unknown keys ------------------------------------------------------------


def test_unknown_keys_are_reported_not_dropped(tmp_path: Path):
    path = write(tmp_path / "ctxpack.toml", 'budegt = 9000\nmode = "depth"\n')
    loaded = load_config(path)
    assert loaded.unknown_keys == ("budegt",)
    # The recognised key still applied: one typo must not discard the file.
    assert loaded.mode == "depth"


def test_unknown_key_warning_suggests_a_correction(tmp_path: Path):
    path = write(tmp_path / "ctxpack.toml", "budegt = 1\n")
    warning = unknown_keys_warning(load_config(path))
    assert warning is not None
    assert "budegt" in warning
    assert "budget" in warning
    assert str(path) in warning


def test_no_warning_when_everything_is_known(tmp_path: Path):
    path = write(tmp_path / "ctxpack.toml", "budget = 1\n")
    assert unknown_keys_warning(load_config(path)) is None


def test_unknown_keys_sorted(tmp_path: Path):
    path = write(tmp_path / "ctxpack.toml", "zeta = 1\nalpha = 2\n")
    assert load_config(path).unknown_keys == ("alpha", "zeta")


# -- merging -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """A parser shaped like ctxpack's, with real defaults filled in."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--budget", type=int, default=32000)
    parser.add_argument("-f", "--format", default="markdown")
    parser.add_argument("-m", "--mode", default="balanced")
    parser.add_argument("--manifest", default="full")
    parser.add_argument("-e", "--encoding", default="o200k_base")
    parser.add_argument("--exact", default="auto", choices=("auto", "exact", "never"))
    parser.add_argument("--reach-weight", type=float, default=0.0)
    parser.add_argument("--outline", action="store_true")
    parser.add_argument("--factor-shared", action="store_true")
    parser.add_argument("--diff", default=None)
    parser.add_argument("-x", "--exclude", action="append", default=[])
    parser.add_argument("-i", "--include", action="append", default=[])
    parser.add_argument("--per-dir-frac", type=float, default=0.40)
    parser.add_argument("--dedupe-threshold", type=float, default=0.85)
    return parser


def test_all_three_layers(tmp_path: Path):
    config = load_config(
        write(
            tmp_path / "ctxpack.toml",
            'budget = 60000\nmode = "coverage"\nformat = "xml"\n',
        )
    )
    args = build_parser().parse_args([])

    # Layer 3 alone: nothing specified anywhere.
    assert merge(args, Config(), None)["budget"] == 32000

    # Layer 2 over layer 3.
    from_config = merge(args, config, None)
    assert from_config["budget"] == 60000
    assert from_config["mode"] == "coverage"
    assert from_config["format"] == "xml"
    # Untouched settings keep the built-in default.
    assert from_config["per_dir_frac"] == 0.40

    # Layer 1 over layer 2.
    args = build_parser().parse_args(["-b", "1000"])
    top = merge(args, config, None, passed={"budget"})
    assert top["budget"] == 1000
    assert top["mode"] == "coverage"


def test_config_beats_default_even_when_the_flag_is_absent(tmp_path: Path):
    config = load_config(write(tmp_path / "ctxpack.toml", "reach_weight = 4.0\n"))
    args = build_parser().parse_args([])
    assert merge(args, config, None)["reach_weight"] == 4.0


def test_explicit_flag_equal_to_the_builtin_default_still_wins(tmp_path: Path):
    """The subtle case, and the reason ``passed`` exists.

    ``-m balanced`` is exactly the built-in default, so argparse hands back a
    value indistinguishable from "user said nothing" -- yet the user did type
    it, on purpose, to override a config file that says ``coverage``. Deciding
    this by comparing the parsed value against the parser's default would
    silently pick the config file and lose the override. The only sound
    answers are to know from ``argv`` (``passed``) or to default to ``UNSET``,
    which is why both exist.
    """
    config = load_config(write(tmp_path / "ctxpack.toml", 'mode = "coverage"\n'))
    args = build_parser().parse_args(["-m", "balanced"])

    assert args.mode == "balanced" == build_parser().parse_args([]).mode
    assert merge(args, config, None, passed={"mode"})["mode"] == "balanced"
    # And the fallback heuristic is honestly unable to see it.
    assert merge(args, config, None)["mode"] == "coverage"


def test_unset_sentinel_needs_no_argv_knowledge(tmp_path: Path):
    config = load_config(write(tmp_path / "ctxpack.toml", 'mode = "coverage"\n'))
    args = build_parser().parse_args([])
    args.mode = UNSET
    assert merge(args, config, None)["mode"] == "coverage"

    args.mode = "depth"
    assert merge(args, config, None)["mode"] == "depth"


def test_unset_beats_nothing_and_none_is_never_explicit(tmp_path: Path):
    args = build_parser().parse_args([])
    args.budget = UNSET
    args.diff = None
    config = load_config(write(tmp_path / "ctxpack.toml", "diff = \"main...HEAD\"\n"))
    resolved = merge(args, config, None)
    assert resolved["budget"] == 32000
    assert resolved["diff"] == "main...HEAD"


def test_merge_accepts_a_namespace_or_a_mapping(tmp_path: Path):
    config = load_config(write(tmp_path / "ctxpack.toml", "budget = 5\n"))
    args = build_parser().parse_args([])
    assert merge(vars(args), config, None)["budget"] == 5
    assert merge(args, config, None)["budget"] == 5


def test_custom_defaults_mapping():
    resolved = merge({}, Config(), {"budget": 1, "mode": "depth"})
    assert resolved["budget"] == 1
    assert resolved["mode"] == "depth"


def test_merge_result_updates_a_namespace(tmp_path: Path):
    config = load_config(write(tmp_path / "ctxpack.toml", "budget = 700\n"))
    args = build_parser().parse_args([])
    vars(args).update(merge(args, config, None))
    assert args.budget == 700


# -- list merging ------------------------------------------------------------


def test_lists_extend_rather_than_replace(tmp_path: Path):
    """Config first, then the command line, duplicates removed.

    Extending is the least surprising behaviour: an exclude list is shared
    policy that a one-off command adds to, and either side losing its entries
    would make the result depend on which of the two the reader happened to
    look at first.
    """
    config = load_config(
        write(tmp_path / "ctxpack.toml", 'exclude = ["build/", "dist/"]\n')
    )
    args = build_parser().parse_args(["-x", "*.log", "-x", "build/"])
    resolved = merge(args, config, None, passed={"exclude"})
    assert resolved["exclude"] == ("build/", "dist/", "*.log")


def test_include_also_extends(tmp_path: Path):
    config = load_config(write(tmp_path / "ctxpack.toml", 'include = ["src/**"]\n'))
    args = build_parser().parse_args(["-i", "docs/**"])
    resolved = merge(args, config, None, passed={"include"})
    assert resolved["include"] == ("src/**", "docs/**")


def test_lists_from_config_alone(tmp_path: Path):
    config = load_config(write(tmp_path / "ctxpack.toml", 'exclude = ["a", "b"]\n'))
    args = build_parser().parse_args([])
    assert merge(args, config, None)["exclude"] == ("a", "b")


def test_lists_are_tuples_even_from_argparse(tmp_path: Path):
    """The packer iterates them; a list leaking through breaks hashing."""
    config = load_config(write(tmp_path / "ctxpack.toml", 'exclude = ["a"]\n'))
    args = build_parser().parse_args(["-x", "b"])
    resolved = merge(args, config, None, passed={"exclude"})
    assert isinstance(resolved["exclude"], tuple)
    assert resolved["include"] == ()


# -- relative paths ----------------------------------------------------------


def test_relative_paths_resolve_against_the_config_directory(
    tmp_path: Path, monkeypatch
):
    """Run from an unrelated CWD; the answer must not move.

    The bug this guards against is reading ``exclude = ["tests/**"]`` as
    relative to wherever the command happened to run, so that
    ``cd src && ctxpack .`` quietly applies different filters than
    ``ctxpack .`` from the root.
    """
    repo = tmp_path / "repo"
    (repo / "src" / "deep").mkdir(parents=True)
    path = write(repo / "ctxpack.toml", 'exclude = ["tests/**"]\ninclude = ["src/**"]\n')
    config = load_config(path)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    for cwd in (repo, elsewhere, repo / "src"):
        monkeypatch.chdir(cwd)
        rebased = config.with_patterns(repo)
        assert rebased.exclude == ("tests/**",)
        assert rebased.include == ("src/**",)

    # merge() must do the same when handed the root.
    monkeypatch.chdir(elsewhere)
    args = build_parser().parse_args([])
    assert merge(args, config, None, root=repo)["exclude"] == ("tests/**",)


def test_config_deeper_than_the_root_is_rebased(tmp_path: Path):
    """Config in a subdirectory: patterns need the subdirectory prefix."""
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    config = load_config(write(repo / "tools" / "ctxpack.toml", 'exclude = ["gen/**"]\n'))
    assert config.with_patterns(repo).exclude == ("tools/gen/**",)


def test_config_above_the_root_drops_out_of_scope_patterns(tmp_path: Path):
    """Config at the repo root, packing src/: `tests/**` is out of scope.

    It becomes ``../tests/**``, which matches nothing under ``src``. That is
    the correct answer -- tests are not inside the packed root -- and it is
    better than silently keeping ``tests/**``, which would then match a
    ``src/tests`` directory and exclude code the user asked for.
    """
    repo = tmp_path / "repo"
    (repo / "src" / "tests").mkdir(parents=True)
    (repo / "tests").mkdir()
    config = load_config(write(repo / "ctxpack.toml", 'exclude = ["tests/**"]\n'))
    rebased = config.with_patterns(repo / "src")
    assert rebased.exclude == ("../tests/**",)


def test_absolute_patterns_become_root_relative(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    target = repo / "generated"
    config = load_config(
        write(repo / "ctxpack.toml", f'exclude = ["{target.as_posix()}/**"]\n')
    )
    assert config.with_patterns(repo / "src").exclude == ("../generated/**",)


def test_resolve_patterns_is_a_no_op_at_the_same_directory(tmp_path: Path):
    patterns = ("build/", "*.log", "")
    assert resolve_patterns(patterns, base=tmp_path, root=tmp_path) == patterns


def test_resolve_patterns_with_nothing_to_do(tmp_path: Path):
    assert resolve_patterns((), base=tmp_path, root=tmp_path / "x") == ()


def test_resolve_patterns_normalises_dot_prefix(tmp_path: Path):
    resolved = resolve_patterns(("./gen/**",), base=tmp_path / "tools", root=tmp_path)
    assert resolved == ("tools/gen/**",)


# -- passed_dests ------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], set()),
        (["-b", "60000"], {"budget"}),
        (["--budget", "60000"], {"budget"}),
        (["--budget=60000"], {"budget"}),
        (["-b60000"], {"budget"}),
        (["-xb"], {"exclude"}),
        (["-xb", "-xc"], {"exclude"}),
        (["--mode", "depth", "--exact", "never"], {"mode", "exact"}),
        (["--mode", "depth", "positional"], {"mode"}),
        (["--budget", "1", "--", "-m", "depth"], {"budget"}),
        (["--bud", "1"], {"budget"}),  # unambiguous abbreviation
        (["-m", "coverage", "-b", "1"], {"mode", "budget"}),
    ],
)
def test_passed_dests(argv, expected):
    assert set(passed_dests(build_parser(), argv)) == expected


def test_passed_dests_ignores_unknown_flags():
    """Unknown flags belong to argparse's own error, not to us."""
    assert passed_dests(build_parser(), ["--nope"]) == frozenset()


def test_passed_dists_with_the_real_cli_parser():
    from ctxpack.cli import build_parser as real_parser

    found = passed_dests(real_parser(), ["-b", "1000", "--outline"])
    assert {"budget", "outline"} <= found


# -- describe ----------------------------------------------------------------


def test_describe_covers_every_effective_setting(tmp_path: Path):
    config = load_config(
        write(
            tmp_path / "ctxpack.toml",
            'budget = 60000\nmode = "coverage"\nexclude = ["build/"]\n',
        )
    )
    text = describe(config)
    assert str(tmp_path / "ctxpack.toml") in text
    assert "60000" in text
    assert "coverage" in text
    assert "(config)" in text
    assert "(default)" in text
    # Every default-able setting gets a line, including unset ones: an absent
    # line is indistinguishable from a forgotten one.
    for key in ("budget", "format", "mode", "manifest", "encoding", "exact"):
        assert key in text
    # A setting the file said nothing about shows its built-in default, not
    # nothing: an absent line is indistinguishable from a forgotten one.
    assert "0.85" in text
    assert "markdown" in text  # format default; mode was overridden above
    assert len(text.splitlines()) >= len(cfg.DEFAULTS)


def test_describe_with_no_config():
    text = describe(Config())
    assert "(none found)" in text


def test_describe_defaults_to_an_empty_config():
    assert describe() == describe(Config())


def test_describe_flags_unknown_keys(tmp_path: Path):
    config = load_config(write(tmp_path / "ctxpack.toml", "budegt = 1\n"))
    text = describe(config)
    assert "budegt" in text
    assert "budget" in text


def test_describe_accepts_custom_defaults():
    assert "111" in describe(Config(), {"budget": 111})


# -- Python 3.10 -------------------------------------------------------------


def test_toml_on_310_explains_itself(tmp_path: Path, monkeypatch):
    """3.10 has no stdlib TOML reader; the message must be actionable.

    A traceback here would be the worst outcome: the user has a valid config
    and a supported interpreter, and the only thing missing is a parser that
    Python did not ship.
    """
    monkeypatch.setattr(cfg, "_tomllib", lambda: None)
    path = write(tmp_path / "ctxpack.toml", "budget = 1\n")
    with pytest.raises(CtxpackError) as exc:
        load_config(path)
    message = str(exc.value)
    assert str(path) in message
    assert "3.11" in message
    assert "ctxpack.json" in message  # a way forward, not just a complaint
    assert "3.10" in message  # says which interpreter it is running


def test_json_still_works_on_310(tmp_path: Path, monkeypatch):
    """The escape hatch has to actually work, or the message is a lie."""
    monkeypatch.setattr(cfg, "_tomllib", lambda: None)
    path = tmp_path / "ctxpack.json"
    path.write_text(json.dumps({"budget": 4000}), encoding="utf-8")
    assert load_config(path).budget == 4000


def test_discovery_ignores_json_first_on_310(tmp_path: Path, monkeypatch):
    """A TOML file is still discovered, so the error names the real file."""
    monkeypatch.setattr(cfg, "_tomllib", lambda: None)
    write(tmp_path / "ctxpack.toml", "budget = 1\n")
    assert discover_config(tmp_path) == tmp_path / "ctxpack.toml"
