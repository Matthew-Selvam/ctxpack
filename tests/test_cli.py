"""The command line surface: exit codes, streams, and argument handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ctxpack.cli import _parse_size, main
from ctxpack.errors import CtxpackError


def run(capsys, argv) -> tuple[int, str, str]:
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# -- argument plumbing -------------------------------------------------------


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "ctxpack" in capsys.readouterr().out


def test_pack_is_the_default_subcommand(capsys, project: Path):
    code, out, _ = run(capsys, [str(project), "-b", "5000", "-q"])
    assert code == 0
    assert out.startswith("# ctxpack bundle")


def test_explicit_pack_matches_default(capsys, project: Path):
    _, implicit, _ = run(capsys, [str(project), "-b", "4000", "-q"])
    _, explicit, _ = run(capsys, ["pack", str(project), "-b", "4000", "-q"])
    assert implicit == explicit


def test_root_flag(capsys, project: Path):
    code, out, _ = run(capsys, ["pack", "-C", str(project), "-b", "4000", "-q"])
    assert code == 0
    assert "# ctxpack bundle" in out


def test_missing_path_is_an_error(capsys, tmp_path: Path):
    code, _, err = run(capsys, ["pack", str(tmp_path / "nope"), "-q"])
    assert code == 2
    assert "ctxpack:" in err


def test_empty_directory_is_an_error(capsys, tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    code, _, err = run(capsys, ["pack", str(empty), "-q"])
    assert code == 2
    assert "no text files" in err


def test_no_matching_include_is_an_error(capsys, project: Path):
    code, _, err = run(capsys, ["pack", str(project), "-i", "*.rs", "-q"])
    assert code == 2
    assert "no text files" in err


def test_budget_must_be_positive(capsys, project: Path):
    code, _, err = run(capsys, ["pack", str(project), "-b", "0", "-q"])
    assert code == 2
    assert "positive" in err


def test_bad_encoding_rejected_by_argparse(capsys, project: Path):
    with pytest.raises(SystemExit):
        main(["pack", str(project), "-e", "not-real"])


# -- size parsing ------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("500", 500),
        ("500k", 500_000),
        ("500kb", 500_000),
        ("5M", 5_000_000),
        ("2G", 2_000_000_000),
        ("1.5M", 1_500_000),
    ],
)
def test_parse_size(text, expected):
    assert _parse_size(text) == expected


def test_parse_size_rejects_garbage():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        _parse_size("banana")


# -- pack --------------------------------------------------------------------


def test_pack_respects_budget(capsys, project: Path):
    from ctxpack.tokens import Tokenizer

    _, out, _ = run(capsys, ["pack", str(project), "-b", "6000", "-q"])
    counted = Tokenizer().count(out).tokens
    assert counted <= 6_000 * 1.35  # header/format overhead is real


@pytest.mark.parametrize("fmt", ["markdown", "xml", "json", "tree"])
def test_every_format_from_the_cli(capsys, project: Path, fmt):
    code, out, _ = run(capsys, ["pack", str(project), "-b", "5000", "-f", fmt, "-q"])
    assert code == 0
    assert out.strip()


def test_write_to_file(capsys, project: Path, tmp_path: Path):
    target = tmp_path / "bundle.md"
    code, out, err = run(
        capsys, ["pack", str(project), "-b", "5000", "-o", str(target), "-q"]
    )
    assert code == 0
    assert out == ""
    assert target.exists()
    assert "wrote" in err or err == ""


def test_stats_to_stderr(capsys, project: Path):
    _, _, err = run(
        capsys, ["pack", str(project), "-b", "8000", "--stats", "-o", "/dev/null"]
    )
    assert "files packed" in err


def test_include_and_exclude(capsys, project: Path):
    _, only_py, _ = run(
        capsys, ["pack", str(project), "-b", "5000", "-i", "*.py", "-q", "-f", "tree"]
    )
    assert "README.md" not in only_py
    assert ".py" in only_py


def test_exclude_flag(capsys, project: Path):
    _, out, _ = run(
        capsys,
        ["pack", str(project), "-b", "5000", "-x", "*.py", "-q", "-f", "tree"],
    )
    assert "main.py" not in out


def test_mode_choices(capsys, project: Path):
    for mode in ("balanced", "coverage", "depth"):
        code, out, _ = run(
            capsys, ["pack", str(project), "-b", "5000", "-m", mode, "-q"]
        )
        assert code == 0
        assert out


def test_manifest_choices(capsys, project: Path):
    _, out, _ = run(
        capsys,
        ["pack", str(project), "-b", "5000", "--manifest", "none", "-q", "-f", "tree"],
    )
    assert out.strip()


def test_no_dedupe_and_no_truncate(capsys, project: Path):
    code, out, _ = run(
        capsys,
        ["pack", str(project), "-b", "5000", "--no-dedupe", "--no-truncate", "-q"],
    )
    assert code == 0
    assert out


def test_exact_requires_tiktoken(capsys, project: Path, tiktoken_available):
    code, _, err = run(
        capsys, ["pack", str(project), "-b", "4000", "--exact", "exact", "-q"]
    )
    if tiktoken_available:
        assert code == 0
    else:
        assert code == 2
        assert "tiktoken" in err


def test_never_exact_uses_the_estimator(capsys, project: Path):
    _, out, _ = run(
        capsys,
        ["pack", str(project), "-b", "4000", "--exact", "never", "-q", "-f", "tree"],
    )
    assert "(estimate)" in out


# -- count -------------------------------------------------------------------


def test_count_table(capsys, project: Path):
    code, _, err = run(capsys, ["count", str(project), "-n", "5"])
    assert code == 0
    assert "score" in err
    assert "path" in err


def test_count_json(capsys, project: Path):
    _, out, _ = run(capsys, ["count", str(project), "--json", "-n", "5"])
    payload = json.loads(out)
    assert payload["discovered"] > 0
    assert len(payload["files"]) <= 5
    assert payload["files"][0]["path"]


def test_count_all_rows(capsys, project: Path):
    _, out, _ = run(capsys, ["count", str(project), "--json", "-n", "0"])
    assert len(json.loads(out)["files"]) > 5


# -- explain -----------------------------------------------------------------


def test_explain(capsys, project: Path):
    code, out, _ = run(capsys, ["explain", str(project), "-n", "3"])
    assert code == 0
    assert "README.md" in out


def test_explain_filter(capsys, project: Path):
    _, out, _ = run(
        capsys, ["explain", str(project), "--filter", "locales/*", "-n", "10"]
    )
    assert "locales/" in out
    assert "README.md" not in out


def test_explain_json(capsys, project: Path):
    _, out, _ = run(capsys, ["explain", str(project), "--json", "-n", "2"])
    payload = json.loads(out)
    assert len(payload["files"]) == 2
    assert payload["files"][0]["signals"]


# -- calibrate ---------------------------------------------------------------


def test_calibrate(capsys, project: Path, tiktoken_available):
    if not tiktoken_available:
        pytest.skip("tiktoken is not installed")
    code, out, _ = run(capsys, ["calibrate", str(project)])
    assert code == 0
    assert "least-squares factor" in out
    assert "MAPE" in out


def test_calibrate_write(capsys, project: Path, tmp_path, monkeypatch, tiktoken_available):
    if not tiktoken_available:
        pytest.skip("tiktoken is not installed")
    target = tmp_path / "cal.json"
    monkeypatch.setenv("CTXPACK_CALIBRATION", str(target))
    code, out, _ = run(capsys, ["calibrate", str(project), "--write"])
    assert code == 0
    assert target.exists()
    assert "wrote" in out


def test_calibrate_needs_enough_files(capsys, tmp_path: Path, tiktoken_available):
    (tmp_path / "one.py").write_text("X = 1\n", encoding="utf-8")
    code, _, err = run(capsys, ["calibrate", str(tmp_path)])
    assert code == 2
    if tiktoken_available:
        assert "at least 5" in err
    else:
        # Without tiktoken there is nothing to calibrate against, and saying so
        # is more useful than a sample-count complaint.
        assert "tiktoken" in err


# -- robustness --------------------------------------------------------------


def test_broken_pipe_is_not_a_traceback(capsys, project: Path):
    """`ctxpack ... | head` must exit quietly."""

    class Closed:
        def write(self, *_):
            raise BrokenPipeError

        def flush(self):
            pass

        def close(self):
            pass

        def isatty(self):
            return False

    import sys

    original = sys.stdout
    sys.stdout = Closed()
    try:
        code = main(["pack", str(project), "-b", "5000", "-q"])
    finally:
        sys.stdout = original
    assert code == 0


def test_ctxpack_error_is_a_clean_message(capsys):
    from ctxpack.render import render

    with pytest.raises(CtxpackError):
        render(None, "nope")


def test_subcommand_name_that_is_also_a_directory(tmp_path, monkeypatch, capsys):
    """`ctxpack count` in a repo containing `count/` must not silently misread.

    Guessing wrong is invisible: the subcommand happily reports on the whole
    current directory, so the user gets a token table for their repository
    instead of a bundle for the directory they named.
    """
    root = tmp_path / "proj"
    (root / "count").mkdir(parents=True)
    (root / "count" / "a.py").write_text("X = 1\n", encoding="utf-8")
    monkeypatch.chdir(root)

    code, _, err = run(capsys, ["count"])
    assert code == 2
    assert "both a subcommand and a path" in err

    # Disambiguating forms all work. Bundle paths are relative to the packed
    # root, so packing ./count yields "a.py", not "count/a.py".
    code, out, _ = run(capsys, ["./count", "-b", "2000", "-q", "-f", "tree"])
    assert code == 0
    assert out.rstrip().endswith("a.py")
    assert str((root / "count").resolve()) in out

    code, _, _ = run(capsys, ["count", "--json"])
    assert code == 0


def test_subcommand_name_that_is_a_directory_is_fine_when_flagged(capsys, project: Path):
    """No ambiguity error when the user clearly means the subcommand."""
    (project / "count").mkdir()
    (project / "count" / "x.py").write_text("Y = 2\n", encoding="utf-8")
    code, out, _ = run(capsys, ["count", str(project), "--json", "-n", "0"])
    assert code == 0
    assert json.loads(out)["files"]


def test_missing_subcommand_name_directory_is_unaffected(capsys, project: Path, monkeypatch):
    monkeypatch.chdir(project)
    code, out, _ = run(capsys, ["count", "--json", "-n", "0"])
    assert code == 0
    assert out


def test_module_entrypoint_exists():
    import ctxpack.__main__ as entry

    assert entry.main is main


# -- watch -------------------------------------------------------------------


class _FakeClock:
    """A clock the test drives, so the loop is exercised without wall time.

    ``on_sleep`` runs after each tick and is where a test mutates the tree, which
    is how a "file was edited" event is produced with no threads and no flakiness.
    """

    def __init__(self, on_sleep=None, edit_after: int | None = None, edit=None):
        self.now = 0.0
        self.ticks = 0
        self._on_sleep = on_sleep
        self._edit_after = edit_after
        self._edit = edit

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.ticks += 1
        if self._edit is not None and self.ticks == self._edit_after:
            self._edit()
        if self._on_sleep is not None:
            self._on_sleep(self)


def _append_to(path: Path, text: str):
    """Append to ``path`` -- the edit a fake clock performs."""

    def do():
        path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")

    return do


def _patch_loop(monkeypatch, wrapper):
    """Replace the watch loop with ``wrapper``.

    The CLI imports ``run_watch`` from inside ``_cmd_watch``, so patching the
    module attribute is what takes effect. The *original* is captured first,
    because a wrapper that wants to run real cycles must call the real thing --
    calling ``watch.run_watch`` would find itself.
    """
    from ctxpack import watch

    real = watch.run_watch

    def entry(paths, repack, **kwargs):
        return wrapper(real, paths, repack, **kwargs)

    monkeypatch.setattr(watch, "run_watch", entry)


def test_watch_writes_the_first_bundle_immediately(
    capsys, project: Path, tmp_path: Path, monkeypatch
):
    """A watcher that blocks before writing anything looks like a hang.

    This is the whole reason the first build happens outside the loop rather
    than on the first detected change: nothing may have been edited yet.
    """
    seen: list[list[str]] = []

    def wrapper(real, paths, repack, **kwargs):
        seen.append(list(paths))
        kwargs.pop("iterations", None)
        return real(paths, repack, iterations=1, **kwargs)

    _patch_loop(monkeypatch, wrapper)
    target = tmp_path / "w.md"
    code, _, _ = run(
        capsys, ["pack", str(project), "-b", "4000", "--watch", "-q", "-o", str(target)]
    )

    assert code == 0
    assert target.exists(), "the first bundle must exist before the loop starts"
    assert target.read_text(encoding="utf-8").strip()
    assert seen, "the watch loop should have been entered"
    assert seen[0], "the watch set should be the discovered files, not empty"


def test_watch_rebuilds_on_change(capsys, project: Path, tmp_path: Path, monkeypatch):
    """A real edit reaches the real repack through the real debounce."""
    builds = []

    def wrapper(real, paths, repack, **kwargs):
        clock = _FakeClock(edit_after=2, edit=_append_to(project / "src" / "app" / "main.py", "\n# edit\n"))
        kwargs.pop("iterations", None)
        return real(
            paths,
            lambda: (builds.append(1), repack())[1],
            iterations=4,
            clock=clock,
            sleep=clock.sleep,
            **kwargs,
        )

    _patch_loop(monkeypatch, wrapper)
    code, _, err = run(
        capsys,
        ["pack", str(project), "-b", "4000", "--watch",
         "--watch-interval", "1", "-o", str(tmp_path / "w.md")],
    )

    assert code == 0
    assert builds, "the edit should have triggered a rebuild"
    assert "modified" in err, "the change should be reported"
    assert "wrote" in err, "the rebuild should have written a bundle"


def test_watch_coalesces_a_burst_into_one_rebuild(
    capsys, project: Path, tmp_path: Path, monkeypatch
):
    """Several writes in quick succession must not rebuild several times.

    This is the difference between `--watch` and a script that repacks in a save
    hook: the CPU is spent once per logical edit, not once per write.
    """
    builds = []
    main_py = project / "src" / "app" / "main.py"
    state = {"w": 0}

    def keep_writing(_clock):
        state["w"] += 1
        if 2 <= state["w"] <= 5:
            _append_to(main_py, f"\n# step {state['w']}\n")()

    def wrapper(real, paths, repack, **kwargs):
        clock = _FakeClock(on_sleep=keep_writing)
        kwargs.pop("iterations", None)
        return real(
            paths,
            lambda: (builds.append(1), repack())[1],
            iterations=8,
            clock=clock,
            sleep=clock.sleep,
            **kwargs,
        )

    _patch_loop(monkeypatch, wrapper)
    run(
        capsys,
        ["pack", str(project), "-b", "4000", "--watch", "-q",
         "--watch-interval", "1", "--watch-settle", "3",
         "-o", str(tmp_path / "w.md")],
    )
    assert builds == [1], f"one logical edit produced {len(builds)} rebuilds"


def test_watch_survives_a_failing_rebuild(
    capsys, project: Path, tmp_path: Path, monkeypatch
):
    """A syntax error mid-edit must not kill the watcher.

    A half-typed file is the normal state of a repository being worked on, and
    a watcher that dies on the first one is useless exactly when it is needed.
    """
    attempts: list[int] = []

    def boom():
        attempts.append(1)
        raise SyntaxError("invalid syntax")

    def wrapper(real, paths, repack, **kwargs):
        clock = _FakeClock(
            edit_after=2, edit=_append_to(project / "src" / "app" / "main.py", "\n# edit\n")
        )
        kwargs.pop("iterations", None)
        return real(
            paths, boom, iterations=3, clock=clock, sleep=clock.sleep, **kwargs
        )

    _patch_loop(monkeypatch, wrapper)
    target = tmp_path / "w.md"
    code, _, err = run(
        capsys,
        ["pack", str(project), "-b", "4000", "--watch", "--watch-interval", "1",
         "-o", str(target)],
    )

    assert code == 0
    assert attempts, "the failing rebuild should have been attempted"
    assert "rebuild failed" in err
    # The previous bundle stays valid: a failed rebuild must not truncate it.
    assert target.read_text(encoding="utf-8").strip(), "bundle was lost"


def test_watch_interval_flag_beats_the_environment(
    capsys, project: Path, tmp_path: Path, monkeypatch
):
    """Same precedence as every other flag here: typed flag > env > default."""
    seen: dict = {}

    def wrapper(real, paths, repack, **kwargs):
        seen.update(kwargs)
        return 0

    _patch_loop(monkeypatch, wrapper)
    monkeypatch.setenv("CTXPACK_WATCH_INTERVAL", "9.0")

    run(capsys, ["pack", str(project), "-b", "4000", "--watch", "-q",
                 "--watch-interval", "0.25", "-o", str(tmp_path / "w.md")])
    assert seen["interval"] == 0.25

    seen.clear()
    run(capsys, ["pack", str(project), "-b", "4000", "--watch", "-q",
                 "-o", str(tmp_path / "w2.md")])
    assert seen["interval"] == 9.0, "env should win over the built-in default"


def test_watch_reports_the_changed_paths(
    capsys, project: Path, tmp_path: Path, monkeypatch
):
    """A rebuild that does not say what changed is not debuggable."""
    from ctxpack.watch import Change

    def wrapper(real, paths, repack, **kwargs):
        kwargs["on_change"]([Change("src/app/main.py", "modified")])
        return 0

    _patch_loop(monkeypatch, wrapper)
    _, _, err = run(
        capsys,
        ["pack", str(project), "-b", "4000", "--watch", "--watch-interval", "0.1",
         "-o", str(tmp_path / "w.md")],
    )
    assert "modified src/app/main.py" in err
    assert "watching" in err


def test_watch_is_off_by_default(capsys, project: Path, monkeypatch):
    """A regression guard: `pack` must not start blocking by accident."""

    def wrapper(real, paths, repack, **kwargs):  # pragma: no cover
        raise AssertionError("run_watch called without --watch")

    _patch_loop(monkeypatch, wrapper)
    code, _, _ = run(capsys, ["pack", str(project), "-b", "3000", "-q"])
    assert code == 0
