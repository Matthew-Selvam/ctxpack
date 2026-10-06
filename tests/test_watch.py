"""Tests for the watch loop.

The settle/debounce logic is the part worth testing, and it is tested with an
injected clock so the suite never actually sleeps. A watch test that really
sleeps is a flaky watch test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctxpack.watch import (
    DEFAULT_INTERVAL,
    MAX_INTERVAL,
    MIN_INTERVAL,
    Change,
    diff_snapshots,
    env_interval,
    mtime_of,
    run_watch,
    snapshot,
    watch_paths,
)


class Clock:
    """A clock the test advances by hand.

    ``sleep`` moves time forward rather than blocking, so debounce logic can be
    exercised exactly, at full speed, with no flakiness.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


# -- snapshots ---------------------------------------------------------------


def test_snapshot_records_mtime_and_size(tmp_path: Path):
    (tmp_path / "a.py").write_text("X = 1\n", encoding="utf-8")
    snap = snapshot(["a.py"], root=tmp_path)
    assert snap["a.py"][1] == len("X = 1\n")


def test_snapshot_skips_missing_paths(tmp_path: Path):
    """A file deleted between listing and stat is absent, not an error."""
    (tmp_path / "gone.py").write_text("x", encoding="utf-8")
    snap = snapshot(["gone.py", "never-existed.py"], root=tmp_path)
    assert set(snap) == {"gone.py"}


def test_snapshot_is_empty_for_an_empty_repo(tmp_path: Path):
    assert snapshot([], root=tmp_path) == {}


# -- diffing -----------------------------------------------------------------


def test_diff_reports_added_modified_deleted():
    before = {"keep": (1, 10), "edit": (1, 10), "drop": (1, 10)}
    after = {"keep": (1, 10), "edit": (2, 10), "new": (1, 5)}
    assert diff_snapshots(before, after) == [
        Change("edit", "modified"),
        Change("new", "added"),
        Change("drop", "deleted"),
    ]


def test_diff_is_empty_for_no_change():
    stamp = {"a": (1, 2)}
    assert diff_snapshots(stamp, stamp) == []


def test_diff_detects_a_rewrite_with_identical_mtime():
    """Size catches the write mtime granularity missed.

    Some filesystems round timestamps coarsely enough that a fast rewrite keeps
    the same mtime. Size is the cheap second signal.
    """
    before = {"a": (1000, 5)}
    after = {"a": (1000, 500)}
    assert diff_snapshots(before, after) == [Change("a", "modified")]


def test_diff_is_deterministically_ordered():
    before = {f"f{i}": (1, 1) for i in range(20)}
    after = {f"f{i}": (2, 1) for i in range(20)}
    first = diff_snapshots(before, after)
    second = diff_snapshots(dict(reversed(list(before.items()))), after)
    assert first == second
    assert [c.path for c in first] == sorted(c.path for c in first)


# -- the loop ----------------------------------------------------------------


def test_repack_only_on_change(tmp_path: Path):
    (tmp_path / "a.py").write_text("X = 1\n", encoding="utf-8")
    paths = ["a.py"]
    clock = Clock()
    calls: list[list[Change]] = []

    # Three idle cycles, then a change.
    plan = {"tick": 0}

    def fake_sleep(seconds: float) -> None:
        clock.slept.append(seconds)
        clock.now += seconds
        plan["tick"] += 1
        if plan["tick"] == 4:
            (tmp_path / "a.py").write_text("X = 2\n", encoding="utf-8")

    builds = []
    run_watch(
        paths,
        lambda: builds.append(1),
        root=tmp_path,
        interval=1.0,
        settle=2.0,
        iterations=6,
        clock=clock,
        sleep=fake_sleep,
        on_change=calls.append,
    )
    assert len(builds) == 1
    assert calls == [[Change("a.py", "modified")]]


def test_no_repack_when_nothing_changes(tmp_path: Path):
    (tmp_path / "a.py").write_text("X = 1\n", encoding="utf-8")
    clock = Clock()
    builds = []
    run_watch(
        ["a.py"],
        lambda: builds.append(1),
        root=tmp_path,
        interval=1.0,
        settle=1.0,
        iterations=10,
        clock=clock,
        sleep=clock.sleep,
    )
    assert builds == []


def test_burst_of_writes_triggers_one_repack(tmp_path: Path):
    """A formatter plus a save is several writes; it should be one rebuild.

    Without the debounce the bundle is rebuilt three times for one edit, which
    wastes exactly the CPU the watch mode is trying to save.
    """
    # Watch a path that exists from the start and then gets rewritten repeatedly,
    # rather than relying on newly created files: the watch set is the set the
    # caller discovered, and a brand new file is not in it until the next
    # re-discovery. That is documented in `watch_paths`.
    (tmp_path / "a.py").write_text("X = 1\n", encoding="utf-8")
    clock = Clock()
    builds = []
    writes = 0

    def fake_sleep(seconds: float) -> None:
        nonlocal writes
        clock.slept.append(seconds)
        clock.now += seconds
        writes += 1
        if 2 <= writes <= 5:
            # One logical edit, written in several steps.
            (tmp_path / "a.py").write_text(f"X = {writes}\n" + "# pad\n" * writes, encoding="utf-8")

    run_watch(
        ["a.py"],
        lambda: builds.append(1),
        root=tmp_path,
        interval=1.0,
        settle=3.0,
        max_settle=100.0,
        iterations=8,
        clock=clock,
        sleep=fake_sleep,
    )
    assert len(builds) == 1


def test_settle_is_bounded_when_the_tree_never_goes_quiet(tmp_path: Path):
    """Continuous changes must still produce a rebuild.

    The debounce pushes its own deadline out every time it sees a change, so a
    tree that never settles -- a format-on-save loop, a test runner rewriting
    fixtures, someone holding ctrl-S -- would otherwise spin forever and never
    rebuild once. `max_settle` is the bound that prevents that.
    """
    (tmp_path / "a.py").write_text("X = 1\n", encoding="utf-8")
    clock = Clock()
    builds = []
    writes = 0

    def fake_sleep(seconds: float) -> None:
        nonlocal writes
        clock.slept.append(seconds)
        clock.now += seconds
        writes += 1
        (tmp_path / "a.py").write_text(f"X = {writes}\n", encoding="utf-8")

    run_watch(
        ["a.py"],
        lambda: builds.append(1),
        root=tmp_path,
        interval=1.0,
        settle=3.0,
        max_settle=4.0,
        iterations=3,
        clock=clock,
        sleep=fake_sleep,
    )
    assert len(builds) == 3


def test_settle_stops_early_once_the_tree_goes_quiet(tmp_path: Path):
    """The common case must not pay the full max_settle budget."""
    (tmp_path / "a.py").write_text("X = 1\n", encoding="utf-8")
    clock = Clock()
    builds = []
    writes = 0

    def fake_sleep(seconds: float) -> None:
        nonlocal writes
        clock.slept.append(seconds)
        clock.now += seconds
        writes += 1
        if writes == 2:
            (tmp_path / "a.py").write_text("X = 2\n", encoding="utf-8")

    run_watch(
        ["a.py"],
        lambda: builds.append(1),
        root=tmp_path,
        interval=1.0,
        settle=2.0,
        max_settle=100.0,
        iterations=3,
        clock=clock,
        sleep=fake_sleep,
    )
    assert len(builds) == 1


def test_a_failing_repack_does_not_kill_the_watch(tmp_path: Path):
    """A syntax error mid-edit is the normal state of a repo being worked on.

    A watcher that dies on the first failed rebuild is useless precisely when it
    is needed, so the error is reported and the loop continues.
    """
    (tmp_path / "a.py").write_text("X = 1\n", encoding="utf-8")
    clock = Clock()
    errors: list[BaseException] = []
    attempts = 0
    writes = 0

    def fake_sleep(seconds: float) -> None:
        nonlocal writes
        clock.slept.append(seconds)
        clock.now += seconds
        writes += 1
        if writes in (2, 5):
            (tmp_path / "a.py").write_text(f"X = {writes}\n", encoding="utf-8")

    def repack() -> None:
        nonlocal attempts
        attempts += 1
        raise SyntaxError("invalid syntax")

    run_watch(
        ["a.py"],
        repack,
        root=tmp_path,
        interval=1.0,
        settle=1.0,
        iterations=6,
        clock=clock,
        sleep=fake_sleep,
        on_error=errors.append,
    )
    assert attempts == 2
    assert len(errors) == 2
    assert all(isinstance(e, SyntaxError) for e in errors)


def test_repack_error_propagates_without_a_handler(tmp_path: Path):
    """Without an error handler the failure is the caller's problem, not hidden."""
    (tmp_path / "a.py").write_text("X = 1\n", encoding="utf-8")
    clock = Clock()
    writes = 0

    def fake_sleep(seconds: float) -> None:
        nonlocal writes
        clock.sleep(seconds)
        writes += 1
        if writes == 2:
            (tmp_path / "a.py").write_text("X = 2\n", encoding="utf-8")

    def repack() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_watch(
            ["a.py"],
            repack,
            root=tmp_path,
            interval=1.0,
            settle=1.0,
            iterations=4,
            clock=clock,
            sleep=fake_sleep,
        )


def test_watch_terminates_when_given_iterations(tmp_path: Path):
    """The bounded mode is what makes this loop testable at all."""
    clock = Clock()
    run_watch(
        [],
        lambda: None,
        root=tmp_path,
        iterations=5,
        clock=clock,
        sleep=clock.sleep,
    )
    assert len(clock.slept) == 5


# -- helpers -----------------------------------------------------------------


def test_watch_paths_uses_discovered_paths_only():
    class FakeCandidate:
        def __init__(self, path):
            self.path = path

    candidates = [FakeCandidate("a.py"), FakeCandidate("b.py")]
    assert watch_paths(candidates) == ["a.py", "b.py"]


def test_mtime_of_missing_file_is_negative():
    assert mtime_of("definitely-not-here.py") < 0


def test_mtime_of_real_file(tmp_path: Path):
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    assert mtime_of("a.py", root=tmp_path) > 0


def test_env_interval_default():
    assert env_interval({}) == DEFAULT_INTERVAL


def test_env_interval_reads_the_process_environment_by_default(monkeypatch):
    monkeypatch.setenv("CTXPACK_WATCH_INTERVAL", "1.5")
    assert env_interval() == 1.5


def test_env_interval_is_clamped():
    assert env_interval({"CTXPACK_WATCH_INTERVAL": "0.001"}) == MIN_INTERVAL
    assert env_interval({"CTXPACK_WATCH_INTERVAL": "9999"}) == MAX_INTERVAL


def test_env_interval_ignores_garbage():
    assert env_interval({"CTXPACK_WATCH_INTERVAL": "banana"}) == DEFAULT_INTERVAL
    assert env_interval({"CTXPACK_WATCH_INTERVAL": ""}) == DEFAULT_INTERVAL
