"""Re-packing on change, without a file-watching dependency.

The loop this exists for: you are working on a repository with an agent open
against it, you change a file, and you want fresh context without re-typing a
command. Rebuilding a bundle is cheap enough to do on every save.

Why polling rather than inotify/FSEvents/ReadDirectoryChangesW:

* **No dependency.** Every cross-platform watcher for Python is a third-party
  package, and this project has none.
* **Cross-platform for free.** One code path instead of three backends with
  subtly different semantics.
* **The workload does not need events.** A pack is tens to hundreds of
  milliseconds and re-reads the whole tree anyway, so there is nothing to gain
  from being told about the *first* of a burst of writes. What matters is that
  the tree is quiet again, which is a question polling answers directly.

The cost is latency and syscalls: one ``stat`` per file per interval, versus
zero when idle but an event subscription. For a repository of a few thousand
files at a 0.5s interval that is a few thousand stats a second -- cheap, but not
free, so the interval is configurable and the default is not aggressive.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DEFAULT_INTERVAL",
    "DEFAULT_SETTLE",
    "Change",
    "diff_snapshots",
    "run_watch",
    "snapshot",
]

#: Seconds between polls. Not lower: a pack already re-reads everything, and
#: stat-ing a few thousand files twice a second is noticeable for no benefit.
DEFAULT_INTERVAL = 0.5

#: Seconds the tree must be unchanged before a rebuild fires. An editor that
#: writes a temp file and renames it, or a formatter plus a save, produces a
#: burst of writes; without this the bundle is rebuilt three times for one edit.
DEFAULT_SETTLE = 0.4


@dataclass(frozen=True)
class Change:
    """One path whose state differs between two snapshots."""

    path: str
    kind: str  # "modified" | "added" | "deleted"

    def __str__(self) -> str:
        return f"{self.kind} {self.path}"


#: (mtime_ns, size). Size is included because mtime granularity is not always
#: fine enough -- some filesystems round timestamps coarsely enough that a fast
#: write keeps the same mtime, and a rename-over keeps it too.
Stamp = tuple[int, int]


def snapshot(paths: Iterable[str], *, root: Path | None = None) -> dict[str, Stamp]:
    """Record (mtime, size) for each path that currently exists."""
    out: dict[str, Stamp] = {}
    for rel in paths:
        target = (root / rel) if root is not None else Path(rel)
        try:
            stat = target.stat()
        except OSError:
            # Deleted between listing and stat, or unreadable. Either way it is
            # not in the snapshot, and diffing will report the deletion.
            continue
        out[rel] = (stat.st_mtime_ns, stat.st_size)
    return out


def diff_snapshots(
    before: dict[str, Stamp], after: dict[str, Stamp]
) -> list[Change]:
    """What changed between two snapshots, in a deterministic order.

    Sorted by path so two runs over the same edits report identically -- a
    watch loop that prints changes in hash order is maddening to read.
    """
    changes: list[Change] = []
    for path in sorted(after):
        if path not in before:
            changes.append(Change(path, "added"))
        elif after[path] != before[path]:
            changes.append(Change(path, "modified"))
    changes.extend(
        Change(path, "deleted") for path in sorted(before) if path not in after
    )
    return changes


def run_watch(
    paths: Sequence[str],
    repack: Callable[[], object],
    *,
    root: Path | None = None,
    interval: float = DEFAULT_INTERVAL,
    settle: float = DEFAULT_SETTLE,
    iterations: int | None = None,
    max_settle: float = 5.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    on_change: Callable[[list[Change]], None] | None = None,
    on_error: Callable[[BaseException], None] | None = None,
) -> int:
    """Poll ``paths``, rebuilding via ``repack`` whenever the tree settles.

    ``iterations`` bounds the number of poll cycles; ``None`` runs until
    interrupted, which is what the CLI does. It exists so the tests can drive
    the loop deterministically instead of sleeping.

    ``clock`` and ``sleep`` are injectable for the same reason -- the settle
    logic is the interesting part and it must be testable without wall-clock
    time. Injected callables also mean a test can make the tree "change" by
    mutating the clock, with no filesystem at all.

    ``max_settle`` bounds the debounce. Without it, a repository that keeps
    changing never goes quiet, the settle deadline keeps being pushed out, and
    the rebuild never happens at all -- a format-on-save loop would leave the
    watcher spinning indefinitely and never once produce a bundle. Better to
    rebuild on a moving target than never to rebuild.

    A rebuild that raises does not stop the watch. A syntax error in a file
    being edited is the *normal* state of a repository mid-edit, and a watcher
    that dies on the first one is useless exactly when it is needed.
    """
    current = snapshot(paths, root=root)
    cycles = 0

    while iterations is None or cycles < iterations:
        cycles += 1
        sleep(interval)
        after = snapshot(paths, root=root)
        changes = diff_snapshots(current, after)
        if not changes:
            continue

        # Wait for the tree to go quiet. An editor writing several files, or a
        # formatter plus a save, produces a burst; rebuilding on each one wastes
        # the very CPU the debounce is meant to protect.
        started = clock()
        deadline = started + settle
        while clock() < deadline and clock() - started < max_settle:
            sleep(min(interval, max(0.0, deadline - clock())))
            settled = snapshot(paths, root=root)
            if settled != after:
                after = settled
                changes = diff_snapshots(current, after)
                deadline = clock() + settle

        current = after
        if on_change is not None:
            on_change(changes)
        try:
            repack()
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            if on_error is None:
                raise
            on_error(exc)

    return 0


def watch_paths(
    discovery_files: Iterable, *, root: Path | None = None
) -> list[str]:
    """The paths worth watching, taken from a :class:`~ctxpack.walk.Discovery`.

    Watching *every* file on disk would be both slow and wrong: a change to a
    file ctxpack would ignore should not trigger a rebuild. So the watch set is
    the discovered set, and newly created files are picked up on the next
    re-discovery by the repack callback rather than by the watcher.
    """
    return [candidate.path for candidate in discovery_files]


def mtime_of(path: str, *, root: Path | None = None) -> float:
    """Seconds-resolution mtime, or ``-1.0`` when the file is gone.

    Exposed for callers that want a cheap freshness check rather than the whole
    snapshot machinery.
    """
    target = (root / path) if root is not None else Path(path)
    try:
        return target.stat().st_mtime
    except OSError:
        return -1.0


MIN_INTERVAL = 0.05
MAX_INTERVAL = 60.0


def env_interval(
    env: Mapping[str, str] | None = None, default: float = DEFAULT_INTERVAL
) -> float:
    """Interval from ``CTXPACK_WATCH_INTERVAL``, clamped to something sane.

    ``env`` is injectable so the clamping is testable without mutating the real
    environment, which is a shared resource and a source of order-dependent
    failures.
    """
    raw = (os.environ if env is None else env).get("CTXPACK_WATCH_INTERVAL")
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    # Below 50ms the poll is indistinguishable from a spin loop and does real
    # work for nothing; above 60s it stops being a watch.
    return min(max(value, MIN_INTERVAL), MAX_INTERVAL)

