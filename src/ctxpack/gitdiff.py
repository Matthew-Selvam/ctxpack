"""Packing only what changed.

The dominant agent workflow is not "here is my repository" -- it is "review my
pull request" or "fix this bug", where ninety-five percent of the tree is
irrelevant.  Answering either question by packing the whole repository spends
the budget on context the model does not need and crowds out the handful of
files that do matter.  So this module asks git which paths differ and hands
them back as repo-relative posix strings, which is the same shape
:func:`ctxpack.walk.discover` produces.

Everything here is a thin, honest wrapper around ``git diff``.  There is no
libgit2, no third-party diff parser, and no reimplementation of rename
detection -- git already does all of that better than we would.

Four git behaviours drive the design, and all four look like bugs until you
read the manual:

* ``a..b`` is not ``a...b``.  Two dots compare two trees; three dots compare
  ``b`` against the *merge base* of ``a`` and ``b``, which is what a reviewer
  means when a branch has picked up upstream merges.  :class:`MergeBase` is how
  that distinction survives being flattened into a two-element tuple.
* A rename is one change with two paths.  ``--name-status`` reports
  ``R100 old new``; keeping both paths would ask the packer to read a file
  that no longer exists, so the old path is dropped and the new one kept.
* ``git diff`` cannot see untracked files at all.  That is fine for auditing
  history and actively harmful for "fix this bug", where the brand-new file is
  the whole point -- hence ``include_untracked``, off by default because it
  contradicts what the word "diff" means to most people.
* git speaks in *repo*-relative paths while ``ctxpack.walk`` walks in
  *discovery*-relative paths.  :func:`relativise` is the seam between the two,
  and it is the only function in this module that has to reason about both.

Deleted paths are deliberately absent from :func:`changed_files`: the file
cannot be read, so a bundle that claims to contain it would be a lie.
:func:`diffstat` still reports them, because "this was removed" is exactly what
a diffstat is for.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .errors import CtxpackError

__all__ = [
    "MergeBase",
    "changed_count",
    "changed_files",
    "diffstat",
    "is_git_repo",
    "parse_refs",
    "relativise",
    "repo_root",
]

#: Detection flags for every diff run here.  ``-C`` on its own never reports a
#: copy of an *unmodified* file, which is the case people actually make copies
#: in, so ``--find-copies-harder`` is not optional if the ``C`` status is worth
#: handling at all.  It does make git consider every file in the tree as a copy
#: source; on the diffs agents actually ask for that is cheaper than it sounds,
#: and getting the status right matters more than the millisecond.
DETECT_FLAGS: tuple[str, ...] = ("-M", "-C", "--find-copies-harder")

#: git's exit code for a usage error, which is how an old git reports that it
#: has never heard of ``--merge-base``.  Detected rather than guessed from a
#: version string, because the option's version history is not something to
#: hard-code.
USAGE_EXIT = 129

#: Seconds any single git invocation may take before ctxpack gives up on it.
#:
#: Overridable because the right value depends on the repository: a monorepo with
#: a cold object cache legitimately needs longer than a single-file project, and
#: a hard-coded ceiling would be wrong for one and useless for the other.
GIT_TIMEOUT: float = float(os.environ.get("CTXPACK_GIT_TIMEOUT", "30"))

#: Status letters whose entry names two paths: the source first, then the
#: destination.  Both are treated as "keep the destination".
_RENAME_LIKE = frozenset({"R", "C"})

#: Status letter for a path that no longer exists in the right-hand tree.
_DELETED = "D"


class MergeBase(str):
    """A base revision that means "compare from the merge base".

    Three-dot has to survive the trip out of :func:`parse_refs`, which is
    specified to return a two-element tuple, and three-dot is not something the
    two strings can express.  Tagging the base half with a ``str`` subclass
    smuggles the flag through without widening the signature.  It looks like a
    hack; it is the smallest honest way to keep both constraints, and ``base``
    stays usable anywhere a plain ``str`` is expected.

    Never construct one directly -- ``git diff --merge-base`` needs a revision
    on *both* sides, which :func:`parse_refs` and :func:`changed_files` enforce.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return f"MergeBase({str.__repr__(self)})"


@dataclass(frozen=True)
class _Entry:
    """One row of ``--name-status`` or ``--numstat``, after parsing.

    ``path`` is always the path a reader should look at: the destination of a
    rename or copy, the plain path otherwise.  ``old`` keeps the source around
    so a caller can be honest about provenance without asking git twice.

    ``status`` is empty for numstat rows, which report no status at all; only
    :func:`changed_files` cares, and only in order to skip deletions.
    """

    path: str
    status: str = ""
    old: str | None = None
    added: int | None = None
    removed: int | None = None


@dataclass(frozen=True)
class _Plan:
    """A resolved diff request: what to run, and against which work tree."""

    root: Path
    revs: tuple[str, ...] = ()
    staged: bool = False
    merge_base: bool = False
    include_untracked: bool = False


# ---------------------------------------------------------------------------
# process plumbing
# ---------------------------------------------------------------------------


def _git(root: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    """Run git and hand back the process whatever its exit code.

    ``shell=True`` is never an option here, and no argument is ever assembled
    into a string: a revision comes from the user's command line and has no
    business anywhere near a shell.

    ``errors="replace"`` is belt-and-braces for a non-UTF-8 locale.  With
    ``-z`` git hands back raw path bytes, and letting that raise
    UnicodeDecodeError would turn a recoverable odd filename into a crash.
    """
    try:
        return subprocess.run(
            ["git", *argv],
            cwd=str(root),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=GIT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        # Without this, a `git` that never returns -- a hung network filesystem,
        # a hook blocking on input, a pathological `--find-copies-harder` over a
        # huge tree -- leaves `ctxpack` waiting forever with no output at all.
        raise CtxpackError(
            f"git {' '.join(argv[:3])} did not finish within {GIT_TIMEOUT:.0f}s; "
            f"raise CTXPACK_GIT_TIMEOUT to wait longer"
        ) from None
    except (FileNotFoundError, NotADirectoryError, PermissionError) as exc:
        # The cwd is known to exist by this point, so the only remaining cause is
        # git itself being absent -- which is worth saying plainly, since nothing
        # else in ctxpack needs git and the user may not expect this.
        raise CtxpackError(f"could not run git (is it on PATH?): {exc}") from None


def _first_line(text: str) -> str:
    """Git's first complaint, on one line.

    The CLI renders a CtxpackError as a single line, and git's stderr is three
    lines of advice ending in a usage block.  The first line is the one that
    names the problem.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _failure(proc: subprocess.CompletedProcess[str]) -> str:
    detail = _first_line(proc.stderr) or _first_line(proc.stdout) or "no output"
    # proc.args is ["git", <subcommand>, ...]; showing the subcommand alone
    # keeps the message short while still saying which call failed.
    subcommand = proc.args[1] if len(proc.args) > 1 else "?"
    return f"git {subcommand} failed (exit {proc.returncode}): {detail}"


def _check(proc: subprocess.CompletedProcess[str]) -> str:
    """Return stdout, or raise with git's own words attached."""
    if proc.returncode != 0:
        raise CtxpackError(_failure(proc))
    return proc.stdout


# ---------------------------------------------------------------------------
# locating the work tree
# ---------------------------------------------------------------------------


def _as_cwd(root: str | Path) -> Path:
    """The directory git should run in to describe ``root``.

    ``walk.discover()`` accepts a single file as its root, so ``root`` may well
    be one -- and git cannot run in a file.  The parent is the right answer: a
    file's repository is its parent's, and this module only ever asks git about
    the tree, never about the file itself.
    """
    path = Path(root).expanduser()
    if not path.exists():
        raise CtxpackError(f"no such path: {root}")
    return (path if path.is_dir() else path.parent).resolve()


def _work_tree(root: str | Path) -> Path:
    """Resolve ``root`` and assert it sits inside a git work tree."""
    cwd = _as_cwd(root)
    proc = _git(cwd, "rev-parse", "--is-inside-work-tree")
    if proc.returncode != 0 or proc.stdout.strip() != "true":
        raise CtxpackError(
            f"{cwd} is not inside a git work tree; ctxpack reads a diff from "
            "git history, so run it inside a repository, or pack the directory "
            "without asking for a diff"
        )
    return cwd


def is_git_repo(root: str | Path) -> bool:
    """True when ``root`` is inside a work tree that can produce a diff.

    A bare repository is deliberately False.  It has history but no files to
    pack, so every function here would fail on it -- and ``rev-parse
    --is-inside-work-tree`` answers "false" for both a bare repo and the ``.git``
    directory itself, which is exactly the boundary wanted.
    """
    try:
        cwd = _as_cwd(root)
    except CtxpackError:
        return False
    proc = _git(cwd, "rev-parse", "--is-inside-work-tree")
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def repo_root(root: str | Path) -> Path:
    """The top of the work tree containing ``root``.

    Needed to translate git's repo-relative paths into the discovery-relative
    ones :func:`ctxpack.walk.discover` produces; see :func:`relativise`.
    """
    path = _work_tree(root)
    return Path(_check(_git(path, "rev-parse", "--show-toplevel")).strip())


# ---------------------------------------------------------------------------
# revision specs
# ---------------------------------------------------------------------------


def _check_ref(ref: str) -> str:
    """Reject a revision string that could mean something other than a revision.

    This is a sanity check, not validation: whether ``main`` exists is git's
    business, and git says it far better than we could.  What this catches is a
    string that would be *read* as an option or split into several tokens --
    argv is a list, so there is no shell to escape into, but a leading dash
    would still let a ref pick flags.
    """
    if not ref or ref != ref.strip() or ref.startswith("-"):
        raise CtxpackError(f"invalid revision {ref!r}")
    if any(not char.isprintable() or char.isspace() for char in ref):
        raise CtxpackError(f"invalid revision {ref!r}: whitespace is not allowed")
    return ref


def parse_refs(spec: str) -> tuple[str | None, str | None]:
    """Split a revision spec into ``(base, head)``.

    Understood forms, and what each one asks for:

    ==================================  ==================================
    ``""``                             no revisions; diff against HEAD
    ``"main"`` / ``"HEAD~3"``          ``git diff <rev>`` -- the work tree
                                       against that revision, which is what
                                       "changed since main" means in git
    ``"main.."`` / ``"..HEAD"``        the same, with the missing half spelled
                                       out explicitly
    ``"main..HEAD"``                   ``git diff main HEAD``
    ``"main...HEAD"``                  ``git diff --merge-base main HEAD``
    ==================================  ==================================

    ``""`` returns ``(None, None)`` rather than raising: an *absent* spec is not
    a user mistake, it is the default every entry point here falls back to.  A
    structurally empty range such as ``".."`` or ``"..."`` does raise, because
    it means the user typed something and got it wrong.

    Three dots are represented by returning a :class:`MergeBase` for ``base``,
    which still compares equal to the string it wraps.
    """
    text = spec.strip()
    if not text:
        return (None, None)

    if "..." in text:
        left, _, right = text.partition("...")
        three_dot = True
    elif ".." in text:
        left, _, right = text.partition("..")
        three_dot = False
    else:
        left, right, three_dot = text, "", False
    if ".." in left or ".." in right:
        raise CtxpackError(f"cannot read {spec!r}; use one range, as in main...HEAD")

    base = left.strip() or None
    head = right.strip() or None
    if base is None and head is None:
        raise CtxpackError(
            f"empty revision range in {spec!r}; use a revision, as in HEAD~3"
        )

    base = _check_ref(base) if base else None
    head = _check_ref(head) if head else None

    if three_dot:
        if base is None or head is None:
            raise CtxpackError(
                f"{spec!r} needs a revision on both sides, as in main...HEAD"
            )
        base = MergeBase(base)
    return (base, head)


def _plan(
    root: str | Path,
    base: str | None,
    head: str | None,
    *,
    staged: bool,
    uncommitted: bool,
    include_untracked: bool,
    merge_base: bool | None,
) -> _Plan:
    """Turn loose arguments into a command line git will accept."""
    work = _work_tree(root)

    if staged and uncommitted:
        raise CtxpackError(
            "choose either staged (index vs HEAD) or uncommitted (work tree vs "
            "HEAD), not both"
        )

    if merge_base is None:
        merge_base = isinstance(base, MergeBase)
    # Strip the marker so the string that reaches git is an ordinary revision.
    base = str(base) if base is not None else None

    if merge_base and (base is None or head is None):
        raise CtxpackError(
            "a merge-base diff needs two revisions, as in main...HEAD"
        )
    if uncommitted and head is not None:
        raise CtxpackError(
            "uncommitted diffs the work tree, which cannot also take a head "
            f"revision ({head!r}); drop it, or ask for base..head instead"
        )

    revs: list[str] = []
    if base is not None:
        revs.append(_check_ref(base))
    if head is not None:
        revs.append(_check_ref(head))
    if not revs and not staged:
        # Deliberately `git diff HEAD` rather than a bare `git diff`, which
        # compares the work tree to the *index* and silently omits everything
        # already staged. A default that quietly drops half the answer is worse
        # than a redundant flag.
        _verify_head(work)
        revs.append("HEAD")

    return _Plan(
        root=work,
        revs=tuple(revs),
        staged=staged,
        merge_base=merge_base,
        # Untracked paths exist only in the work tree. With a head revision on
        # the right there is no work tree in the comparison, so asking for them
        # would be asking git for something that cannot exist.
        include_untracked=include_untracked and len(revs) < 2,
    )


def _verify_head(root: Path) -> None:
    """Fail usefully on an unborn branch.

    ``git diff HEAD`` in a repository with no commits reports a confusing
    "ambiguous argument 'HEAD'", which reads like a ctxpack bug.
    """
    proc = _git(root, "rev-parse", "--verify", "--quiet", "HEAD")
    if proc.returncode != 0:
        raise CtxpackError(
            "this repository has no commits yet (unborn HEAD); there is no "
            "history to diff against until the first commit"
        )


# ---------------------------------------------------------------------------
# running the diff
# ---------------------------------------------------------------------------


def _diff_argv(plan: _Plan, kind: str) -> list[str]:
    argv = ["diff", kind, "-z", *DETECT_FLAGS]
    if plan.merge_base:
        argv.append("--merge-base")
    if plan.staged:
        argv.append("--cached")
    argv.extend(plan.revs)
    return argv


def _run_diff(plan: _Plan, kind: str) -> str:
    """Run the diff, degrading gracefully when git is too old for --merge-base."""
    argv = _diff_argv(plan, kind)
    proc = _git(plan.root, *argv)
    if proc.returncode == 0:
        return proc.stdout
    if plan.merge_base and proc.returncode == USAGE_EXIT:
        return _run_merge_base_fallback(plan, kind)
    raise CtxpackError(_failure(proc))


def _run_merge_base_fallback(plan: _Plan, kind: str) -> str:
    """Do a three-dot diff by hand for git versions without ``--merge-base``.

    ``git merge-base`` predates the option by a decade, so asking for it and then
    diffing merge-base(A, B) against B reproduces the three-dot result exactly.
    The failure cost is one extra subprocess, and only on git old enough to need
    it, which is not worth a cache to invalidate.
    """
    fork = _check(_git(plan.root, "merge-base", plan.revs[0], plan.revs[1])).strip()
    if plan.staged:
        # `--merge-base --cached A B` compares the *index* to the merge base, not
        # to B, so the faithful translation is the one-sided form.
        argv = ["diff", kind, "-z", *DETECT_FLAGS, "--cached", fork]
    else:
        argv = _diff_argv(_Plan(root=plan.root, revs=(fork, plan.revs[1])), kind)
    return _check(_git(plan.root, *argv))


def _untracked(root: Path) -> list[str]:
    """Paths git does not know about, honouring .gitignore.

    ``--exclude-standard`` is what makes this consistent with the rest of git:
    a build artefact that is gitignored is not a file the author was working on.
    """
    raw = _check(_git(root, "ls-files", "--others", "--exclude-standard", "-z"))
    return [part for part in raw.split("\0") if part]


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def _parse_name_status(text: str) -> list[_Entry]:
    """Parse ``--name-status -z``.

    ``-z`` is doing real work here, not decoration.  Without it git quotes any
    path containing a space, a quote or a non-ASCII byte (``"src/caf\\303\\251.py"``)
    and emits renames as one tab-joined line, which is unparseable without
    re-implementing git's brace elision.  With it, statuses and paths are
    separate NUL-terminated fields and a rename is simply two fields.
    """
    fields = text.split("\0")
    entries: list[_Entry] = []
    index = 0
    while index < len(fields):
        token = fields[index]
        index += 1
        if not token:
            continue
        status = token[0].upper()
        if status in _RENAME_LIKE:
            if index + 1 >= len(fields):
                break
            old, new = fields[index], fields[index + 1]
            index += 2
            entries.append(_Entry(path=new, status=status, old=old))
            continue
        if index >= len(fields):
            break
        entries.append(_Entry(path=fields[index], status=status))
        index += 1
    return entries


def _lines(field: str) -> int:
    """One numstat column. git writes ``-`` when it will not count a binary."""
    # 0 rather than -1: a negative number in a "removed" column reads as a
    # delta, and nothing here can honestly claim to know a binary's line count.
    return 0 if field == "-" else int(field or 0)


def _parse_numstat(text: str) -> list[_Entry]:
    """Parse ``--numstat -z``.

    The shape of a row depends on what changed.  A plain change is a single
    field holding ``3<TAB>1<TAB>src/main.py``.  A rename or copy is
    ``3<TAB>1<TAB>`` -- note the trailing tab, which ``split`` hands back as an
    empty third field and which is easy to mistake for the path -- followed by
    the source and destination as fields of their own.

    ``maxsplit=2`` matters either way: a filename may legally contain a tab, and
    splitting without a limit would truncate it.
    """
    fields = text.split("\0")
    entries: list[_Entry] = []
    index = 0
    while index < len(fields):
        token = fields[index]
        index += 1
        if not token:
            continue
        parts = token.split("\t", 2)
        old: str | None = None
        if len(parts) == 3 and parts[2]:
            path = parts[2]
        elif len(parts) >= 2 and index + 1 < len(fields):
            # The trailing-tab shape: the third field was empty, and the two
            # paths are the next fields of their own.
            old, path = fields[index], fields[index + 1]
            index += 2
        else:
            continue
        entries.append(
            _Entry(
                path=path, old=old, added=_lines(parts[0]), removed=_lines(parts[1])
            )
        )
    return entries


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def changed_files(
    root: str | Path,
    base: str | None = None,
    head: str | None = None,
    *,
    staged: bool = False,
    uncommitted: bool = False,
    include_untracked: bool = False,
    merge_base: bool | None = None,
) -> list[str]:
    """Repo-relative posix paths that differ, sorted and deduplicated.

    With no revisions this is ``git diff HEAD``: everything not yet committed,
    staged or not.  That is a deliberate departure from a bare ``git diff``,
    which would silently ignore whatever is already in the index.

    ``base`` and ``head`` take the output of :func:`parse_refs`.  Pass
    ``base="main", head="feature"`` for two trees, or
    ``merge_base=True`` for the three-dot comparison; a :class:`MergeBase`
    returned by :func:`parse_refs` sets that flag on its own.

    Deleted paths are excluded.  The file is gone, so there is nothing to pack,
    and returning it would produce a bundle that lies about its contents.

    The list is sorted, so it is stable across runs and safe to diff against
    another run's output.
    """
    plan = _plan(
        root,
        base,
        head,
        staged=staged,
        uncommitted=uncommitted,
        include_untracked=include_untracked,
        merge_base=merge_base,
    )
    entries = _parse_name_status(_run_diff(plan, "--name-status"))
    paths = {entry.path for entry in entries if entry.status != _DELETED}
    if plan.include_untracked:
        paths.update(_untracked(plan.root))
    return sorted(paths)


def changed_count(
    root: str | Path,
    base: str | None = None,
    head: str | None = None,
    *,
    staged: bool = False,
    uncommitted: bool = False,
    include_untracked: bool = False,
    merge_base: bool | None = None,
) -> int:
    """How many paths :func:`changed_files` would return."""
    return len(
        changed_files(
            root,
            base,
            head,
            staged=staged,
            uncommitted=uncommitted,
            include_untracked=include_untracked,
            merge_base=merge_base,
        )
    )


def diffstat(
    root: str | Path,
    base: str | None = None,
    head: str | None = None,
    *,
    staged: bool = False,
    uncommitted: bool = False,
    include_untracked: bool = False,
    merge_base: bool | None = None,
) -> dict[str, dict[str, int]]:
    """Added and removed line counts per path, keyed by path.

    Keys are sorted for the same reason :func:`changed_files` sorts.  Unlike
    :func:`changed_files`, deletions *are* included here: knowing that a path
    lost forty lines is the point of a diffstat, and there is no file to pack
    but there is still a change to report.

    Renames and copies are keyed by their destination path, matching
    :func:`changed_files`.  Binary files report ``0`` for both counts, because
    git refuses to count lines in them and inventing a number would be worse
    than admitting there is none.
    """
    plan = _plan(
        root,
        base,
        head,
        staged=staged,
        uncommitted=uncommitted,
        include_untracked=include_untracked,
        merge_base=merge_base,
    )
    counts: dict[str, dict[str, int]] = {}
    for entry in _parse_numstat(_run_diff(plan, "--numstat")):
        counts[entry.path] = {
            "added": entry.added or 0,
            "removed": entry.removed or 0,
        }
    if plan.include_untracked:
        for path in _untracked(plan.root):
            # setdefault, never overwrite: a path can be both untracked in the
            # work tree and present in the compared trees.
            counts.setdefault(path, {"added": 0, "removed": 0})
    return {path: counts[path] for path in sorted(counts)}


def relativise(
    paths: list[str],
    repo_root: str | Path,
    discovery_root: str | Path,
) -> list[str]:
    """Rewrite repo-relative git paths as discovery-root-relative ones.

    git reports paths relative to the top of the work tree; ``ctxpack.walk``
    reports them relative to whatever directory it was pointed at.  When those
    are the same directory this is the identity, which is the common case and
    the one worth being boring about.

    Paths outside ``discovery_root`` are dropped, so the result can be used
    directly to select from a :class:`~ctxpack.walk.Discovery`.  Order is
    preserved and duplicates removed; an already-sorted input stays sorted.

    Both roots are resolved before comparing.  On macOS ``/tmp`` and
    ``/private/tmp`` are the same directory with two names, and ``discover``
    resolves what it is given, so skipping this step would silently drop every
    path.

    The ``repo_root`` parameter deliberately shadows the function of that name:
    the two arguments are the caller's paths, not something to re-derive.
    """
    repo = Path(repo_root).expanduser().resolve()
    discovery = Path(discovery_root).expanduser().resolve()

    # A discovery root may be a single file -- walk.discover() accepts one --
    # so the match can be exact rather than a prefix.
    exact: str | None = None
    prefix = ""
    if discovery == repo:
        prefix = ""
    elif repo in discovery.parents:
        relative = discovery.relative_to(repo).as_posix()
        if discovery.is_dir():
            prefix = relative + "/"
        else:
            exact = relative
    else:
        # The discovery root is outside the work tree, so nothing git reported
        # can be inside it.
        return []

    out: list[str] = []
    seen: set[str] = set()
    for path in paths:
        candidate = path.strip()
        if exact is not None:
            if candidate != exact or candidate in seen:
                continue
            seen.add(candidate)
            out.append(candidate)
        elif candidate.startswith(prefix):
            trimmed = candidate[len(prefix) :]
            if trimmed and trimmed not in seen:
                seen.add(trimmed)
                out.append(trimmed)
    return out
