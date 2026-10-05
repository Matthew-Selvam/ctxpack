"""Repository settings, so the same packing policy applies to everyone.

Typing ``ctxpack . -b 60000 --reach-weight 4 --diff main...HEAD -f xml`` is
mildly unpleasant once, and unmaintainable when four people review the same
change with the same context. Worse, the *reason* those flags were chosen --
"60k because the review model's window is 64k and we want headroom for the
prompt" -- lives in somebody's shell history, where it does nothing.

So the settings move into a file that can be committed next to the code they
describe. That is the whole rationale; everything below is implementation.

**Why TOML, with JSON as the 3.10 escape hatch.** TOML is the only common
config format that carries comments, and comments are load-bearing here: the
file is also where the reasoning gets recorded. JSON cannot do that, and
``tomli`` would be a runtime dependency in a project whose selling point is
having none. Python 3.11 added ``tomllib`` to the stdlib, but this package
supports 3.10, where no stdlib TOML parser exists.

Rather than hand-rolling a TOML subset (a language spec is a bad thing to
approximate, and a parser that accepts ``budegt = 1`` because it skipped an
unfamiliar directive is worse than no parser), on 3.10 a TOML file is
rejected with a message that names the file, says which interpreter it needs,
and points at the JSON alternative. ``.ctxpack.json`` is always accepted, so a
3.10 user is never stuck -- they just lose the ability to write down why.

**Where relative paths point.** ``exclude = ["tests/**"]`` in a config file is
about the *repository*, not about wherever the command happened to run. Every
relative path in a config is therefore interpreted against the directory
containing that config file, never the process CWD -- otherwise
``cd src && ctxpack .`` quietly ignores the excludes, and the bug only shows
up for people who ran the command from a subdirectory. Because the packed
root is not known until later, :class:`Config` stores the patterns verbatim
and :func:`resolve_patterns` rebases them once the root is chosen.

**Which file wins.** Exactly one: explicit CLI flag, then config file, then
built-in default. See :func:`merge` for how "the user passed this" is
established, which is harder than it looks.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from difflib import get_close_matches
from pathlib import Path
from typing import Any, NoReturn

from .errors import CtxpackError
from .pack import MODES
from .render import FORMATS
from .tokens import KNOWN_ENCODINGS

__all__ = [
    "CONFIG_FILENAMES",
    "DEFAULTS",
    "JSON_FILENAMES",
    "TOML_FILENAMES",
    "UNSET",
    "Config",
    "Unset",
    "describe",
    "discover_config",
    "load_config",
    "load_or_empty",
    "merge",
    "passed_dests",
    "resolve_patterns",
    "unknown_keys_warning",
]


#: Config filenames in priority order. TOML first because it is the primary
#: format; JSON exists so a 3.10 user is never blocked by the missing stdlib
#: parser. Within one directory the earlier name wins, so a repo cannot carry
#: two configs whose relative precedence depends on which name a contributor
#: happened to type.
TOML_FILENAMES = ("ctxpack.toml", ".ctxpack.toml")
JSON_FILENAMES = ("ctxpack.json", ".ctxpack.json")
CONFIG_FILENAMES = TOML_FILENAMES + JSON_FILENAMES

#: Tables that may hold the settings. A flat top-level document is also
#: accepted, because that is what JSON has to be and because a one-key file
#: should not need a table header to be understood.
_TABLES = ("ctxpack", "tool")

_MANIFEST_MODES = ("full", "paths", "none")
_EXACT_MODES = ("auto", "exact", "never")

#: Values argparse fills in when the user says nothing, mirrored here so
#: :func:`merge` can fill the same gaps without importing the parser. Kept in
#: step with ``build_parser()`` by hand; only the settings this module models
#: are listed, because only those can come from a config file.
DEFAULTS: dict[str, Any] = {
    "budget": 32000,
    "format": "markdown",
    "mode": "balanced",
    "manifest": "full",
    "encoding": "o200k_base",
    "exact": "auto",
    "reach_weight": 0.0,
    "outline": False,
    "factor_shared": False,
    "diff": None,
    "exclude": (),
    "include": (),
    "per_dir_frac": 0.40,
    "dedupe_threshold": 0.85,
}

#: Directory entries that end an upward search. A ``.git`` entry marks the top
#: of a work tree; going above it would let a config in a monorepo parent, or
#: worse in ``$HOME``, silently change how a repository packs.
_STOP_ENTRIES = (".git",)

#: Similarity cut-off for "did you mean" on an unrecognised key.
_SUGGEST_CUTOFF = 0.7


class Unset:
    """Marker for "no value at any layer", distinct from ``None``.

    ``None`` is a real value for ``--diff``. ``UNSET`` is the absence of one,
    and it is what makes it possible to tell ``--diff main`` from silence.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False


#: The single shared instance. ``argparse`` accepts it as a default, which is
#: the fully sound way to detect explicit flags -- see :func:`merge`.
UNSET = Unset()


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


def _fail(path: Path, key: str, message: str) -> NoReturn:
    """Raise the one-line, exit-2 error every bad config file deserves.

    Always names the file and the key. A user who mistypes ``budegt`` or types
    ``budget = "60000"`` should not have to guess which of a hundred numbers in
    a build system is being complained about.
    """
    raise CtxpackError(f"{path}: key '{key}': {message}")


def _type_name(value: Any) -> str:
    return type(value).__name__


def _check_enum(path: Path, key: str, value: Any, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        _fail(path, key, f"must be one of {', '.join(allowed)}; got {_type_name(value)}")
    if value not in allowed:
        _fail(path, key, f"{value!r} is not one of {', '.join(allowed)}")
    return value


def _check_int(path: Path, key: str, value: Any, *, minimum: int | None = None) -> int:
    # ``bool`` is a subclass of ``int``; ``outline = 1`` is a type error, not a
    # request for the integer one.
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, key, f"must be a whole number; got {value!r} ({_type_name(value)})")
    if minimum is not None and value < minimum:
        _fail(path, key, f"must be at least {minimum}; got {value}")
    return value


def _check_fraction(path: Path, key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, key, f"must be a number; got {value!r} ({_type_name(value)})")
    number = float(value)
    if not 0.0 < number <= 1.0:
        _fail(path, key, f"must be a fraction in (0, 1]; got {value}")
    return number


def _check_number(path: Path, key: str, value: Any, *, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, key, f"must be a number; got {value!r} ({_type_name(value)})")
    number = float(value)
    if number < minimum:
        _fail(path, key, f"must be at least {minimum}; got {value}")
    return number


def _check_bool(path: Path, key: str, value: Any) -> bool:
    if not isinstance(value, bool):
        _fail(path, key, f"must be true or false; got {value!r} ({_type_name(value)})")
    return value


def _check_string_list(path: Path, key: str, value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        _fail(path, key, f"must be a list of strings; got {value!r} ({_type_name(value)})")
    for item in value:
        if not isinstance(item, str):
            _fail(path, key, f"must contain only strings; got {item!r} ({_type_name(item)})")
    return tuple(value)


def _check_str(path: Path, key: str, value: Any) -> str:
    if not isinstance(value, str):
        _fail(path, key, f"must be a string; got {value!r} ({_type_name(value)})")
    return value


#: key -> validator. One table so ``Config`` fields, validation, and the
#: unknown-key check can never drift apart.
_VALIDATORS = {
    "budget": lambda p, k, v: _check_int(p, k, v, minimum=1),
    "format": lambda p, k, v: _check_enum(p, k, v, FORMATS),
    "mode": lambda p, k, v: _check_enum(p, k, v, MODES),
    "manifest": lambda p, k, v: _check_enum(p, k, v, _MANIFEST_MODES),
    "encoding": lambda p, k, v: _check_enum(p, k, v, KNOWN_ENCODINGS),
    "exact": lambda p, k, v: _check_enum(p, k, v, _EXACT_MODES),
    "reach_weight": lambda p, k, v: _check_number(p, k, v, minimum=0.0),
    "outline": _check_bool,
    "factor_shared": _check_bool,
    "diff": _check_str,
    "exclude": _check_string_list,
    "include": _check_string_list,
    "per_dir_frac": _check_fraction,
    "dedupe_threshold": _check_fraction,
}

#: Keys whose values are lists. These are the only settings where a merge has a
#: choice to make, and it is documented in :func:`merge`.
_LIST_KEYS = ("exclude", "include")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """One validated config file.

    Every setting defaults to ``None`` -- meaning *not specified*. That is the
    whole trick behind precedence: a config file that sets only ``budget`` must
    be able to say "I have no opinion about ``mode``" in a way that
    :func:`merge` can act on, and ``None`` is the only value free for that
    (``--diff`` legitimately takes ``None`` as a meaning, so no setting uses a
    bare ``None`` to mean absent).

    ``source`` is the file this came from, or ``None`` for the empty config
    used when no file was found. ``unknown_keys`` holds unrecognised keys
    rather than discarding them: a typo is far more likely than a deliberate
    setting this module does not model, and silently ignoring it means the
    user's ``budegt = 9000`` has no effect and no explanation.
    """

    budget: int | None = None
    format: str | None = None
    mode: str | None = None
    manifest: str | None = None
    encoding: str | None = None
    exact: str | None = None
    reach_weight: float | None = None
    outline: bool | None = None
    factor_shared: bool | None = None
    diff: str | None = None
    exclude: tuple[str, ...] = ()
    include: tuple[str, ...] = ()
    per_dir_frac: float | None = None
    dedupe_threshold: float | None = None
    source: Path | None = None
    unknown_keys: tuple[str, ...] = ()

    @property
    def base(self) -> Path | None:
        """Directory that relative paths in this file are relative to."""
        return self.source.parent if self.source is not None else None

    def get(self, key: str) -> Any:
        """The configured value for ``key``, or ``None`` if unset."""
        return getattr(self, key, None)

    def is_empty(self) -> bool:
        """True when the file carried no settings at all."""
        return not self.specifications()

    def specifications(self) -> tuple[str, ...]:
        """Names of the settings this file actually set, in field order."""
        return tuple(
            f.name
            for f in fields(self)
            if f.name not in ("source", "unknown_keys")
            and getattr(self, f.name) not in (None, (), False)
        )

    def with_patterns(self, root: Path | None) -> Config:
        """Copy with ``exclude`` / ``include`` rebased onto ``root``.

        Convenience for callers that have the root in hand at merge time; see
        :func:`resolve_patterns` for what "rebased" means.
        """
        if root is None or self.source is None:
            return self
        return replace(
            self,
            exclude=resolve_patterns(self.exclude, base=self.source.parent, root=root),
            include=resolve_patterns(self.include, base=self.source.parent, root=root),
        )


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def _first_existing(directory: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def discover_config(start: Path, names: tuple[str, ...] = CONFIG_FILENAMES) -> Path | None:
    """Find the nearest config at or above ``start``, or ``None``.

    ``start`` may be a file (its parent is searched) or a directory. The walk
    goes up one directory at a time and stops at whichever comes first:

    * a directory holding a config file -- which is checked *before* the stop
      conditions, so a config in the repository root is found even when the
      root is also where the search would have stopped;
    * a directory containing a ``.git`` entry;
    * the filesystem root.

    Stopping at ``.git`` rather than always going to ``/`` is a deliberate
    narrowing. A team commits ``ctxpack.toml`` so that the policy travels with
    the code; if the search escaped the work tree, the same command would pick
    up a different policy depending on where the checkout happens to live,
    including a file in ``$HOME`` that no commit ever mentions. Truncated
    upward search is predictable -- it follows the repository, which is the
    unit the file belongs to.
    """
    origin = Path(start).expanduser()
    if origin.is_file():
        origin = origin.parent
    if not origin.is_absolute():
        origin = Path.cwd() / origin
    origin = Path(os.path.normpath(origin))

    for directory in (origin, *origin.parents):
        found = _first_existing(directory, names)
        if found is not None:
            return found
        if any((directory / entry).exists() for entry in _STOP_ENTRIES):
            return None
    return None


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def _tomllib() -> Any:
    """The stdlib TOML reader, or ``None`` on 3.10.

    Returned through a function so a test can simulate the 3.10 interpreter on
    a newer one; the real check is a plain import guard.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        return None
    return tomllib


def _read_document(path: Path) -> Mapping[str, Any]:
    """Parse the file into a mapping, or explain precisely why we cannot."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise CtxpackError(f"{path}: no such config file") from None
    except IsADirectoryError:
        raise CtxpackError(f"{path}: is a directory, not a config file") from None
    except UnicodeDecodeError as exc:
        raise CtxpackError(f"{path}: is not UTF-8 text ({exc.reason})") from None
    except OSError as exc:
        raise CtxpackError(f"{path}: cannot be read ({exc.strerror or exc})") from None

    if path.suffix == ".json":
        return _parse_json(path, text)

    tomllib = _tomllib()
    if tomllib is None:
        raise CtxpackError(
            f"{path}: TOML config files need Python 3.11 or newer, and this is "
            f"Python {sys.version_info.major}.{sys.version_info.minor}. Either "
            f"run ctxpack on a newer interpreter, or use "
            f"{path.with_suffix('.json').name} instead -- JSON works on 3.10, "
            "at the cost of not being able to record why a setting was chosen."
        )
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise CtxpackError(f"{path}: malformed TOML: {exc}") from None


def _parse_json(path: Path, text: str) -> Mapping[str, Any]:
    if not text.strip():
        return {}
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CtxpackError(
            f"{path}: malformed JSON: {exc.msg} (line {exc.lineno} column {exc.colno})"
        ) from None
    if not isinstance(document, dict):
        raise CtxpackError(
            f"{path}: must be a JSON object at the top level; got "
            f"{_type_name(document)}"
        )
    return document


def _settings_section(path: Path, document: Mapping[str, Any]) -> Mapping[str, Any]:
    """Pull our keys out of the document.

    Accepts a bare settings document, or one nested under ``[ctxpack]`` or
    ``[tool.ctxpack]`` -- the nesting is what lets a config file coexist with
    other tools' tables, and it is required for TOML only when the file also
    holds something else.
    """
    for table in _TABLES:
        if table not in document:
            continue
        section = document[table]
        if not isinstance(section, dict):
            _fail(path, table, f"must be a table of settings; got {_type_name(section)}")
        if table == "tool":
            return _settings_section(path, section)
        return section

    nested = set(_TABLES) & set(document)
    if nested:
        # A table was requested but every known table is shadowed by a scalar.
        raise CtxpackError(
            f"{path}: key '{sorted(nested)[0]}': must be a table of settings"
        )
    return document


def _suggest(key: str) -> str:
    close = get_close_matches(key, tuple(_VALIDATORS), n=1, cutoff=_SUGGEST_CUTOFF)
    return f" (did you mean '{close[0]}'?)" if close else ""


def load_config(path: Path) -> Config:
    """Parse and validate one config file.

    Raises :class:`CtxpackError` -- naming the file, and the key where there
    is one -- for anything malformed. A file that is empty, or that contains
    nothing but comments, is legal and yields a config with no settings; TOML's
    own rules make that free, and a placeholder file should not need a dummy
    key to exist.
    """
    path = Path(path).expanduser()
    document = _read_document(path)
    section = _settings_section(path, document)

    values: dict[str, Any] = {}
    unknown: list[str] = []
    for key, raw in section.items():
        validator = _VALIDATORS.get(key)
        if validator is None:
            unknown.append(key)
            continue
        values[key] = validator(path, key, raw)

    return Config(
        source=path,
        unknown_keys=tuple(sorted(unknown)),
        **values,
    )


def load_or_empty(path: Path | None) -> Config:
    """:func:`load_config`, or the empty config when there is no file.

    The "no config anywhere" path is the common one, so it must not raise:
    absence of configuration is not an error, it is the default state.
    """
    return load_config(path) if path is not None else Config()


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def resolve_patterns(
    patterns: tuple[str, ...], *, base: Path, root: Path
) -> tuple[str, ...]:
    """Rewrite ``patterns`` -- written relative to ``base`` -- for ``root``.

    Packing matches patterns against paths relative to the packed root, so a
    pattern authored against the config's directory only lines up when the two
    are the same. When they are not, the pattern is prefixed with the relative
    path between them: a config at the repo root and a root of ``src`` turns
    ``tests/**`` into ``../tests/**``, which matches nothing, correctly, since
    ``tests`` is not inside ``src``.

    Absolute patterns are made relative to ``root`` for the same reason. No
    ``os.getcwd()`` appears anywhere in this function: the process CWD is
    exactly the thing that must not influence the result.
    """
    if not patterns:
        return ()

    base = Path(os.path.normpath(Path(base).expanduser()))
    root = Path(os.path.normpath(Path(root).expanduser()))

    try:
        prefix = Path(os.path.relpath(base, root)).as_posix()
    except ValueError:  # different drives on Windows
        return patterns
    if prefix in (".", ""):
        return tuple(patterns)

    out = []
    for pattern in patterns:
        text = pattern.strip()
        if not text:
            out.append(pattern)
        elif Path(text).is_absolute():
            out.append(Path(os.path.relpath(text, root)).as_posix())
        else:
            out.append(f"{prefix}/{text.lstrip('./')}")
    return tuple(out)


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return vars(value)


def _looks_unset(value: Any) -> bool:
    if isinstance(value, Unset):
        return True
    if value is None:
        return True
    # Empty list/tuple is how ``action="append"`` reports "no occurrences".
    return isinstance(value, (list, tuple)) and not value


def passed_dests(parser: Any, argv: Iterable[str]) -> frozenset[str]:
    """Names of the argparse destinations actually present in ``argv``.

    The sound way to answer "did the user type this flag?", for parsers that
    were not built with ``default=UNSET``. It reads ``option_strings`` off each
    action -- a documented part of ``argparse.Action`` -- so it survives parser
    subclasses and does not depend on how a given flag is implemented
    (``store_true``, ``append``, ``type=``).

    Handles ``--opt value``, ``--opt=value``, ``-b 60000``, ``-b60000``,
    clustered short flags, unique long-option abbreviations, and stops at a
    bare ``--``.
    """
    actions = _all_actions(parser)
    dests: set[str] = set()
    tokens = list(argv)
    index = 0

    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            break
        if token.startswith("--"):
            head = token.split("=", 1)[0]
            action = actions.get(head) or _abbreviated(actions, head)
            if action is not None:
                dests.add(action.dest)
            index += 1
            continue
        if token.startswith("-") and token != "-":
            cluster = token[1:]
            consumed = False
            while cluster:
                action = actions.get(f"-{cluster[0]}")
                if action is None:
                    break
                dests.add(action.dest)
                if action.nargs == 0:
                    cluster = cluster[1:]
                    continue
                # A flag that takes a value either has it attached (``-b1000``)
                # or swallows the next token.
                consumed = len(cluster) > 1
                cluster = ""
            index += 2 if consumed else 1
            continue
        index += 1  # positional

    return frozenset(dests)


def _abbreviated(actions: Mapping[str, Any], prefix: str) -> Any:
    matches = [
        action
        for option, action in actions.items()
        if option.startswith(prefix) and len(option) > len(prefix)
    ]
    return matches[0] if len(matches) == 1 else None


def _all_actions(parser: Any) -> dict[str, Any]:
    """Option string -> action, for ``parser`` and every subparser beneath it.

    ctxpack's flags live on ``pack``, not on the top-level parser, so a lookup
    that ignored subparsers would find nothing but ``--version``.
    """
    table: dict[str, Any] = dict(getattr(parser, "_option_string_actions", {}))
    for action in getattr(parser, "_actions", ()):
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict):  # a _SubParsersAction
            for sub in choices.values():
                for option, sub_action in _all_actions(sub).items():
                    table.setdefault(option, sub_action)
    return table


def merge(
    explicit: Any,
    config: Config | None = None,
    defaults: Mapping[str, Any] | None = None,
    *,
    passed: Iterable[str] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Combine explicit flags, a config file, and built-in defaults.

    Precedence, highest first:

    1. **explicit** -- a flag the user actually typed;
    2. **config** -- a setting the file specifies;
    3. **defaults** -- argparse's own defaults, or ``DEFAULTS``.

    **Telling a typed flag from a filled-in default.** argparse gives no way
    to ask this afterwards: every unspecified flag already carries a value, so
    ``args.budget`` is 32000 whether the user wrote ``-b 32000`` or said
    nothing. Comparing against the parser's defaults therefore cannot work, and
    the failure is not exotic -- ``-m balanced``, ``-f markdown`` or
    ``--manifest full`` are all values a user may genuinely want *to* override a
    config file with, and all of them equal the built-in default. Two sound
    mechanisms are supported, in order of preference:

    * Build the parser with ``default=UNSET`` for configurable flags and fill
      in the real defaults after merging. ``UNSET`` is detected here directly,
      so nothing has to be guessed. This is the recommended wiring.
    * Pass ``passed`` -- the output of :func:`passed_dests` -- so "was it
      typed" comes from ``argv`` rather than from the parsed values.

    With neither, ``explicit`` is compared against ``defaults``. That is a
    heuristic and is treated as one: it cannot see ``-b 32000`` typed over a
    config file of ``budget = 60000``, and the docstring says so rather than
    pretending otherwise.

    **Lists accumulate, they do not replace.** ``exclude`` and ``include``
    combine as config first, then explicit. The alternatives are worse: a
    config that lists build output and a command line that adds one lockfile
    would each lose half the intent, and the usual mental model of an exclude
    list is a shared set nobody owns alone. Order is config-then-explicit and
    duplicates are removed, so the result is stable across runs. There is no
    way to *clear* a list from the command line; a config file that excludes
    something a user wants back needs the config edited, which is a deliberate
    act rather than an accident of flag order.

    ``root``, when given, rebases the config's relative patterns onto the
    packed root before they are combined -- see :func:`resolve_patterns`.

    Returns a plain ``dict``; ``vars(args).update(...)`` applies it.
    """
    explicit_map = _as_mapping(explicit)
    defaults_map = dict(DEFAULTS if defaults is None else defaults)
    config = config if config is not None else Config()

    if root is not None and config.source is not None:
        config = config.with_patterns(root)

    # ``None`` means "caller could not tell me", which selects the heuristic.
    typed: frozenset[str] | None = None if passed is None else frozenset(passed)

    resolved: dict[str, Any] = {}
    for key in defaults_map:
        value = explicit_map.get(key, UNSET)

        # Lists accumulate and never take the "explicit wins" branch, so the
        # explicit check is skipped for them entirely.
        if key in _LIST_KEYS:
            from_cli = tuple(value) if isinstance(value, (list, tuple)) else ()
            resolved[key] = _dedupe(tuple(getattr(config, key, ())) + from_cli)
            continue

        if _is_explicit(key, value, typed, defaults_map[key], explicit_map):
            resolved[key] = value
            continue

        from_config = getattr(config, key, None)
        if from_config is None:
            resolved[key] = defaults_map[key]
        else:
            resolved[key] = from_config

    return resolved


def _is_explicit(
    key: str,
    value: Any,
    typed: frozenset[str] | None,
    default: Any,
    explicit_map: Mapping[str, Any],
) -> bool:
    """Whether ``explicit_map[key]`` was something the user actually asked for."""
    if _looks_unset(value):
        # ``UNSET`` is the definitive "no default was filled in". ``None`` is
        # likewise never a typed value: ``--diff`` takes a spec string. An empty
        # list is how ``action="append"`` reports zero occurrences, which is
        # worth special-casing because ``[] != ()`` -- argparse's default is a
        # list, and a tuple default would compare unequal and look like input.
        return False
    if typed is not None:
        return key in typed
    if key not in explicit_map:
        return False
    # Fallback heuristic. Documented as unable to see a typed value that
    # happens to equal the built-in default.
    return value != default


def _dedupe(items: Iterable[Any]) -> tuple[Any, ...]:
    """Order-preserving de-duplication, so merged output is deterministic."""
    seen: set[Any] = set()
    out: list[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def _show(value: Any) -> str:
    if isinstance(value, tuple):
        return ", ".join(str(item) for item in value) if value else "(none)"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "(none)"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def describe(config: Config | None = None, defaults: Mapping[str, Any] | None = None) -> str:
    """One line per effective setting, for ``--show-config``.

    Reports the value in force, tagged with where it came from, so the output
    answers the question a config file always raises: *why did my command just
    do that?* A setting the file did not mention shows its default and says so
    rather than being omitted, because an absent line is indistinguishable from
    a forgotten one.
    """
    config = config if config is not None else Config()
    defaults_map = dict(DEFAULTS if defaults is None else defaults)

    lines = [
        f"{'config':<20}{config.source if config.source is not None else '(none found)'}"
    ]
    for key, default in defaults_map.items():
        configured = getattr(config, key, None)
        # A list is "set" when it has entries; an empty list and ``False``
        # both mean the file said nothing about it.
        is_set = bool(configured) if key in _LIST_KEYS else configured is not None
        if is_set:
            lines.append(f"{key:<20}{_show(configured)}  (config)")
        else:
            lines.append(f"{key:<20}{_show(default)}  (default)")

    if config.unknown_keys:
        lines.append("")
        lines.append(
            f"unknown key(s) in {config.source}: "
            + ", ".join(
                f"'{key}'{_suggest(key)}" for key in config.unknown_keys
            )
        )
    return "\n".join(lines)


def unknown_keys_warning(config: Config | None) -> str | None:
    """The warning to print for unrecognised keys, or ``None``.

    Kept separate from :func:`describe` so a library call never writes to
    stderr and the same wording serves both ``--show-config`` and the warning
    printed during an ordinary pack.
    """
    if config is None or not config.unknown_keys:
        return None
    return (
        f"{config.source}: ignoring unknown key(s) "
        + ", ".join(f"'{key}'{_suggest(key)}" for key in config.unknown_keys)
    )
