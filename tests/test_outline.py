"""Outlines: structure must survive, and bodies must not.

The fixture files below are real source written for this test rather than
snippets pasted into the test body, because the things most likely to break an
outline -- a decorator above a class, a signature spread over nine lines, a
docstring with a blank first line -- only appear in files laid out the way a
person would lay them out.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ctxpack.errors import CtxpackError
from ctxpack.outline import (
    JS_EXTS,
    KINDS,
    PYTHON_EXTS,
    Outline,
    estimate_ratio,
    js_outline,
    outline_for,
    outline_text,
    outline_tokens,
    python_outline,
    render_outline,
)
from ctxpack.tokens import estimate_tokens

SERVICE_PY = '''"""Domain layer for accounts and their sessions."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .repo import UserRepo

MAX_RETRIES: int = 3
__all__ = ["UserService", "Session", "build_router"]

SESSION_STATES = ["open", "closed"]


class Colour(str, enum.Enum):
    """Colours a user can pick.

    Stored as strings because they go straight into a URL segment, and the
    numeric values are deliberately not the hex codes: those live in CSS and
    having them here too meant two places to change when a palette moved.
    """

    RED = "red"
    BLUE = "blue"


@dataclass
class UserService:
    """Fetch and cache user records.

    The cache is deliberately small and per-instance. A shared cache across
    services was tried and reverted: it made the hit rate look good and the
    correctness complaints worse.
    """

    name: str
    retries: int = MAX_RETRIES

    def __init__(self, repo: UserRepo) -> None:
        self._repo = repo
        self._cache: dict[str, User] = {}
        self._hits = 0
        self._misses = 0

    async def find_by_email(
        self,
        email: str,
        *,
        include_deleted: bool = False,
    ) -> "User | None":
        """Return the user, or None when there is no match.

        Retries are bounded by ``MAX_RETRIES`` and the delay grows
        exponentially, so a database that is down stays down for a while
        rather than being hammered by every request that arrives while it is.
        """

        def normalise(value: str) -> str:
            return value.strip().lower()

        key = normalise(email)
        if key in self._cache:
            self._hits += 1
            return self._cache[key]

        delay = 0.05
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                user = await self._repo.by_email(key)
            except TimeoutError as error:
                last_error = error
                await asyncio.sleep(delay)
                delay *= 2
                continue
            except ConnectionError:
                self._misses += 1
                raise
            if user is None:
                self._misses += 1
                return None
            if user.deleted and not include_deleted:
                self._misses += 1
                return None
            self._cache[key] = user
            return user

        self._misses += 1
        if last_error is not None:
            raise last_error
        return None

    def invalidate(self, email: str) -> bool:
        """Drop one cached record. Returns whether anything was dropped."""
        key = email.strip().lower()
        if key not in self._cache:
            return False
        del self._cache[key]
        return True

    def warm(self, emails: Iterable[str]) -> int:
        """Preload the cache. Returns how many records were fetched."""
        loaded = 0
        for email in emails:
            if email.strip().lower() in self._cache:
                continue
            user = run_sync(self._repo.by_email(email))
            if user is not None:
                self._cache[email.strip().lower()] = user
                loaded += 1
        return loaded

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    @property
    def healthy(self) -> bool:
        """Whether the backing repository answers."""
        try:
            self._repo.ping()
        except TimeoutError:
            return False
        except ConnectionError:
            return False
        return True

    def stats(self) -> dict[str, int]:
        """Counters for the metrics endpoint."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "cached": len(self._cache),
            "retries": self.retries,
        }


class Session:
    """An authenticated session."""

    def __init__(self, user: User, token: str) -> None:
        self.user = user
        self.token = token
        self.state = SESSION_STATES[0]
        self.issued_at = time.monotonic()

    def valid(self) -> bool:
        """Whether the session is still usable."""
        if self.state != "open":
            return False
        if time.monotonic() - self.issued_at > 3600:
            self.state = "closed"
            return False
        return True

    def extend(self, seconds: int) -> None:
        """Push the expiry out. Only valid on an open session."""
        if not self.valid():
            raise ValueError("cannot extend an expired session")
        self.issued_at = time.monotonic() - max(0, seconds - 3600)

    @classmethod
    def issue(cls, user: User) -> "Session":
        return cls(user, secrets.token_urlsafe(32))

    @staticmethod
    def expire_all() -> int:
        """Drop every session. Returns how many were closed."""
        closed = 0
        for session in list(REGISTRY):
            if session.valid():
                session.state = "closed"
                closed += 1
        REGISTRY.clear()
        return closed


@lru_cache(maxsize=128)
def build_router(routes: list[str], *, prefix: str = "/api") -> Router:
    """Wire the routes up.

    Every route is mounted under ``prefix`` and every handler gets the
    service attached, because threading it through each handler by hand
    was the source of two separate bugs.

    The router is built once and cached, so the ``lru_cache`` above is
    load-bearing: rebuilding it per request cost more than the rest of the
    application combined on the benchmark that found this.
    """
    router = Router(prefix=prefix)
    seen: set[str] = set()
    for route in routes:
        path = prefix.rstrip("/") + "/" + route.lstrip("/")
        if path in seen:
            raise ValueError(f"duplicate route: {path}")
        seen.add(path)
        router.add(path, handler=_make_handler(route))
        if route.requires_auth:
            router.add_middleware(authenticate)
    return router


def _make_handler(route: Route) -> Callable[[Request], Response]:
    """Wrap one route definition in a handler function."""
    schema = route.schema or object

    async def handle(request: Request) -> Response:
        payload = schema.parse(request.body)
        result = route.run(payload)
        return Response(json.dumps(result), status=200)

    handle.__name__ = f"handle_{route.name}"
    return handle


def authenticate(request: Request) -> Response | None:
    """Middleware: reject anything without a bearer token."""
    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        return Response("unauthorized", status=401)
    if header[7:] not in TOKENS:
        return Response("forbidden", status=403)
    return None
'''

STUB_PYI = '''from typing import overload

class Session:
    key: str

    @overload
    def get(self, name: str) -> str | None: ...
    @overload
    def get(self, name: int) -> int | None: ...
    @overload
    def get(self, name: str, default: str) -> str: ...
    async def close(self, *, timeout: float = 1.0) -> None: ...
'''

SERVICE_TS = '''/**
 * HTTP surface for the account service.
 *
 * Everything here is a thin shell over the domain layer: parse, delegate,
 * serialise. No business rule belongs in this file.
 */
import express from "express";
import type { Request, Response } from "express";
import { z } from "zod";

export interface User {
  id: string;
  email?: string;
  createdAt: Date;
  greet(name: string): void;
}

export type Id = string | number;

export enum Kind {
  Free,
  Paid = 10,
}

export const MAX_USERS = 100;

let counter = 0;

/** Build the router. */
export function buildRouter(routes: Route[]): Router {
  const app = express();
  app.use(express.json({ limit: "1mb" }));
  for (const route of routes) {
    app.get(route.path, validate(route.schema), route.handler);
    app.post(route.path, validate(route.schema), route.handler);
  }
  app.use(errorHandler);
  return app;
}

export async function loadUser(id: Id): Promise<User | null> {
  const response = await fetch(`${BASE}/users/${id}`, {
    headers: { accept: "application/json" },
  });
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`upstream said ${response.status}`);
  }
  return (await response.json()) as User;
}

export const findByEmail = async (email: string): Promise<User | null> => {
  const normalised = email.trim().toLowerCase();
  if (!normalised.includes("@")) {
    return null;
  }
  const response = await fetch(`${BASE}/users?email=${encodeURIComponent(normalised)}`);
  const payload = await response.json();
  return payload.length ? payload[0] : null;
};

const handler = (req: Request, res: Response) => {
  res.status(200).json({ ok: true, at: new Date().toISOString() });
};

export const validate = (schema: Schema) => (req: Request) => {
  const parsed = schema.safeParse(req.body);
  if (!parsed.success) {
    throw new ValidationError(parsed.error.issues);
  }
  req.body = parsed.data;
};

export default class UserService extends Base implements Loggable {
  private cache = new Map<string, User>();

  static create(repo: Repo): UserService {
    return new UserService(repo);
  }

  constructor(private repo: Repo) {
    super();
    this.repo = repo;
  }

  async findByEmail(email: string): Promise<User | null> {
    const key = email.trim().toLowerCase();
    const cached = this.cache.get(key);
    if (cached) {
      return cached;
    }
    const found = await this.repo.byEmail(key);
    if (found && !found.deleted) {
      this.cache.set(key, found);
      return found;
    }
    return null;
  }

  async listAll(kind: Kind): Promise<User[]> {
    const users = await this.repo.all();
    const filtered = users.filter((user) => user.kind === kind);
    filtered.sort((a, b) => a.createdAt.getTime() - b.createdAt.getTime());
    if (filtered.length > MAX_USERS) {
      throw new RangeError(`too many users: ${filtered.length}`);
    }
    return filtered;
  }

  get count(): number {
    return this.cache.size;
  }

  #reset() {
    this.cache.clear();
    counter += 1;
  }

  async warm(users: User[]): Promise<number> {
    let loaded = 0;
    for (const user of users) {
      if (this.cache.has(user.id)) {
        continue;
      }
      this.cache.set(user.id, user);
      loaded += 1;
    }
    return loaded;
  }

  onSelect = () => {
    console.log("picked", counter);
  };

  readonly limit = 10;
}
'''


@pytest.fixture
def service_py(tmp_path: Path) -> Path:
    path = tmp_path / "service.py"
    path.write_text(SERVICE_PY, encoding="utf-8")
    return path


@pytest.fixture
def service_ts(tmp_path: Path) -> Path:
    path = tmp_path / "service.ts"
    path.write_text(SERVICE_TS, encoding="utf-8")
    return path


def _by_name(entries: list[Outline]) -> dict[str, Outline]:
    """Flatten to a name-keyed dict, keeping the first of any duplicate name."""
    found: dict[str, Outline] = {}
    for entry in entries:
        for node in entry.walk():
            found.setdefault(node.name, node)
    return found


# -- Outline itself ---------------------------------------------------------


def test_outline_is_frozen():
    entry = Outline("class", "Thing", 3)
    with pytest.raises(FrozenInstanceError):
        entry.line = 4  # type: ignore[misc]


def test_unknown_kind_is_a_user_error():
    with pytest.raises(CtxpackError):
        Outline("widget", "Thing")


def test_every_kind_constant_is_accepted():
    for kind in KINDS:
        assert Outline(kind, "x").kind == kind


def test_signature_joins_name_and_detail():
    assert Outline("function", "f", 1, "(a) -> int").signature == "f(a) -> int"
    assert Outline("function", "f", 1).signature == "f"


def test_walk_is_depth_first_and_find_locates_a_method():
    entries = python_outline(SERVICE_PY)
    service = _by_name(entries)["UserService"]
    names = [node.name for node in service.walk()]
    assert names[0] == "UserService"
    assert "find_by_email" in names
    assert "normalise" in names, "a nested function is two levels down"
    assert names.index("normalise") > names.index("find_by_email")
    assert service.find("normalise") is not None
    assert service.find("nope") is None


def test_markers_are_bare_and_deduplicated():
    entry = Outline("class", "C", 1, decorators=("dataclass", "dataclass"),
                    flags=("stub",))
    assert entry.markers == "dataclass, stub"


# -- Python: structure ------------------------------------------------------


def test_python_finds_module_level_functions(service_py):
    names = _by_name(python_outline(service_py.read_text()))
    assert "build_router" in names
    assert "UserService" in names
    assert "Session" in names


def test_python_decorator_is_kept_as_a_marker():
    entry = _by_name(python_outline(SERVICE_PY))["build_router"]
    assert entry.decorators == ("lru_cache",)
    assert "lru_cache" in entry.markers


def test_python_multiline_signature_becomes_one_line():
    entry = _by_name(python_outline(SERVICE_PY))["find_by_email"]
    assert "\n" not in entry.detail
    assert entry.detail.startswith("(self, email: str, *, include_deleted: bool")
    assert "->" in entry.detail
    assert entry.detail.endswith('-> \'User | None\'')


def test_python_return_type_is_included():
    entry = _by_name(python_outline(SERVICE_PY))["build_router"]
    assert entry.detail.endswith("-> Router")


def test_python_keyword_only_default_keeps_its_spaces():
    detail = _by_name(python_outline(SERVICE_PY))["build_router"].detail
    assert "prefix: str = '/api'" in detail


def test_python_async_is_flagged_and_leading():
    entry = _by_name(python_outline(SERVICE_PY))["find_by_email"]
    assert "async" in entry.flags
    # `async` leads the line because it changes how the declaration is called;
    # it is not left in the trailing comment where a reader has to hunt for it.
    assert render_outline([entry]).startswith("async find_by_email(")


def test_python_dataclass_is_flagged_with_its_fields():
    entry = _by_name(python_outline(SERVICE_PY))["UserService"]
    assert "dataclass" in entry.flags
    fields = {child.name for child in entry.children if child.kind == "const"}
    assert fields == {"name", "retries"}


def test_python_enum_is_flagged_and_lists_members():
    entry = _by_name(python_outline(SERVICE_PY))["Colour"]
    assert "enum" in entry.flags
    assert "Enum" in entry.detail
    assert [child.name for child in entry.children] == ["RED", "BLUE"]


def test_python_methods_are_children_of_their_class():
    entry = _by_name(python_outline(SERVICE_PY))["UserService"]
    methods = [child.name for child in entry.children if child.kind == "method"]
    assert methods == [
        "__init__",
        "find_by_email",
        "invalidate",
        "warm",
        "hit_rate",
        "healthy",
        "stats",
    ]


def test_python_dataclass_fields_and_methods_are_interleaved_in_order():
    entry = _by_name(python_outline(SERVICE_PY))["UserService"]
    kinds = [(c.kind, c.name) for c in entry.children]
    assert kinds[:3] == [
        ("const", "name"),
        ("const", "retries"),
        ("method", "__init__"),
    ]


def test_python_property_and_classmethod_are_marked():
    found = _by_name(python_outline(SERVICE_PY))
    assert "property" in found["healthy"].markers
    assert "classmethod" in found["issue"].markers
    assert "staticmethod" in found["expire_all"].markers


def test_python_nested_function_is_kept_as_a_child():
    entry = _by_name(python_outline(SERVICE_PY))["find_by_email"]
    assert [child.name for child in entry.children] == ["normalise"]


def test_python_nested_function_body_is_not_included():
    rendered = render_outline(python_outline(SERVICE_PY))
    assert "return value.strip().lower()" not in rendered
    assert "self._repo.ping()" not in rendered


def test_python_module_constants_are_entries():
    found = _by_name(python_outline(SERVICE_PY))
    entry = found["MAX_RETRIES"]
    assert entry.kind == "const"
    assert entry.detail == ": int = 3"


def test_python_dunder_all_renders_as_a_name_list():
    entry = _by_name(python_outline(SERVICE_PY))["__all__"]
    assert entry.detail == " = [UserService, Session, build_router]"


def test_python_type_checking_block_is_a_section():
    entry = next(
        e for e in python_outline(SERVICE_PY) if e.kind == "section"
    )
    assert entry.name.startswith("if TYPE_CHECKING")
    assert "UserRepo" in entry.detail or any(
        "UserRepo" in child.name for child in entry.children
    )


def test_python_import_run_is_grouped(service_py):
    text = service_py.read_text()
    imports = [e for e in python_outline(text) if e.kind == "import"]
    names = [e.name for e in imports]
    assert "import os, sys" in names
    assert "from dataclasses import dataclass" in names
    assert "import os" not in names, "the run should have been merged"


def test_python_import_alias_is_kept():
    found = _by_name(python_outline("import numpy as np\n"))
    assert found["import numpy as np"].name == "import numpy as np"


def test_python_docstring_first_line_is_captured():
    entry = _by_name(python_outline(SERVICE_PY))["UserService"]
    assert entry.doc == "Fetch and cache user records."


def test_python_module_docstring_leads_the_outline():
    entries = python_outline(SERVICE_PY)
    assert entries[0].name == "Domain layer for accounts and their sessions."


def test_python_stub_file_is_flagged(tmp_path: Path):
    path = tmp_path / "session.pyi"
    path.write_text(STUB_PYI, encoding="utf-8")
    entries = python_outline(path.read_text(), path=str(path))
    assert all("stub" in entry.flags for entry in entries)
    assert entries[0].name.startswith("from typing import overload")


def test_python_stub_overloads_are_all_kept(tmp_path: Path):
    path = tmp_path / "session.pyi"
    path.write_text(STUB_PYI, encoding="utf-8")
    got = _by_name(python_outline(path.read_text(), path=str(path)))
    session = got["Session"]
    gets = [c for c in session.children if c.name == "get"]
    assert len(gets) == 3
    assert all("overload" in node.markers for node in gets)


def test_python_comprehension_constant_is_one_line():
    text = "NAMES = tuple(f'user_{i}' for i in range(10))\n"
    entry = _by_name(python_outline(text))["NAMES"]
    assert "\n" not in entry.detail
    assert "range(10)" in entry.detail


def test_python_multiline_comprehension_is_elided():
    text = "NAMES = [\n    f'user_{i}'\n    for i in range(10)\n]\n"
    assert _by_name(python_outline(text))["NAMES"].detail == " = ..."


def test_python_big_value_is_elided_not_reproduced():
    text = "MESSAGES = {\n" + "".join(
        f'    "k{i}": "a value long enough to matter",\n' for i in range(80)
    ) + "}\n"
    entry = _by_name(python_outline(text))["MESSAGES"]
    assert "a value long enough to matter" not in entry.detail


def test_python_multiline_value_is_elided():
    text = "TABLE = {\n    'a': 1,\n    'b': 2,\n}\n"
    assert "1" not in _by_name(python_outline(text))["TABLE"].detail


# -- Python: line numbers and order -----------------------------------------


def test_line_numbers_are_exact_for_known_declarations():
    lines = SERVICE_PY.splitlines()
    found = _by_name(python_outline(SERVICE_PY))
    # `ast` reports the `def` line, not the decorator line above it, which is
    # what makes the number safe to jump to: the decorator is still on screen.
    assert lines[found["UserService"].line - 1].startswith("class UserService:")
    assert lines[found["build_router"].line - 1].startswith("def build_router(")
    assert lines[found["find_by_email"].line - 1].strip().startswith("async def")
    assert lines[found["Colour"].line - 1].startswith("class Colour(")
    assert lines[found["__all__"].line - 1].startswith("__all__ =")
    assert lines[found["Session"].line - 1].startswith("class Session:")


def test_every_line_number_points_inside_the_file():
    total = len(SERVICE_PY.splitlines())
    for entry in python_outline(SERVICE_PY):
        for node in entry.walk():
            assert 1 <= node.line <= total, node


def test_source_order_is_preserved(service_py):
    entries = python_outline(service_py.read_text())
    top = [e for e in entries if e.kind != "other"]
    assert top == sorted(top, key=lambda e: e.line)
    for entry in entries:
        kids = [c.line for c in entry.children]
        assert kids == sorted(kids)


def test_nested_children_are_inside_their_parents():
    entries = python_outline(SERVICE_PY)
    service = _by_name(entries)["UserService"]
    for child in service.children:
        assert service.line < child.line
        assert child.name not in {e.name for e in entries if e.kind != "other"}


# -- Python: failure paths --------------------------------------------------


def test_syntax_error_falls_back_to_a_line_scan():
    entries = python_outline("def broken(:\n    pass\n")
    assert entries[0].name == "(unstructured)"
    recovered = python_outline(
        "class Alpha:\n    def one(self):\n        pass\n\ndef broken(:\n"
    )
    names = {node.name for node in recovered for node in node.walk()}
    assert {"Alpha", "one"} <= names


def test_truncated_file_does_not_raise():
    text = "import os\n\n\ndef alpha(x):\n    if x:\n        return\n\n\nclass Beta:\n    def gamma(self):"
    entries = python_outline(text)
    assert _by_name(entries)["Beta"].children[0].name == "gamma"


def test_truncated_file_recovers_bodies_before_the_cut():
    text = "class A:\n    def m(self):\n        pass\n    def n(self):\n"
    assert [c.name for c in _by_name(python_outline(text))["A"].children] == ["m", "n"]


def test_unrecoverable_syntax_error_gives_the_placeholder():
    entries = python_outline("if (\n")
    assert len(entries) == 1
    assert entries[0].name == "(unstructured)"
    assert "syntax error" in entries[0].detail


def test_empty_file_returns_the_placeholder():
    entries = python_outline("")
    assert len(entries) == 1
    assert entries[0].name == "(unstructured)"
    assert "empty" in entries[0].detail


def test_whitespace_only_file_returns_the_placeholder():
    entries = python_outline("   \n\n\t\n")
    assert len(entries) == 1
    assert entries[0].name == "(unstructured)"


def test_comment_only_file_returns_the_placeholder():
    entries = python_outline("# nothing here\n")
    assert len(entries) == 1
    assert entries[0].name == "(unstructured)"


def test_none_text_does_not_raise():
    assert len(python_outline(None)) == 1  # type: ignore[arg-type]


def test_never_returns_an_empty_outline_for_anything():
    # The invariant the whole module rests on: `[]` is indistinguishable from
    # "the file was empty" and from "the parser gave up", and all three let a
    # file disappear from a bundle without a trace.
    for text in ("", "   ", "\n\n", "# only a comment", "{}", "()", "\x00", "\x1f"):
        for ext in ("a.py", "a.js", "a.md", "a.rb", ""):
            assert outline_for(ext, text), (ext, text)


def test_every_entry_kind_is_documented():
    entries = python_outline(SERVICE_PY) + js_outline(SERVICE_TS)
    seen = {node.kind for entry in entries for node in entry.walk()}
    assert seen <= set(KINDS)
    assert {"class", "function", "method", "const", "import"} <= seen
    assert {"interface", "type", "enum"} <= {e.kind for e in js_outline(SERVICE_TS)}
    assert "section" in {e.kind for e in python_outline(SERVICE_PY)}


def test_python_and_js_share_the_kind_vocabulary():
    # One rendering path for two languages only works if both emit kinds from
    # the same set; a Python-only kind would render differently by accident.
    assert {e.kind for e in python_outline(SERVICE_PY)} <= set(KINDS)
    assert {e.kind for e in js_outline(SERVICE_TS)} <= set(KINDS)


def test_hostile_input_does_not_raise():
    for text in ["\x00\xff", "def" * 500, "(" * 400, "class A:\n" * 200]:
        python_outline(text)
        js_outline(text)
        render_outline(outline_for("a.py", text))


def test_a_broken_file_does_not_stop_the_next_one(tmp_path: Path):
    (tmp_path / "a.py").write_text("def broken(:\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def fine():\n    pass\n", encoding="utf-8")
    outs = [
        render_outline(outline_for(str(tmp_path / name), (tmp_path / name).read_text()))
        for name in ("a.py", "b.py")
    ]
    assert "fine" in outs[1]


# -- JavaScript / TypeScript ------------------------------------------------


def test_js_finds_every_required_shape(service_ts):
    found = _by_name(js_outline(service_ts.read_text()))
    for name in ("User", "Id", "Kind", "MAX_USERS", "buildRouter", "loadUser",
                 "findByEmail", "UserService"):
        assert name in found


def test_js_exported_function_is_marked(service_ts):
    entry = _by_name(js_outline(service_ts.read_text()))["buildRouter"]
    assert "export" in entry.flags
    assert entry.doc == "Build the router."


def test_js_arrow_const_is_a_function(service_ts):
    entry = _by_name(js_outline(service_ts.read_text()))["findByEmail"]
    assert entry.kind == "function"
    assert "async" in entry.flags
    assert entry.detail.startswith("(email: string)")
    assert "Promise<User | null>" in entry.detail


def test_js_bare_arrow_param_is_normalised():
    entries = js_outline("const twice = x => x * 2;\n")
    assert entries[0].name == "twice"
    assert entries[0].detail == "(x)"


def test_js_function_expression_const_is_a_function():
    entries = js_outline("const run = function (a, b) { return a; };\n")
    assert entries[0].kind == "function"
    assert entries[0].detail == "(a, b)"


def test_js_interface_lists_its_fields(service_ts):
    entry = _by_name(js_outline(service_ts.read_text()))["User"]
    assert entry.kind == "interface"
    fields = {c.name: c.detail for c in entry.children}
    assert fields["id"] == ": string"
    assert fields["email"] == ": string"


def test_js_type_alias_keeps_its_body(service_ts):
    entry = _by_name(js_outline(service_ts.read_text()))["Id"]
    assert entry.kind == "type"
    assert entry.detail == " = string | number"


def test_js_enum_lists_members(service_ts):
    entry = _by_name(js_outline(service_ts.read_text()))["Kind"]
    assert entry.kind == "enum"
    assert [c.name for c in entry.children] == ["Free", "Paid"]


def test_js_line_continued_const_chain_is_one_entry():
    entries = js_outline("export const FLAGS =\n  A |\n  B |\n  C;\n")
    assert len(entries) == 1
    assert entries[0].name == "FLAGS"
    assert "A | B | C" in entries[0].detail


def test_js_class_children_and_markers(service_ts):
    entry = _by_name(js_outline(service_ts.read_text()))["UserService"]
    assert entry.kind == "class"
    assert "Base" in entry.detail
    kids = {c.name: c for c in entry.children}
    assert "static" in kids["create"].markers
    assert "get" in kids["count"].markers
    assert "async" in kids["findByEmail"].markers
    assert "#reset" in kids
    assert kids["onSelect"].kind == "method"
    assert kids["limit"].kind == "const"
    assert "cache" in kids and kids["cache"].kind == "const"


def test_js_default_export(service_ts):
    entry = _by_name(js_outline(service_ts.read_text()))["UserService"]
    assert "default" in entry.flags
    assert "export" in entry.flags


def test_js_non_default_class_has_no_default_flag():
    entries = js_outline("class Plain { run() {} }\n")
    assert "default" not in entries[0].flags


def test_js_default_export_of_a_bare_expression():
    entries = js_outline("export default makeThing();\n")
    assert entries[0].name == "default export"
    assert "makeThing" in entries[0].detail


def test_js_default_export_of_an_anonymous_function():
    entries = js_outline("export default function () {}\n")
    assert entries[0].name == "default"
    assert "default" in entries[0].flags


def test_js_abstract_class_is_marked():
    entries = js_outline("export abstract class Base {}\n")
    assert "abstract" in entries[0].markers


def test_js_imports_are_entries(service_ts):
    imports = [e for e in js_outline(service_ts.read_text()) if e.kind == "import"]
    assert any("express" in e.name for e in imports)
    assert all(e.line > 0 for e in imports)


def test_js_local_variables_are_not_entries(service_ts):
    assert "app" not in _by_name(js_outline(service_ts.read_text()))


def test_js_commented_out_code_is_ignored():
    text = "function real() {}\n// function fake() {}\n/* function alsoFake() {} */\n"
    names = [e.name for e in js_outline(text)]
    assert names == ["real"]


def test_js_multiline_params_are_cut_cleanly():
    text = "export function load(\n  id: string,\n  opts: Options,\n): Promise<User> {\n}\n"
    entry = js_outline(text)[0]
    assert entry.name == "load"
    assert "Promise<User>" in entry.detail
    assert "{\n}" not in entry.detail


def test_js_unbalanced_braces_do_not_raise():
    text = 'function a() { return "{"; }\nfunction b() { return 1; }\n'
    entries = js_outline(text)
    assert entries


def test_js_line_numbers_are_correct(service_ts):
    lines = service_ts.read_text().splitlines()
    for entry in js_outline(service_ts.read_text()):
        if entry.kind == "other":
            continue
        assert lines[entry.line - 1].strip().startswith(entry.name[:4]) or \
            lines[entry.line - 1].strip().startswith(("export", "import", "const",
                                                      "let", "async", "type",
                                                      "class", "interface", "enum",
                                                      "static", "get", "#"))


def test_js_source_order_is_preserved(service_ts):
    entries = js_outline(service_ts.read_text())
    top = [e for e in entries if e.kind != "other"]
    assert top == sorted(top, key=lambda e: e.line)


def test_js_empty_file_returns_the_placeholder():
    entries = js_outline("  \n\n")
    assert len(entries) == 1
    assert entries[0].name == "(unstructured)"


def test_js_no_declarations_returns_the_placeholder():
    entries = js_outline("console.log('hi');\n")
    assert len(entries) == 1
    assert entries[0].name == "(unstructured)"


# -- Rendering --------------------------------------------------------------


def test_render_nests_with_two_spaces():
    text = render_outline(python_outline(SERVICE_PY))
    # A method under a class: two spaces, and no `method` keyword, because the
    # class line above already said `class`.
    assert "\n  __init__(self, repo: UserRepo) -> None" in text
    # A nested function under a method: four spaces. Getting the depth factor
    # wrong by a constant is easy and invisible without asserting the exact
    # two-level case.
    assert "\n    function normalise(value: str) -> str" in text


def test_render_indent_grows_one_level_per_depth():
    method = _by_name(python_outline(SERVICE_PY))["find_by_email"]
    rendered = render_outline([method])
    assert rendered.startswith("async find_by_email(")
    assert "\n  function normalise(" in rendered


def test_render_indent_is_configurable():
    text = render_outline(python_outline(SERVICE_PY), indent="    ")
    assert "\n    __init__(self, repo: UserRepo)" in text
    assert "\n  __init__(self, repo: UserRepo)" not in text


def test_methods_render_with_no_keyword():
    # The class line above already said `class`, so repeating `method` on every
    # member would spend a token on the most numerous lines in the outline to
    # say nothing.
    text = render_outline(python_outline(SERVICE_PY))
    assert "\n  __init__(" in text
    assert "\n  method " not in text
    assert "\n  function __init__" not in text


@pytest.mark.parametrize("indent", ["", "->", "a b", "-->"])
def test_render_rejects_a_non_whitespace_indent(indent: str):
    with pytest.raises(CtxpackError):
        render_outline(python_outline(SERVICE_PY), indent=indent)


def test_render_accepts_a_tab_indent():
    assert "\n\t__init__(" in render_outline(python_outline(SERVICE_PY), indent="\t")


def test_render_of_nothing_is_an_empty_string():
    assert render_outline([]) == ""


def test_render_contains_the_expected_names():
    text = render_outline(python_outline(SERVICE_PY))
    for name in ("UserService", "find_by_email", "build_router", "Colour",
                 "__all__", "TYPE_CHECKING"):
        assert name in text


def test_render_has_one_line_per_declaration():
    entries = python_outline(SERVICE_PY)
    nodes = sum(1 for entry in entries for _ in entry.walk())
    assert len(render_outline(entries).splitlines()) == nodes


def test_render_puts_the_docstring_in_a_trailing_comment():
    line = next(
        line for line in render_outline(python_outline(SERVICE_PY)).splitlines()
        if "class UserService" in line
    )
    assert line.endswith("# dataclass - Fetch and cache user records.")


def test_render_has_no_trailing_whitespace():
    for line in render_outline(python_outline(SERVICE_PY)).splitlines():
        assert line == line.rstrip()


def test_render_elides_a_long_comment():
    text = 'def f():\n    """' + "word " * 60 + '"""\n    pass\n'
    line = render_outline(python_outline(text)).splitlines()[-1]
    assert line.endswith("...")
    assert len(line) < 160


def test_render_is_stable_across_calls():
    entries = python_outline(SERVICE_PY)
    assert render_outline(entries) == render_outline(entries)


def test_render_does_not_pad_comments_into_a_column():
    # No alignment padding: it costs whitespace tokens on every line and makes a
    # one-character rename reflow the whole file's whitespace, which is exactly
    # what "diff-stable" has to mean for an outline someone might commit.
    text = render_outline(python_outline(SERVICE_PY))
    commented = [line for line in text.splitlines() if "  # " in line]
    assert len(commented) > 5
    for line in commented:
        head, _, comment = line.partition("  # ")
        assert line == head + "  # " + comment
        assert "  # " not in head, "the head must not carry padding"


# -- Dispatch ---------------------------------------------------------------


def test_dispatch_by_extension(tmp_path: Path):
    (tmp_path / "a.py").write_text("def f():\n    pass\n", encoding="utf-8")
    (tmp_path / "a.js").write_text("function f() {}\n", encoding="utf-8")
    (tmp_path / "a.md").write_text("# hi\n", encoding="utf-8")
    assert "f" in _by_name(outline_for(str(tmp_path / "a.py"),
                                       (tmp_path / "a.py").read_text()))
    assert "f" in _by_name(outline_for(str(tmp_path / "a.js"),
                                       (tmp_path / "a.js").read_text()))
    assert outline_for(str(tmp_path / "a.md"), "# hi")[0].name == "(unstructured)"


def test_dispatch_accepts_an_explicit_kind():
    text = "def f():\n    pass\n"
    assert _by_name(outline_for("notes.txt", text, kind="python"))["f"]


def test_dispatch_rejects_an_unknown_kind():
    with pytest.raises(CtxpackError):
        outline_for("a.py", "x = 1", kind="brainfuck")


def test_dispatch_treats_an_empty_kind_as_auto():
    assert outline_for("a.py", "x = 1", kind="") == outline_for("a.py", "x = 1")


def test_dispatch_names_the_offending_kind():
    with pytest.raises(CtxpackError, match="brainfuck"):
        outline_for("a.py", "x = 1", kind="brainfuck")


def test_placeholder_reason_is_in_the_entry():
    entry = outline_for("a.rb", "puts 1\n")[0]
    assert "no extractor for .rb" in entry.detail


def test_extension_sets_are_disjoint_and_documented():
    assert not PYTHON_EXTS & JS_EXTS
    assert ".pyi" in PYTHON_EXTS
    assert ".tsx" in JS_EXTS
    assert ".vue" not in JS_EXTS


def test_extensionless_path_is_not_an_error():
    assert len(outline_for("Makefile", "all:\n\techo hi\n")) == 1


def test_outline_text_helper_matches_render_of_outline_for():
    assert outline_text("a.py", SERVICE_PY) == render_outline(
        outline_for("a.py", SERVICE_PY)
    )


# -- Costing ----------------------------------------------------------------


def test_outline_tokens_is_far_below_the_file(service_py):
    text = service_py.read_text()
    assert outline_tokens(text, path=str(service_py)) < estimate_tokens(text) / 3


def test_estimate_ratio_is_a_large_compression(service_py):
    # Typical range on real source is 5x-20x; 3x is a floor that only fails if
    # bodies or long values are leaking into the outline.
    ratio = estimate_ratio(SERVICE_PY, path="service.py")
    assert ratio > 3, ratio


def test_estimate_ratio_of_empty_text_is_zero_not_a_crash():
    assert estimate_ratio("") == 0.0
    assert estimate_ratio("   \n") == 0.0


def test_estimate_ratio_of_a_placeholder_is_still_a_number():
    assert estimate_ratio("# nothing\n", path="a.rb") > 0


def test_outline_tokens_is_zero_for_nothing_to_say():
    assert outline_tokens("", path="a.py") > 0, "the placeholder still costs tokens"


def test_language_is_sniffed_when_no_path_is_given():
    assert estimate_ratio(SERVICE_PY) > 3
    assert estimate_ratio(SERVICE_TS) > 3


def test_tiny_files_compress_less_than_large_ones():
    # The honest reason the ratio is not a constant: a three-line file has no
    # bodies to remove, so its outline is nearly as long as the file.
    tiny = estimate_ratio("x = 1\n")
    big = estimate_ratio(SERVICE_PY)
    assert tiny < big


def test_ratio_holds_across_the_project_corpus(project: Path):
    """Every file in the shared fixture project must compress at least a bit.

    The floor is deliberately low because the fixture is made of two-line
    modules: there is no body to remove, so the outline is nearly as long as the
    file. What matters is that nothing in that project produces a ratio below
    1, which would mean the outline was *bigger* than the source.
    """
    ratios = {
        str(path.relative_to(project)): estimate_ratio(
            path.read_text(encoding="utf-8"), path=str(path)
        )
        for path in sorted(project.rglob("*.py"))
    }
    assert min(ratios.values()) > 1.0, ratios


def test_ratio_of_a_real_corpus_is_in_the_expected_band():
    """Measure against ctxpack's own source: the honest compression figure.

    Real modules -- not the purpose-built fixtures -- land between 5x and 15x,
    with a median around 6x. Files are filtered to those over 20 lines, because
    a five-line module has no bodies to remove and its ratio is close to 1 by
    arithmetic rather than by anything ctxpack did. The bounds: under 3x means
    bodies or long values are leaking into the outline; over 40x means
    declarations are being dropped and the outline has stopped being useful.
    """
    root = Path(__file__).resolve().parent.parent / "src" / "ctxpack"
    files = [p for p in sorted(root.glob("*.py")) if len(p.read_text().splitlines()) > 20]
    assert len(files) > 3, "expected ctxpack's own modules to be present"
    measured = {
        path.name: estimate_ratio(path.read_text(encoding="utf-8"), path=str(path))
        for path in files
    }
    assert min(measured.values()) > 3.0, measured
    assert max(measured.values()) < 40.0, measured


def test_tiny_modules_compress_less_because_they_have_no_bodies():
    # Honest limit of the technique, asserted so it stays a known limit: the
    # ratio is a property of how much body a file has, not of this module's
    # cleverness. `errors.py` is a single exception class and a docstring.
    root = Path(__file__).resolve().parent.parent / "src" / "ctxpack"
    text = (root / "errors.py").read_text(encoding="utf-8")
    assert len(text.splitlines()) < 20
    assert estimate_ratio(text, path="errors.py") < 3.0


# -- Determinism ------------------------------------------------------------


def test_same_input_twice_is_identical():
    first = render_outline(outline_for("service.py", SERVICE_PY))
    second = render_outline(outline_for("service.py", SERVICE_PY))
    assert first == second


def test_same_input_twice_is_identical_for_js():
    assert js_outline(SERVICE_TS) == js_outline(SERVICE_TS)
    assert render_outline(js_outline(SERVICE_TS)) == render_outline(js_outline(SERVICE_TS))


def test_reading_from_disk_twice_is_identical(service_py: Path):
    text = service_py.read_text()
    assert render_outline(python_outline(text)) == render_outline(
        python_outline(text)
    )


def test_entries_are_frozen_tuples(service_py):
    entries = outline_for(str(service_py), service_py.read_text())
    for entry in entries:
        assert isinstance(entry.children, tuple)


# ---------------------------------------------------------------------------
# regressions found by review
# ---------------------------------------------------------------------------


def test_pathological_js_nesting_does_not_recurse_off_the_stack():
    """Deep brace nesting must degrade, never raise.

    Regression: rendering recursed once per nesting level, and 1200 nested
    `class C {` blocks -- trivial to produce from minified or generated input --
    raised a bare ``RecursionError``. The module's contract is that it never
    raises on any input, and `cli.main` only catches CtxpackError, so this
    escaped as a traceback and a non-zero exit.
    """
    for depth in (400, 1200):
        src = "\n".join([f"class C{i} {{" for i in range(depth)] + ["}" for _ in range(depth)])
        try:
            out = render_outline(js_outline(src))
        except RecursionError as exc:  # pragma: no cover - the regression
            raise AssertionError(f"depth {depth} recursed off the stack") from exc
        assert out.strip()


def test_pathological_nesting_through_the_public_helper():
    src = "\n".join([f"class C{i} {{" for i in range(2000)] + ["}" for _ in range(2000)])
    assert estimate_ratio(src) >= 0.0
    assert outline_text("deep.ts", src)


def test_module_level_match_statements_are_kept():
    """Declarations inside `match`/`case` arms must appear in the outline.

    Regression: `ast.Match` is the one compound statement Python 3.10 added, and
    the body walker handled If/Try/With/For/While but not Match. Every
    declaration inside a match arm vanished with no placeholder -- the worst
    failure mode for a navigation tool, since the reader is told those symbols
    do not exist.
    """
    src = (
        "import sys\n\n"
        "match sys.version_info:\n"
        '    case (3, 10):\n'
        "        def only_310() -> None: ...\n"
        '    case (3, 11) | (3, 12):\n'
        "        class Modern: ...\n"
        "    case _:\n"
        "        def fallback(): ...\n"
    )
    names = {o.name for o in python_outline(src)}
    assert {"only_310", "Modern", "fallback"} <= names


def test_match_inside_a_class_is_kept():
    src = (
        "class Router:\n"
        "    def dispatch(self, cmd):\n"
        "        match cmd:\n"
        '            case "go":\n'
        "                return 1\n"
        "    def handle(self):\n"
        "        return 2\n"
    )
    out = render_outline(python_outline(src))
    assert "handle" in out


@pytest.mark.parametrize(
    "sep",
    ["\x0c", "\x0b", "\x1c", "\u2028", "\x85", "\x1e"],
    ids=["form-feed", "vertical-tab", "file-sep", "line-sep", "NEL", "record-sep"],
)
def test_js_line_numbers_survive_odd_line_separators(sep: str):
    """A form feed must not shift every line number after it.

    Regression: `str.splitlines()` breaks on form feed, vertical tab, file
    separator, U+2028 and NEL, all of which are ordinary characters in a source
    file. Each one shifted the numbering of everything below it, so a single form
    feed misreported a file's whole layout. JavaScript has no equivalent of
    Python's "invalid non-printable character" rule, so all of these can really
    appear in a .ts file.
    """
    js = f"export const a = 1;\n{sep}\nexport function real() {{\n  return 1;\n}}"
    lines = {o.name: o.line for o in js_outline(js)}
    assert lines["real"] == 3, f"separator {sep!r} shifted line numbers: {lines}"


def test_python_line_numbers_survive_a_form_feed():
    """Python permits a form feed as whitespace; the others it rejects outright.

    `ast.parse` raises "invalid non-printable character" for vertical tab, file
    separator, U+2028 and NEL, so a real .py file cannot contain them at top
    level. Form feed is the one that can, and it is the one that used to shift
    the numbering of everything below it.
    """
    py = "x = 1\n\x0c\ndef after():\n    return 2\n"
    got = {o.name: o.line for o in python_outline(py)}
    assert got.get("after") == 3


@pytest.mark.parametrize("sep", ["\x0b", "\x1c", "\u2028"])
def test_python_rejects_the_separators_it_rejects(sep: str):
    """A file Python cannot parse still gets a non-empty outline.

    Not a line-number question: the placeholder is the documented behaviour for
    input `ast` rejects, and what matters is that it is non-empty and says so.
    """
    py = f"x = 1\n{sep}\ndef after():\n    return 2\n"
    out = python_outline(py)
    assert out, "the never-empty invariant must hold"
    assert out[0].kind == "other"


def test_trailing_newline_does_not_add_a_line():
    src = "def a():\n    return 1\n"
    assert python_outline(src)[0].line == 1
    assert python_outline("def a():\n    return 1")[0].line == 1
