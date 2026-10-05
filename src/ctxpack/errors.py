"""One exception type, so the CLI can separate user error from a real crash."""

from __future__ import annotations


class CtxpackError(Exception):
    """A problem caused by the user's input rather than a ctxpack bug.

    Anything raised as a ``CtxpackError`` is reported as a one-line message on
    stderr with exit status 2.  Everything else is left to explode loudly.
    """
