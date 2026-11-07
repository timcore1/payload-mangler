"""Success oracles: turn one response into a plain yes/no signal.

A verdict says how the *filter* reacted; an oracle says whether *our* success
marker is present in the body (``SQLITE_ERROR`` for SQLi, ``alert(1)`` for
reflected XSS, a stack trace, a redirect to ``/admin``...). Keeping the two
separate lets the report show "passed the filter **and** the oracle fired".
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .engine import Response

__all__ = ["Oracle", "make_oracle"]

#: A predicate over a response: ``True`` means "the marker was found".
Oracle = Callable[[Response], bool]


def _always_true(response: Response) -> bool:
    """Default oracle: every completed response counts as a hit."""
    return True


def _body_checker(compiled: re.Pattern[str]) -> Oracle:
    """Wrap a compiled pattern into an :data:`Oracle` over ``Response.body``."""

    def check(response: Response) -> bool:
        """True when the pattern matches anywhere in the response body."""
        return compiled.search(response.body or "") is not None

    return check


def make_oracle(pattern: str | None) -> Oracle:
    """Build an :data:`Oracle` from a regex, or ``None`` for "always true".

    ``pattern`` is compiled once (so a batch does not pay for compilation per
    request) and searched against :attr:`Response.body` with
    :func:`re.search`, i.e. it may match anywhere in the body. Matching is
    case-sensitive; add ``(?i)`` to the pattern if you need otherwise.

    With ``pattern=None`` the returned oracle always answers ``True``, which is
    what "I just want to see what the filter does" mode needs.

    Raises:
        re.error: if ``pattern`` is not a valid regular expression.

    >>> make_oracle("SQLITE_ERROR")(Response(status_code=500, body="SQLITE_ERROR"))
    True
    >>> make_oracle(None)(Response(status_code=403, body="blocked"))
    True
    """
    if pattern is None:
        return _always_true
    return _body_checker(re.compile(pattern))
