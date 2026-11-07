"""Classify lab responses: blocked, reflected, passed through or broken.

The detector answers one question per payload: *did the filter notice it?*
The answer is a :class:`Verdict`:

* ``BLOCKED``     -- the WAF/filter rejected the request (403/406 or a block
  page signature);
* ``REFLECTED``   -- our payload came back in the body, so it reached the app
  and the output is (likely) injectable;
* ``PASSTHROUGH`` -- clean response without our payload: nothing happened;
* ``ERROR``       -- the request never completed (transport error).

A *baseline* request with an empty payload is used to tell page-level
boilerplate apart from payload-triggered behaviour: signatures that already
appear on the neutral page do not count as blocking.

Verdicts describe the *filter*, not the *backend*. A payload that slipped past
the filter and blew up the database (Juice Shop answers with ``HTTP 500`` and a
``SQLITE_ERROR`` body) is still ``PASSTHROUGH`` as far as the filter is
concerned -- that is exactly why :func:`reached_backend` exists: it reports
whether the payload got all the way down to the DB/backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import unquote

import requests

from .engine import Response, build_request, send

__all__ = [
    "BACKEND_SIGNALS",
    "BLOCK_SIGNATURES",
    "BLOCK_STATUS_CODES",
    "Analysis",
    "Verdict",
    "analyze",
    "backend_signals_in",
    "baseline",
    "classify",
    "reached_backend",
    "reflection_forms",
    "url_decode",
]

#: Substrings (matched case-insensitively) that indicate a block page.
#: Curated to avoid generic words: a bare ``waf`` would also match "wafer".
BLOCK_SIGNATURES: list[str] = [
    "Unexpected path",
    "Request blocked",
    "Request rejected",
    "Access denied",
    "Forbidden",
    "Not acceptable",
    "blocked by",
    "Attack detected",
    "Malicious input",
    "Security policy",
    "Web application firewall",
    "ModSecurity",
]

#: HTTP statuses a WAF/proxy typically returns instead of the real response.
BLOCK_STATUS_CODES: frozenset[int] = frozenset({403, 406})

#: Substrings (matched case-sensitively) proving the payload reached the
#: database/backend layer -- a driver error page is a *successful bypass*, not a
#: tool failure. On the lab stand a working SQLi shows up as ``HTTP 500`` with
#: ``SQLITE_ERROR`` in the body.
BACKEND_SIGNALS: list[str] = [
    "SQLITE_ERROR",
    "SQL syntax",
    "sqlite3.OperationalError",
    "mysql_fetch",
    "ORA-0",
]


class Verdict(StrEnum):
    """Outcome of comparing one payload response with the baseline."""

    BLOCKED = "blocked"
    REFLECTED = "reflected"
    PASSTHROUGH = "passthrough"
    ERROR = "error"


def url_decode(value: str) -> str:
    """Decode percent-escapes once, the way a server-side filter would.

    >>> url_decode("%3Cscript%3E")
    '<script>'
    """
    return unquote(value)


def reflection_forms(
    raw_payload: str, decoded_payload: str | None = None
) -> tuple[str, ...]:
    """Return the payload spellings to look for in a response body.

    Both the raw payload and its URL-decoded form are checked, because a
    server may echo ``%3Cscript%3E`` or ``<script>`` depending on where the
    value ends up (header, attribute, HTML body...). Empty forms are dropped so
    an empty payload can never be "reflected" by every page.

    >>> reflection_forms("%3Cb%3E")
    ('%3Cb%3E', '<b>')
    """
    decoded = url_decode(raw_payload) if decoded_payload is None else decoded_payload
    forms = (raw_payload, decoded)
    return tuple(form for form in dict.fromkeys(forms) if form.strip())


def baseline(
    session: requests.Session,
    url: str,
    method: str = "GET",
    headers: Mapping[str, str] | None = None,
    cookies: Mapping[str, str] | None = None,
    body_template: str | None = None,
    timeout: float = 10,
    retries: int = 1,
) -> Response:
    """Send the neutral reference request: same target, empty payload.

    Everything that shows up here (status, body, block-page wording) belongs
    to the application itself and must not be blamed on a payload.

    Example::

        base = baseline(session, "http://lab.local/search?q={{PAYLOAD}}")
    """
    request = build_request(url, method, "", headers, cookies, body_template)
    return send(session, request, timeout=timeout, retries=retries)


def _block_signature_in(body: str, baseline_body: str) -> bool:
    """True if ``body`` carries a block signature absent from the baseline."""
    haystack = body.lower()
    reference = baseline_body.lower()
    return any(
        signature.lower() in haystack and signature.lower() not in reference
        for signature in BLOCK_SIGNATURES
    )


def _is_blocked(response: Response, baseline_response: Response | None) -> bool:
    """A status code or a block-page signature marks the request as blocked."""
    if response.status_code in BLOCK_STATUS_CODES:
        return True
    baseline_body = baseline_response.body if baseline_response is not None else ""
    return _block_signature_in(response.body or "", baseline_body or "")


def _is_reflected(
    body: str, raw_payload: str, decoded_payload: str | None
) -> bool:
    """True if any spelling of the payload appears verbatim in ``body``."""
    return any(
        form in body for form in reflection_forms(raw_payload, decoded_payload)
    )


def backend_signals_in(body: str) -> tuple[str, ...]:
    """Return the :data:`BACKEND_SIGNALS` found in ``body``, in list order.

    Matching is case-sensitive on purpose: database drivers print these tokens
    verbatim, and a looser match would fire on harmless prose.

    >>> backend_signals_in("SQLITE_ERROR: no such column: 1")
    ('SQLITE_ERROR',)
    >>> backend_signals_in("<html>all good</html>")
    ()
    """
    return tuple(signal for signal in BACKEND_SIGNALS if signal in body)


def reached_backend(response: Response) -> bool:
    """True if the response carries a database/backend error signature.

    This is the "bypass worked" signal, orthogonal to the :class:`Verdict`: a
    payload may pass the filter (``PASSTHROUGH``) and still crash the backend
    with ``HTTP 500`` + ``SQLITE_ERROR``, which on the lab stand means the
    injection got through. Transport errors have no body and never count.

    >>> reached_backend(Response(status_code=500, body="SQLITE_ERROR: near ..."))
    True
    >>> reached_backend(Response(status_code=200, body="<html>ok</html>"))
    False
    """
    if response.error is not None:
        return False
    return bool(backend_signals_in(response.body or ""))


def classify(
    response: Response,
    baseline_response: Response | None,
    raw_payload: str,
    decoded_payload: str | None = None,
) -> Verdict:
    """Turn one :class:`Response` into a :class:`Verdict`.

    The checks run in order of confidence:

    1. ``error`` set -> ``ERROR``;
    2. status 403/406, or a block signature that the baseline page does not
       already contain -> ``BLOCKED`` (status 200 does not clear a block page);
    3. the raw payload or its URL-decoded form present in the body ->
       ``REFLECTED``;
    4. otherwise -> ``PASSTHROUGH``.

    ``decoded_payload`` lets the caller pass a pre-computed
    :func:`url_decode` form; when omitted it is derived from ``raw_payload``.
    Blocking wins over reflection: a 403 page that echoes our payload is still
    a block, not a hit.
    """
    if response.error is not None:
        return Verdict.ERROR
    if _is_blocked(response, baseline_response):
        return Verdict.BLOCKED
    if _is_reflected(response.body or "", raw_payload, decoded_payload):
        return Verdict.REFLECTED
    return Verdict.PASSTHROUGH


@dataclass(frozen=True, slots=True)
class Analysis:
    """Combined view of one response: filter verdict + backend reach flag.

    ``reached_backend`` is the interesting one for SQLi: a ``PASSTHROUGH`` with
    ``reached_backend=True`` means the filter was bypassed and the query hit the
    database, even though the HTTP status is 500.
    """

    verdict: Verdict
    reached_backend: bool
    signals: tuple[str, ...] = ()

    @property
    def is_bypass(self) -> bool:
        """True when the payload survived the filter and reached the backend."""
        return self.reached_backend and self.verdict is not Verdict.BLOCKED


def analyze(
    response: Response,
    baseline_response: Response | None,
    raw_payload: str,
    decoded_payload: str | None = None,
) -> Analysis:
    """Run :func:`classify` and :func:`reached_backend` in one pass.

    Convenience wrapper for the CLI/reporting stage: the verdict keeps its
    documented meaning (it describes the *filter*), while
    :attr:`Analysis.reached_backend` tells whether the payload got down to the
    database. Neither behaviour is altered by this helper.
    """
    return Analysis(
        verdict=classify(response, baseline_response, raw_payload, decoded_payload),
        reached_backend=reached_backend(response),
        signals=backend_signals_in(response.body or ""),
    )
