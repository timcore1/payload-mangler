"""HTTP execution layer: build, send and batch-run payload requests.

The layer is deliberately thin and side-effect free apart from real network
I/O performed by :func:`send`:

* :func:`build_request` is pure -- it only substitutes :data:`PAYLOAD_MARKER`
  into the URL / body template and returns an immutable :class:`Request`;
* :func:`send` is the only place that talks to the network, and it converts
  every failure into a :class:`Response` with ``error`` set instead of raising;
* :func:`run_batch` fans payloads out over a thread pool and keeps the result
  list aligned with the input order.

The payload is inserted *verbatim* -- it is expected to be already mutated
(URL-encoded, commented, etc.), so no extra quoting happens here.

Lab-only: point it at your own test stand.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import requests

__all__ = [
    "DEFAULT_THREADS",
    "MAX_THREADS",
    "PAYLOAD_MARKER",
    "Request",
    "Response",
    "build_request",
    "run_batch",
    "send",
]

#: Placeholder replaced by the payload in URL and body templates.
PAYLOAD_MARKER = "{{PAYLOAD}}"

DEFAULT_THREADS = 4
MAX_THREADS = 16

#: Errors that are worth retrying: the request never reached the application.
_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


@dataclass(frozen=True, slots=True)
class Request:
    """A single prepared HTTP request (immutable, safe to share)."""

    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    cookies: dict[str, str] = field(default_factory=dict)
    body: str | None = None


@dataclass(frozen=True, slots=True)
class Response:
    """Normalised outcome of one request.

    ``status_code`` is ``None`` and ``error`` is set when the request never
    produced an HTTP response (connection refused, timeout, DNS...).
    """

    status_code: int | None
    body: str = ""
    elapsed_ms: float = 0.0
    size: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the exchange completed without a transport error."""
        return self.error is None


def _fill_marker(template: str, payload: str) -> tuple[str, int]:
    """Return ``template`` with every marker replaced and the hit count."""
    hits = template.count(PAYLOAD_MARKER)
    if hits == 0:
        return template, 0
    return template.replace(PAYLOAD_MARKER, payload), hits


def build_request(
    url_template: str,
    method: str,
    payload: str,
    headers: Mapping[str, str] | None = None,
    cookies: Mapping[str, str] | None = None,
    body_template: str | None = None,
) -> Request:
    """Prepare a request by substituting ``payload`` into the templates.

    The marker may live in the URL, in ``body_template`` or in both; every
    occurrence is replaced. No extra encoding is applied -- the caller decides
    whether the payload should be raw, URL-encoded or otherwise mutated.

    Raises:
        ValueError: if neither the URL nor the body template contains
            :data:`PAYLOAD_MARKER`, i.e. the payload would be silently unused.

    >>> build_request("http://lab.local/s?q={{PAYLOAD}}", "get", "' OR 1=1--").url
    "http://lab.local/s?q=' OR 1=1--"
    """
    url, url_hits = _fill_marker(url_template, payload)
    body: str | None = None
    body_hits = 0
    if body_template is not None:
        body, body_hits = _fill_marker(body_template, payload)
    if url_hits == 0 and body_hits == 0:
        raise ValueError(
            f"no {PAYLOAD_MARKER} marker in url_template or body_template; "
            "the payload would not be sent"
        )
    return Request(
        method=method.strip().upper(),
        url=url,
        headers=dict(headers or {}),
        cookies=dict(cookies or {}),
        body=body,
    )


def _body_of(raw: requests.Response) -> str:
    """Decode the response body, falling back to raw bytes on bad encodings."""
    try:
        return raw.text or ""
    except (UnicodeError, LookupError):  # pragma: no cover - exotic charsets
        return (raw.content or b"").decode("utf-8", errors="replace")


def _error_response(started: float, exc: Exception) -> Response:
    """Wrap a transport failure into a :class:`Response`."""
    return Response(
        status_code=None,
        body="",
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        size=0,
        error=f"{type(exc).__name__}: {exc}",
    )


def _to_response(raw: requests.Response, started: float) -> Response:
    """Convert a successful ``requests`` response into our :class:`Response`."""
    return Response(
        status_code=raw.status_code,
        body=_body_of(raw),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        size=len(raw.content or b""),
        error=None,
    )


def send(
    session: requests.Session,
    request: Request,
    timeout: float = 10,
    retries: int = 1,
) -> Response:
    """Send ``request`` through ``session`` and normalise the outcome.

    Only *network* failures (connection reset, timeout, broken chunk) are
    retried, up to ``retries`` extra attempts. HTTP error statuses are data,
    not exceptions: a 403 comes back as ``Response(status_code=403)``. When
    every attempt fails, the last error lands in ``Response.error`` instead of
    being raised, so one bad payload never kills a batch.
    """
    attempts = max(1, int(retries) + 1)
    for attempt in range(attempts):
        started = time.perf_counter()
        try:
            raw = session.request(
                method=request.method,
                url=request.url,
                headers=request.headers or None,
                cookies=request.cookies or None,
                data=request.body,
                timeout=timeout,
            )
        except _RETRYABLE_ERRORS as exc:
            if attempt + 1 < attempts:
                continue
            return _error_response(started, exc)
        except requests.exceptions.RequestException as exc:
            # Malformed URL / missing schema: retrying cannot help.
            return _error_response(started, exc)
        return _to_response(raw, started)
    raise AssertionError("unreachable")  # pragma: no cover


@dataclass(slots=True)
class _ThreadState:
    """Per-thread session and pacing state (one worker == one session)."""

    session: requests.Session | None = None
    next_at: float | None = None


@dataclass(frozen=True, slots=True)
class _BatchConfig:
    """Everything a worker needs, built once per batch."""

    url: str
    method: str
    headers: Mapping[str, str] | None
    cookies: Mapping[str, str] | None
    body_template: str | None
    timeout: float
    retries: int
    delay: float
    proxy: str | Mapping[str, str] | None
    session_factory: Callable[[], requests.Session]


def _clamp_threads(threads: int) -> int:
    """Keep the worker count inside ``1..MAX_THREADS``."""
    try:
        value = int(threads)
    except (TypeError, ValueError):
        return DEFAULT_THREADS
    return max(1, min(value, MAX_THREADS))


def _proxy_map(proxy: str | Mapping[str, str] | None) -> dict[str, str]:
    """Normalise a proxy setting into the mapping ``requests`` expects."""
    if not proxy:
        return {}
    if isinstance(proxy, str):
        return {"http": proxy, "https": proxy}
    return {str(scheme): str(value) for scheme, value in proxy.items()}


def _state_for(local: threading.local) -> _ThreadState:
    """Fetch (or lazily create) the state of the current worker thread."""
    state: _ThreadState | None = getattr(local, "state", None)
    if state is None:
        state = _ThreadState()
        local.state = state
    return state


def _pace(state: _ThreadState, delay: float) -> None:
    """Sleep so that consecutive requests on this thread are ``delay`` apart."""
    if delay <= 0:
        return
    now = time.monotonic()
    if state.next_at is not None and state.next_at > now:
        time.sleep(state.next_at - now)
    state.next_at = time.monotonic() + delay


def _session_for(config: _BatchConfig, state: _ThreadState) -> requests.Session:
    """Return this thread's session, creating it (with proxies) on first use."""
    if state.session is None:
        session = config.session_factory()
        proxies = _proxy_map(config.proxy)
        if proxies:
            session.proxies.update(proxies)
        state.session = session
    return state.session


def _run_one(config: _BatchConfig, state: _ThreadState, payload: str) -> Response:
    """Build and send a single payload, honouring the per-thread delay."""
    _pace(state, config.delay)
    session = _session_for(config, state)
    request = build_request(
        config.url,
        config.method,
        payload,
        config.headers,
        config.cookies,
        config.body_template,
    )
    return send(session, request, timeout=config.timeout, retries=config.retries)


def _progress_line(done: int, total: int, payload: str, response: Response) -> str:
    """Render one progress line: ``[3/8] 200 142b 12.3ms '<payload>'``."""
    status = str(response.status_code) if response.status_code else "ERR"
    shown = payload if len(payload) <= 40 else f"{payload[:37]}..."
    return (
        f"[{done}/{total}] {status} {response.size}b "
        f"{response.elapsed_ms:.1f}ms {shown!r}"
    )


def _safe_result(future: Future[Response]) -> Response:
    """Turn a worker exception into an ERROR response instead of crashing."""
    try:
        return future.result()
    except Exception as exc:  # noqa: BLE001 - one payload must not kill the batch
        return Response(
            status_code=None,
            body="",
            elapsed_ms=0.0,
            size=0,
            error=f"{type(exc).__name__}: {exc}",
        )


class _Progress:
    """Thread-safe ``[done/total]`` progress printer."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.done = 0
        self._lock = threading.Lock()

    def tick(self, payload: str, response: Response) -> None:
        """Count one finished payload and print its progress line."""
        with self._lock:
            self.done += 1
            print(_progress_line(self.done, self.total, payload, response))


def _execute(
    config: _BatchConfig, items: list[str], threads: int
) -> list[Response]:
    """Run every payload through the pool, keeping the input order."""
    results: list[Response] = [
        Response(status_code=None, error="not executed") for _ in items
    ]
    local = threading.local()
    progress = _Progress(len(items))

    def task(payload: str) -> Response:
        """Send one payload using this worker thread's own session."""
        return _run_one(config, _state_for(local), payload)

    with ThreadPoolExecutor(
        max_workers=_clamp_threads(threads), thread_name_prefix="mangler"
    ) as pool:
        futures = {
            pool.submit(task, payload): index
            for index, payload in enumerate(items)
        }
        for future in as_completed(futures):
            index = futures[future]
            results[index] = _safe_result(future)
            progress.tick(items[index], results[index])
    return results


def run_batch(
    url: str,
    method: str,
    payloads: Iterable[str],
    headers: Mapping[str, str] | None = None,
    cookies: Mapping[str, str] | None = None,
    delay: float = 0.0,
    threads: int = DEFAULT_THREADS,
    proxy: str | Mapping[str, str] | None = None,
    body_template: str | None = None,
    timeout: float = 10,
    retries: int = 1,
    session_factory: Callable[[], requests.Session] = requests.Session,
) -> list[Response]:
    """Send every payload concurrently and print ``[done/total]`` progress.

    Each worker thread owns its own :class:`requests.Session` (sessions are not
    thread-safe) and sleeps ``delay`` seconds between *its own* requests, so
    ``delay=1`` with ``threads=4`` yields at most ~4 req/s. ``threads`` is
    clamped to ``1..MAX_THREADS`` (16). Results come back in input order, one
    entry per payload, so ``responses[i]`` belongs to ``payloads[i]``; failures
    are reported as :class:`Response` objects with ``error`` set.
    """
    items = list(payloads)
    if not items:
        return []
    config = _BatchConfig(
        url=url,
        method=method,
        headers=headers,
        cookies=cookies,
        body_template=body_template,
        timeout=timeout,
        retries=retries,
        delay=max(0.0, float(delay)),
        proxy=proxy,
        session_factory=session_factory,
    )
    return _execute(config, items, threads)
