"""Scope guard: refuse to send payloads anywhere but the local lab.

This is a hard safety rail, not a convenience. There is deliberately **no**
``--allow-remote`` / ``--i-know-what-im-doing`` switch: if the target does not
resolve to loopback, the run stops with :class:`ScopeError`.

Both the literal host and its DNS resolution are checked, so ``127.0.0.1``,
``localhost`` and ``::1`` pass, while ``8.8.8.8``, ``example.com`` or anything
on the LAN is rejected.
"""

from __future__ import annotations

import socket
from urllib.parse import urlparse

__all__ = ["ALLOWED", "ScopeError", "check_scope", "resolve_host"]

#: Hosts (literal or resolved) that may be touched.
ALLOWED: set[str] = {"127.0.0.1", "::1", "localhost"}


class ScopeError(Exception):
    """Raised when a target resolves outside the allowed lab scope."""


def resolve_host(host: str) -> str | None:
    """Resolve ``host`` to an IPv4 address, or ``None`` when it does not resolve.

    IPv4-only by design (:func:`socket.gethostbyname`); the IPv6 loopback is
    matched literally in :data:`ALLOWED` before any resolution happens.
    """
    try:
        return socket.gethostbyname(host)
    except (socket.gaierror, UnicodeError, OSError):
        return None


def _extract_host(url: str) -> str:
    """Pull the lower-cased host out of ``url`` (scheme is optional)."""
    candidate = url if "//" in url else f"//{url}"
    host = urlparse(candidate).hostname
    if not host:
        raise ScopeError(f"cannot extract a host from target {url!r}")
    return host.lower()


def check_scope(url: str) -> None:
    """Raise :class:`ScopeError` unless ``url`` points at the local lab.

    The host is accepted when the literal name is in :data:`ALLOWED` or when
    its IPv4 resolution lands on an allowed address (that is how ``localhost``
    is accepted). Everything else -- including unresolvable names, to avoid
    "it failed to resolve, so let it through" loopholes -- is rejected.

    Raises:
        ScopeError: when the target is out of scope, naming the offending host.

    >>> check_scope("http://127.0.0.1:3000/rest/user")  # allowed
    >>> check_scope("http://localhost:3000/")  # allowed
    """
    host = _extract_host(url)
    if host in ALLOWED:
        return
    resolved = resolve_host(host)
    if resolved is not None and resolved in ALLOWED:
        return
    detail = f"resolved to {resolved}" if resolved else "did not resolve"
    raise ScopeError(
        f"out of scope: {host!r} ({detail}); "
        f"allowed targets: {', '.join(sorted(ALLOWED))}"
    )
