"""Unit tests for :mod:`payload_mangler.scope`.

The guard is a safety rail, so the tests assert both directions: loopback is
accepted, everything else is rejected, and there is no opt-out switch to sneak
a remote target through. ``socket.gethostbyname`` is patched where DNS would
otherwise be required, so the suite stays offline and deterministic.
"""

from __future__ import annotations

import inspect
import socket

import pytest

from payload_mangler.scope import ALLOWED, ScopeError, check_scope


def patch_resolution(monkeypatch: pytest.MonkeyPatch, mapping: dict[str, str]) -> None:
    """Replace ``socket.gethostbyname`` with a table lookup (offline DNS)."""

    def fake_gethostbyname(host: str) -> str:
        """Return the canned address or fail like a real resolver."""
        try:
            return mapping[host]
        except KeyError:
            raise socket.gaierror(f"no such host: {host}") from None

    monkeypatch.setattr(socket, "gethostbyname", fake_gethostbyname)


class TestAllowedScope:
    def test_allowed_set_is_loopback_only(self) -> None:
        assert ALLOWED == {"127.0.0.1", "::1", "localhost"}

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:3000/rest/user/login",
            "http://127.0.0.1/",
            "https://localhost:3000/api",
            "http://localhost",
            "http://[::1]:3000/rest",
            "http://LOCALHOST:3000/",
        ],
    )
    def test_loopback_targets_pass(self, url: str) -> None:
        assert check_scope(url) is None

    def test_bare_host_with_port_is_accepted(self) -> None:
        assert check_scope("localhost:3000/rest") is None

    def test_ipv6_loopback_does_not_need_dns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(host: str) -> str:
            raise AssertionError(f"resolution attempted for {host}")

        monkeypatch.setattr(socket, "gethostbyname", explode)
        assert check_scope("http://[::1]:8080/") is None

    def test_localhost_resolving_to_loopback_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_resolution(monkeypatch, {"localhost": "127.0.0.1"})
        assert check_scope("http://localhost:3000/") is None

    def test_alias_resolving_to_loopback_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_resolution(monkeypatch, {"lab.internal": "127.0.0.1"})
        assert check_scope("http://lab.internal:3000/") is None


class TestRejectedScope:
    @pytest.mark.parametrize(
        "url",
        [
            "http://8.8.8.8/",
            "http://8.8.8.8:53/dns-query",
            "http://192.168.1.1/",
            "https://192.168.1.1:443/admin",
        ],
    )
    def test_public_and_lan_ips_are_rejected(self, url: str) -> None:
        with pytest.raises(ScopeError):
            check_scope(url)

    def test_remote_domain_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_resolution(monkeypatch, {"example.com": "93.184.216.34"})
        with pytest.raises(ScopeError):
            check_scope("https://example.com/")

    def test_rejected_host_is_named_in_the_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_resolution(monkeypatch, {"example.com": "93.184.216.34"})
        with pytest.raises(ScopeError, match="example.com"):
            check_scope("https://example.com/")

    def test_message_mentions_the_resolved_address(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_resolution(monkeypatch, {"example.com": "93.184.216.34"})
        with pytest.raises(ScopeError, match="93.184.216.34"):
            check_scope("https://example.com/")

    def test_unresolvable_host_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_resolution(monkeypatch, {})
        with pytest.raises(ScopeError, match="did not resolve"):
            check_scope("http://no-such-host.invalid/")

    def test_dns_failure_is_not_treated_as_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_resolution(monkeypatch, {})
        with pytest.raises(ScopeError):
            check_scope("http://127.0.0.1.evil.example/")

    def test_lookalike_hostname_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_resolution(monkeypatch, {"localhost.evil.com": "93.184.216.34"})
        with pytest.raises(ScopeError, match="localhost.evil.com"):
            check_scope("http://localhost.evil.com:3000/")


class TestScopeErrorType:
    def test_scope_error_is_an_exception(self) -> None:
        assert issubclass(ScopeError, Exception)

    def test_scope_error_is_not_a_value_error(self) -> None:
        assert not issubclass(ScopeError, ValueError)

    def test_check_scope_exposes_no_bypass_parameters(self) -> None:
        parameters = inspect.signature(check_scope).parameters
        assert list(parameters) == ["url"]

    def test_check_scope_returns_none(self) -> None:
        assert check_scope("http://127.0.0.1/") is None
