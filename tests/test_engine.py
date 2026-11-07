"""Unit tests for :mod:`payload_mangler.engine`.

Everything runs offline: a :class:`FakeSession` returns real
``requests.Response`` objects (configured via ``_content``), so the engine is
exercised without touching the network.
"""

from __future__ import annotations

import pytest
import requests

from payload_mangler.engine import (
    MAX_THREADS,
    Request,
    Response,
    build_request,
    run_batch,
    send,
)


def http_response(
    status_code: int = 200, body: str = "", encoding: str = "utf-8"
) -> requests.Response:
    """Build a real ``requests.Response`` as if it came off the wire."""
    response = requests.Response()
    response.status_code = status_code
    response._content = body.encode(encoding)
    response.encoding = encoding
    response.url = "http://lab.local/"
    return response


class FakeSession:
    """Minimal ``requests.Session`` stand-in that records calls."""

    def __init__(
        self,
        responses: list[requests.Response | Exception] | None = None,
    ) -> None:
        self.responses: list[requests.Response | Exception] = list(responses or [])
        self.proxies: dict[str, str] = {}
        self.calls: list[dict[str, object]] = []

    def request(self, **kwargs: object) -> requests.Response:
        self.calls.append(kwargs)
        item = self.responses.pop(0) if self.responses else http_response()
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture()
def request_stub() -> Request:
    return build_request("http://lab.local/item?id={{PAYLOAD}}", "GET", "' OR 1=1--")


class TestBuildRequest:
    def test_replaces_marker_in_url(self) -> None:
        request = build_request("http://lab.local/?q={{PAYLOAD}}", "GET", "1=1")
        assert request.url == "http://lab.local/?q=1=1"

    def test_does_not_extra_encode_the_payload(self) -> None:
        payload = "%27 OR 1=1-- <script>"
        request = build_request("http://lab.local/?q={{PAYLOAD}}", "GET", payload)
        assert request.url.endswith(payload)
        assert "%2527" not in request.url

    def test_replaces_marker_in_body_template(self) -> None:
        request = build_request(
            "http://lab.local/login",
            "post",
            "' OR 1=1--",
            body_template="username={{PAYLOAD}}&password=x",
        )
        assert request.body == "username=' OR 1=1--&password=x"

    def test_replaces_marker_in_url_and_body(self) -> None:
        request = build_request(
            "http://lab.local/s?q={{PAYLOAD}}",
            "POST",
            "x",
            body_template="q={{PAYLOAD}}",
        )
        assert request.url == "http://lab.local/s?q=x"
        assert request.body == "q=x"

    def test_uppercases_method_and_keeps_body_empty_for_get(self) -> None:
        request = build_request("http://lab.local/{{PAYLOAD}}", " get ", "1")
        assert request.method == "GET"
        assert request.body is None

    def test_copies_headers_and_cookies(self) -> None:
        headers = {"User-Agent": "lab"}
        cookies = {"session": "abc"}
        request = build_request(
            "http://lab.local/{{PAYLOAD}}", "GET", "1", headers, cookies
        )
        assert request.headers == headers
        assert request.cookies == cookies
        assert request.headers is not headers

    def test_empty_payload_still_matches_marker(self) -> None:
        request = build_request("http://lab.local/?q={{PAYLOAD}}", "GET", "")
        assert request.url == "http://lab.local/?q="

    def test_rejects_template_without_marker(self) -> None:
        with pytest.raises(ValueError, match="no \\{\\{PAYLOAD\\}\\} marker"):
            build_request("http://lab.local/static", "GET", "1=1")

    def test_request_is_immutable(self, request_stub: Request) -> None:
        with pytest.raises(AttributeError):
            request_stub.url = "http://evil.local/"  # type: ignore[misc]


class TestSend:
    def test_returns_normalised_response(self, request_stub: Request) -> None:
        session = FakeSession([http_response(200, "hello")])
        response = send(session, request_stub)
        assert response.status_code == 200
        assert response.body == "hello"
        assert response.size == len(b"hello")
        assert response.error is None
        assert response.elapsed_ms >= 0

    def test_passes_method_url_and_timeout(self, request_stub: Request) -> None:
        session = FakeSession()
        send(session, request_stub, timeout=3)
        call = session.calls[0]
        assert call["method"] == "GET"
        assert call["url"] == request_stub.url
        assert call["timeout"] == 3
        assert call["data"] is None

    def test_http_error_status_is_data_not_exception(self) -> None:
        request = build_request("http://lab.local/{{PAYLOAD}}", "GET", "1")
        response = send(FakeSession([http_response(403, "no")]), request)
        assert response.status_code == 403
        assert response.error is None

    def test_network_error_sets_error_and_no_status(self) -> None:
        request = build_request("http://lab.local/{{PAYLOAD}}", "GET", "1")
        session = FakeSession([requests.exceptions.ConnectionError("boom")])
        response = send(session, request, retries=0)
        assert response.status_code is None
        assert response.error is not None
        assert "ConnectionError" in response.error

    def test_retries_network_error_once_by_default(self) -> None:
        request = build_request("http://lab.local/{{PAYLOAD}}", "GET", "1")
        session = FakeSession(
            [requests.exceptions.ConnectionError("boom"), http_response(200, "ok")]
        )
        response = send(session, request)
        assert response.status_code == 200
        assert len(session.calls) == 2

    def test_retries_zero_means_single_attempt(self) -> None:
        request = build_request("http://lab.local/{{PAYLOAD}}", "GET", "1")
        session = FakeSession([requests.exceptions.Timeout("slow")])
        send(session, request, retries=0)
        assert len(session.calls) == 1

    def test_gives_up_after_all_retries(self) -> None:
        request = build_request("http://lab.local/{{PAYLOAD}}", "GET", "1")
        session = FakeSession([requests.exceptions.Timeout("slow")] * 3)
        response = send(session, request, retries=2)
        assert len(session.calls) == 3
        assert "Timeout" in (response.error or "")

    def test_does_not_retry_non_network_errors(self) -> None:
        request = build_request("http://lab.local/{{PAYLOAD}}", "GET", "1")
        session = FakeSession([requests.exceptions.InvalidURL("nope")])
        response = send(session, request, retries=3)
        assert len(session.calls) == 1
        assert "InvalidURL" in (response.error or "")

    def test_does_not_retry_http_statuses(self) -> None:
        request = build_request("http://lab.local/{{PAYLOAD}}", "GET", "1")
        session = FakeSession([http_response(500, "err")])
        response = send(session, request, retries=3)
        assert len(session.calls) == 1
        assert response.status_code == 500


class SessionFactory:
    """Callable session factory that remembers every session it created."""

    def __init__(self) -> None:
        self.created: list[FakeSession] = []

    def __call__(self) -> FakeSession:
        session = FakeSession()
        self.created.append(session)
        return session

    @property
    def urls(self) -> list[str]:
        return [
            str(call["url"]) for session in self.created for call in session.calls
        ]


@pytest.fixture()
def factory() -> SessionFactory:
    return SessionFactory()


class TestRunBatch:
    def test_keeps_input_order(self, factory: SessionFactory, capsys) -> None:
        payloads = ["a", "b", "c", "d", "e"]
        results = run_batch(
            "http://lab.local/?q={{PAYLOAD}}",
            "GET",
            payloads,
            threads=4,
            session_factory=factory,
        )
        assert len(results) == len(payloads)
        assert all(isinstance(item, Response) for item in results)
        assert sorted(factory.urls) == sorted(
            f"http://lab.local/?q={payload}" for payload in payloads
        )
        capsys.readouterr()

    def test_prints_progress_with_counter(self, factory: SessionFactory, capsys) -> None:
        run_batch(
            "http://lab.local/{{PAYLOAD}}",
            "GET",
            ["1", "2", "3"],
            threads=1,
            session_factory=factory,
        )
        out = capsys.readouterr().out
        assert "[1/3]" in out
        assert "[2/3]" in out
        assert "[3/3]" in out

    def test_empty_payload_list_returns_empty(
        self, factory: SessionFactory, capsys
    ) -> None:
        assert run_batch("http://lab.local/{{PAYLOAD}}", "GET", [], session_factory=factory) == []
        assert factory.created == []
        capsys.readouterr()

    def test_clamps_threads_to_maximum(self, factory: SessionFactory, capsys) -> None:
        results = run_batch(
            "http://lab.local/{{PAYLOAD}}",
            "GET",
            [str(index) for index in range(20)],
            threads=MAX_THREADS * 4,
            session_factory=factory,
        )
        assert len(results) == 20
        assert len(factory.created) <= MAX_THREADS
        capsys.readouterr()

    def test_network_failure_becomes_error_response(self, capsys) -> None:
        def factory() -> FakeSession:
            return FakeSession([requests.exceptions.ConnectionError("down")])

        results = run_batch(
            "http://lab.local/{{PAYLOAD}}", "GET", ["1"], retries=0, session_factory=factory
        )
        assert results[0].status_code is None
        assert "ConnectionError" in (results[0].error or "")
        capsys.readouterr()

    def test_applies_proxy_string_to_sessions(
        self, factory: SessionFactory, capsys
    ) -> None:
        run_batch(
            "http://lab.local/{{PAYLOAD}}",
            "GET",
            ["1", "2"],
            proxy="http://127.0.0.1:8080",
            session_factory=factory,
        )
        assert factory.created
        for session in factory.created:
            assert session.proxies == {
                "http": "http://127.0.0.1:8080",
                "https": "http://127.0.0.1:8080",
            }
        capsys.readouterr()

    def test_applies_proxy_mapping(self, factory: SessionFactory, capsys) -> None:
        proxy = {"https": "http://127.0.0.1:8081"}
        run_batch(
            "https://lab.local/{{PAYLOAD}}",
            "GET",
            ["1"],
            proxy=proxy,
            session_factory=factory,
        )
        assert factory.created[0].proxies == proxy
        capsys.readouterr()

    def test_marker_in_body_is_used_for_post(
        self, factory: SessionFactory, capsys
    ) -> None:
        run_batch(
            "http://lab.local/login",
            "POST",
            ["admin"],
            body_template="user={{PAYLOAD}}",
            threads=1,
            session_factory=factory,
        )
        assert factory.created[0].calls[0]["data"] == "user=admin"
        capsys.readouterr()

    def test_missing_marker_is_reported_per_payload(
        self, factory: SessionFactory, capsys
    ) -> None:
        results = run_batch(
            "http://lab.local/static", "GET", ["1"], session_factory=factory
        )
        assert "ValueError" in (results[0].error or "")
        capsys.readouterr()

    def test_each_thread_reuses_one_session(self, factory: SessionFactory, capsys) -> None:
        run_batch(
            "http://lab.local/{{PAYLOAD}}",
            "GET",
            [str(index) for index in range(8)],
            threads=2,
            session_factory=factory,
        )
        assert 1 <= len(factory.created) <= 2
        capsys.readouterr()
