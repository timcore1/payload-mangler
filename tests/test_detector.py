"""Unit tests for :mod:`payload_mangler.detector`.

Sessions are fakes that return real ``requests.Response`` objects (status and
body configured directly), so classification logic is tested without network.
"""

from __future__ import annotations

import pytest
import requests

from payload_mangler.detector import (
    BACKEND_SIGNALS,
    BLOCK_SIGNATURES,
    BLOCK_STATUS_CODES,
    Analysis,
    Verdict,
    analyze,
    backend_signals_in,
    baseline,
    classify,
    reached_backend,
    reflection_forms,
    url_decode,
)
from payload_mangler.engine import Response


def http_response(status_code: int = 200, body: str = "") -> requests.Response:
    """Build a real ``requests.Response`` with the given status and body."""
    response = requests.Response()
    response.status_code = status_code
    response._content = body.encode("utf-8")
    response.encoding = "utf-8"
    response.url = "http://lab.local/"
    return response


class FakeSession:
    """Session stub returning pre-canned responses (or raising them)."""

    def __init__(self, items: list[requests.Response | Exception]) -> None:
        self.items = list(items)
        self.calls: list[dict[str, object]] = []

    def request(self, **kwargs: object) -> requests.Response:
        self.calls.append(kwargs)
        item = self.items.pop(0) if self.items else http_response()
        if isinstance(item, Exception):
            raise item
        return item


def engine_response(
    status_code: int | None = 200, body: str = "", error: str | None = None
) -> Response:
    """Build an engine-level :class:`Response` for classifier tests."""
    return Response(
        status_code=status_code,
        body=body,
        elapsed_ms=1.5,
        size=len(body.encode("utf-8")),
        error=error,
    )


NEUTRAL = engine_response(200, "<html>search results: 0</html>")


class TestVerdict:
    def test_has_exactly_four_members(self) -> None:
        assert {member.name for member in Verdict} == {
            "BLOCKED",
            "REFLECTED",
            "PASSTHROUGH",
            "ERROR",
        }

    @pytest.mark.parametrize(
        ("member", "value"),
        [
            (Verdict.BLOCKED, "blocked"),
            (Verdict.REFLECTED, "reflected"),
            (Verdict.PASSTHROUGH, "passthrough"),
            (Verdict.ERROR, "error"),
        ],
    )
    def test_values_are_lowercase_strings(self, member: Verdict, value: str) -> None:
        assert member == value

    def test_is_string_comparable(self) -> None:
        assert isinstance(Verdict.BLOCKED, str)


class TestBlockSignatures:
    def test_is_a_non_empty_list_of_strings(self) -> None:
        assert isinstance(BLOCK_SIGNATURES, list)
        assert BLOCK_SIGNATURES
        assert all(isinstance(item, str) for item in BLOCK_SIGNATURES)

    def test_contains_the_known_lab_signature(self) -> None:
        assert "Unexpected path" in BLOCK_SIGNATURES

    def test_block_statuses_are_403_and_406(self) -> None:
        assert BLOCK_STATUS_CODES == {403, 406}


class TestUrlDecodeAndReflectionForms:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("%3Cscript%3E", "<script>"),
            ("%27%20OR%201%3D1--", "' OR 1=1--"),
            ("plain", "plain"),
        ],
    )
    def test_url_decode(self, value: str, expected: str) -> None:
        assert url_decode(value) == expected

    def test_reflection_forms_include_raw_and_decoded(self) -> None:
        assert reflection_forms("%3Cb%3E") == ("%3Cb%3E", "<b>")

    def test_reflection_forms_drop_duplicates(self) -> None:
        assert reflection_forms("plain") == ("plain",)

    def test_reflection_forms_accept_precomputed_decoded_value(self) -> None:
        assert reflection_forms("%3Cb%3E", "<b>") == ("%3Cb%3E", "<b>")

    @pytest.mark.parametrize("payload", ["", "   ", "\n"])
    def test_reflection_forms_ignore_blank_payloads(self, payload: str) -> None:
        assert reflection_forms(payload) == ()


class TestBaseline:
    def test_sends_empty_payload(self) -> None:
        session = FakeSession([http_response(200, "neutral")])
        result = baseline(session, "http://lab.local/?q={{PAYLOAD}}")
        assert session.calls[0]["url"] == "http://lab.local/?q="
        assert result.status_code == 200
        assert result.body == "neutral"

    def test_uses_given_method_and_headers(self) -> None:
        session = FakeSession([http_response(200, "")])
        baseline(
            session,
            "http://lab.local/login",
            "POST",
            headers={"X-Lab": "1"},
            body_template="user={{PAYLOAD}}",
        )
        assert session.calls[0]["method"] == "POST"
        assert session.calls[0]["data"] == "user="

    def test_reports_transport_error_as_response(self) -> None:
        session = FakeSession([requests.exceptions.ConnectionError("down")])
        result = baseline(session, "http://lab.local/{{PAYLOAD}}", retries=0)
        assert result.status_code is None
        assert "ConnectionError" in (result.error or "")


class TestClassify:
    def test_transport_error_is_error_verdict(self) -> None:
        broken = engine_response(None, "", error="ConnectionError: down")
        assert classify(broken, NEUTRAL, "' OR 1=1--") is Verdict.ERROR

    @pytest.mark.parametrize("status", [403, 406])
    def test_block_status_codes_are_blocked(self, status: int) -> None:
        response = engine_response(status, "<html>nope</html>")
        assert classify(response, NEUTRAL, "' OR 1=1--") is Verdict.BLOCKED

    def test_block_signature_with_status_200_is_blocked(self) -> None:
        response = engine_response(
            200, "<h1>Unexpected path</h1><p>request denied</p>"
        )
        assert classify(response, NEUTRAL, "' OR 1=1--") is Verdict.BLOCKED

    def test_block_signature_matching_is_case_insensitive(self) -> None:
        response = engine_response(200, "unexpected PATH detected")
        assert classify(response, NEUTRAL, "1=1") is Verdict.BLOCKED

    def test_blocking_wins_over_reflection(self) -> None:
        response = engine_response(403, "blocked: ' OR 1=1--")
        assert classify(response, NEUTRAL, "' OR 1=1--") is Verdict.BLOCKED

    def test_raw_payload_in_body_is_reflected(self) -> None:
        response = engine_response(200, "<p>you searched: ' OR 1=1--</p>")
        assert classify(response, NEUTRAL, "' OR 1=1--") is Verdict.REFLECTED

    def test_url_decoded_payload_in_body_is_reflected(self) -> None:
        response = engine_response(200, "<script>alert(1)</script>")
        verdict = classify(response, NEUTRAL, "%3Cscript%3Ealert(1)%3C/script%3E")
        assert verdict is Verdict.REFLECTED

    def test_precomputed_decoded_payload_is_accepted(self) -> None:
        response = engine_response(200, "hello alert(1) world")
        verdict = classify(
            response, NEUTRAL, "%61lert(1)", decoded_payload="alert(1)"
        )
        assert verdict is Verdict.REFLECTED

    def test_clean_response_is_passthrough(self) -> None:
        response = engine_response(200, "<html>search results: 0</html>")
        assert classify(response, NEUTRAL, "' OR 1=1--") is Verdict.PASSTHROUGH

    def test_empty_payload_is_never_reflected(self) -> None:
        response = engine_response(200, "<html>search results: 0</html>")
        assert classify(response, NEUTRAL, "") is Verdict.PASSTHROUGH

    def test_signature_already_in_baseline_is_not_blocking(self) -> None:
        page = "<html><footer>Forbidden fruit recipes</footer></html>"
        response = engine_response(200, page)
        neutral = engine_response(200, page)
        assert classify(response, neutral, "1=1") is Verdict.PASSTHROUGH

    def test_signature_missing_from_baseline_is_blocking(self) -> None:
        response = engine_response(200, "<html>Request blocked</html>")
        neutral = engine_response(200, "<html>ok</html>")
        assert classify(response, neutral, "1=1") is Verdict.BLOCKED

    def test_baseline_is_optional(self) -> None:
        response = engine_response(200, "<h1>Unexpected path</h1>")
        assert classify(response, None, "1=1") is Verdict.BLOCKED

    def test_other_error_status_is_not_blocked(self) -> None:
        response = engine_response(500, "internal error")
        assert classify(response, NEUTRAL, "1=1") is Verdict.PASSTHROUGH

    def test_all_four_verdicts_are_reachable(self) -> None:
        cases = {
            Verdict.ERROR: classify(engine_response(None, error="boom"), NEUTRAL, "1"),
            Verdict.BLOCKED: classify(engine_response(403, "no"), NEUTRAL, "1"),
            Verdict.REFLECTED: classify(engine_response(200, "1"), NEUTRAL, "1"),
            Verdict.PASSTHROUGH: classify(engine_response(200, "zzz"), NEUTRAL, "1"),
        }
        assert {verdict for verdict in cases} == set(Verdict)


SQLITE_BODY = (
    "Error: SQLITE_ERROR: near \"')) OR 1=1--'\": syntax error"
)


class TestBackendSignals:
    def test_contains_the_five_lab_signals(self) -> None:
        assert BACKEND_SIGNALS == [
            "SQLITE_ERROR",
            "SQL syntax",
            "sqlite3.OperationalError",
            "mysql_fetch",
            "ORA-0",
        ]

    @pytest.mark.parametrize("signal", BACKEND_SIGNALS)
    def test_every_signal_is_detected(self, signal: str) -> None:
        assert reached_backend(engine_response(500, f"<pre>{signal}: boom</pre>"))

    def test_sqlite_error_with_status_500_is_detected(self) -> None:
        response = engine_response(500, SQLITE_BODY)
        assert reached_backend(response) is True

    def test_sqlite_error_with_status_200_is_detected(self) -> None:
        assert reached_backend(engine_response(200, SQLITE_BODY)) is True

    def test_clean_response_is_not_detected(self) -> None:
        response = engine_response(200, "<html>search results: 0</html>")
        assert reached_backend(response) is False

    def test_empty_body_is_not_detected(self) -> None:
        assert reached_backend(engine_response(200, "")) is False

    def test_transport_error_is_not_detected(self) -> None:
        response = engine_response(None, "", error="ConnectionError: down")
        assert reached_backend(response) is False

    def test_matching_is_case_sensitive(self) -> None:
        assert reached_backend(engine_response(500, "sqlite_error: boom")) is False

    def test_plain_prose_does_not_trigger(self) -> None:
        body = "<html>we use SQLite and SQL, no errors here</html>"
        assert reached_backend(engine_response(200, body)) is False

    def test_backend_signals_in_reports_which_matched(self) -> None:
        body = "SQLITE_ERROR: SQL syntax error near 'x'"
        assert backend_signals_in(body) == ("SQLITE_ERROR", "SQL syntax")

    def test_backend_signals_in_returns_empty_tuple_when_clean(self) -> None:
        assert backend_signals_in("<html>ok</html>") == ()


class TestBackendSignalDoesNotChangeVerdict:
    def test_500_with_sqlite_error_stays_passthrough(self) -> None:
        response = engine_response(500, SQLITE_BODY)
        assert classify(response, NEUTRAL, "' OR 1=1--") is Verdict.PASSTHROUGH

    def test_500_with_sqlite_error_is_still_not_blocked(self) -> None:
        response = engine_response(500, SQLITE_BODY)
        assert classify(response, NEUTRAL, "' OR 1=1--") is not Verdict.BLOCKED

    def test_block_status_still_wins_over_backend_signal(self) -> None:
        response = engine_response(403, "Unexpected path: SQLITE_ERROR")
        assert classify(response, NEUTRAL, "1=1") is Verdict.BLOCKED
        assert reached_backend(response) is True

    def test_error_verdict_still_wins(self) -> None:
        response = engine_response(None, SQLITE_BODY, error="Timeout: slow")
        assert classify(response, NEUTRAL, "1") is Verdict.ERROR
        assert reached_backend(response) is False


class TestAnalyze:
    def test_returns_verdict_and_flag_together(self) -> None:
        response = engine_response(500, SQLITE_BODY)
        result = analyze(response, NEUTRAL, "' OR 1=1--")
        assert isinstance(result, Analysis)
        assert result.verdict is Verdict.PASSTHROUGH
        assert result.reached_backend is True
        assert result.signals == ("SQLITE_ERROR",)

    def test_bypass_flag_is_true_for_passthrough_with_backend_signal(self) -> None:
        result = analyze(engine_response(500, SQLITE_BODY), NEUTRAL, "1 OR 1=1")
        assert result.is_bypass is True

    def test_bypass_flag_is_false_for_blocked_response(self) -> None:
        response = engine_response(403, "Request blocked: SQLITE_ERROR")
        result = analyze(response, NEUTRAL, "1 OR 1=1")
        assert result.verdict is Verdict.BLOCKED
        assert result.is_bypass is False

    def test_clean_response_has_no_signals(self) -> None:
        result = analyze(engine_response(200, "<html>ok</html>"), NEUTRAL, "1")
        assert result.verdict is Verdict.PASSTHROUGH
        assert result.reached_backend is False
        assert result.signals == ()

    def test_matches_separate_calls(self) -> None:
        response = engine_response(500, SQLITE_BODY)
        result = analyze(response, NEUTRAL, "' OR 1=1--")
        assert result.verdict == classify(response, NEUTRAL, "' OR 1=1--")
        assert result.reached_backend == reached_backend(response)
