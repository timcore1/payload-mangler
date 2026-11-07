"""Unit tests for :mod:`payload_mangler.oracle`."""

from __future__ import annotations

import re

import pytest

from payload_mangler.engine import Response
from payload_mangler.oracle import make_oracle


def response(body: str, status_code: int = 200, error: str | None = None) -> Response:
    """Build an engine-level response for oracle checks."""
    return Response(
        status_code=status_code,
        body=body,
        elapsed_ms=1.0,
        size=len(body.encode("utf-8")),
        error=error,
    )


class TestNoPattern:
    def test_none_pattern_always_returns_true(self) -> None:
        oracle = make_oracle(None)
        assert oracle(response("")) is True
        assert oracle(response("anything at all")) is True

    def test_none_pattern_is_true_even_for_blocked_response(self) -> None:
        assert make_oracle(None)(response("blocked", status_code=403)) is True


class TestPatternOracle:
    def test_matches_substring_anywhere_in_body(self) -> None:
        oracle = make_oracle("SQLITE_ERROR")
        assert oracle(response("Error: SQLITE_ERROR: near \"x\"")) is True

    def test_does_not_match_absent_text(self) -> None:
        assert make_oracle("SQLITE_ERROR")(response("<html>ok</html>")) is False

    def test_supports_real_regex_syntax(self) -> None:
        oracle = make_oracle(r"alert\(\d+\)")
        assert oracle(response("...alert(1)...")) is True
        assert oracle(response("...alert(x)...")) is False

    def test_anchor_and_case_flag_are_honoured(self) -> None:
        assert make_oracle("^SQLITE_ERROR")(response("SQLITE_ERROR: boom")) is True
        assert make_oracle("^SQLITE_ERROR")(response("x SQLITE_ERROR")) is False
        assert make_oracle("(?i)sqlite_error")(response("Sqlite_Error")) is True

    def test_is_case_sensitive_by_default(self) -> None:
        assert make_oracle("SQLITE_ERROR")(response("sqlite_error")) is False

    def test_empty_pattern_matches_everything(self) -> None:
        assert make_oracle("")(response("")) is True

    def test_handles_empty_body(self) -> None:
        assert make_oracle("x")(response("")) is False

    def test_rejects_invalid_pattern(self) -> None:
        with pytest.raises(re.error):
            make_oracle("(unclosed")

    def test_is_reusable_and_stateless(self) -> None:
        oracle = make_oracle("boom")
        assert [oracle(response("boom")), oracle(response("quiet"))] == [True, False]
