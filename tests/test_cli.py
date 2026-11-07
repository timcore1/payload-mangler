"""Tests for the CLI in :mod:`payload_mangler.__main__`.

The integration tests spin up a real stdlib HTTP server on ``127.0.0.1`` -- that
is the only host the scope guard allows anyway -- so the whole path (scope ->
mutation -> baseline -> batch -> table -> JSON) is exercised without touching
the network outside the loopback interface.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from payload_mangler import __main__ as cli
from payload_mangler.detector import Verdict
from payload_mangler.report import Finding


class LabHandler(BaseHTTPRequestHandler):
    """Lab stand: blocks UNION, crashes on quotes, echoes everything else.

    ``q=hidden`` answers with a fixed page that does *not* echo the payload, so
    tests can exercise a ``PASSTHROUGH`` reply whose oracle still fires.
    ``seen`` records ``(method, raw)`` pairs so tests can prove what was sent.
    """

    seen: list[tuple[str, str]] = []

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        type(self).seen.append(("GET", self.path))
        self._respond(parse_qs(urlparse(self.path).query))

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode("utf-8")
        type(self).seen.append(("POST", body))
        self._respond(parse_qs(body))

    def _respond(self, params: dict[str, list[str]]) -> None:
        """Classify the single ``q`` parameter the way the stand does."""
        query = params.get("q", [""])[0]
        if "UNION" in query.upper():
            self._send(403, "<h1>Unexpected path</h1>")
        elif query == "hidden":
            self._send(200, "<html>secret panel</html>")
        elif "'" in query:
            self._send(500, f'SQLITE_ERROR: near "{query}": syntax error')
        else:
            self._send(200, f"<html>results for: {query}</html>")

    def _send(self, status: int, body: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: object) -> None:
        """Silence per-request logging."""


@pytest.fixture()
def lab_url() -> Iterator[str]:
    """Run the fake stand and yield its base URL (127.0.0.1 only)."""
    LabHandler.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), LabHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture()
def payload_file(tmp_path: Path) -> Path:
    path = tmp_path / "payloads.txt"
    path.write_text(
        "# lab payloads\n' OR 1=1--\n\n' UNION SELECT NULL--\n", encoding="utf-8"
    )
    return path


BASE_ARGS = ["--delay", "0", "--threads", "1", "--no-color"]


class TestParser:
    def test_defaults(self) -> None:
        args = cli.build_parser().parse_args(["--url", "http://127.0.0.1/{{PAYLOAD}}", "--payload", "1"])
        assert args.method == "GET"
        assert args.threads == 4
        assert args.delay == 0.1
        assert args.mutations == "all"
        assert args.chain is False
        assert args.no_color is False
        assert args.report is None
        assert args.header == []

    def test_method_is_upper_cased(self) -> None:
        args = cli.build_parser().parse_args(
            ["--url", "http://127.0.0.1/{{PAYLOAD}}", "--payload", "1", "--method", "post"]
        )
        assert args.method == "POST"

    def test_url_is_required(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["--payload", "1"])

    def test_payload_source_is_required(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["--url", "http://127.0.0.1/{{PAYLOAD}}"])

    def test_payload_sources_are_mutually_exclusive(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(
                ["--url", "http://127.0.0.1/{{PAYLOAD}}", "--payload", "1",
                 "--payload-file", "x.txt"]
            )

    @pytest.mark.parametrize("threads", ["0", "17", "-2", "abc"])
    def test_threads_outside_range_is_rejected(self, threads: str) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(
                ["--url", "http://127.0.0.1/{{PAYLOAD}}", "--payload", "1", "--threads", threads]
            )

    def test_negative_delay_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(
                ["--url", "http://127.0.0.1/{{PAYLOAD}}", "--payload", "1", "--delay", "-1"]
            )

    def test_help_exits_zero(self, capsys) -> None:
        with pytest.raises(SystemExit) as excinfo:
            cli.build_parser().parse_args(["--help"])
        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        assert "--payload-file" in out
        assert "{{PAYLOAD}}" in out

    def test_help_contains_the_documented_example(self, capsys) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["--help"])
        out = capsys.readouterr().out
        assert "payloads/sqli.txt" in out
        assert "--oracle-regex" in out
        assert "--report out.json" in out

    def test_headers_are_repeatable(self) -> None:
        args = cli.build_parser().parse_args(
            ["--url", "http://127.0.0.1/{{PAYLOAD}}", "--payload", "1",
             "-H", "A: 1", "-H", "B: 2"]
        )
        assert args.header == ["A: 1", "B: 2"]


class TestParsingHelpers:
    def test_parse_headers(self) -> None:
        assert cli.parse_headers(["X-Lab: 1", "User-Agent:  mangler "]) == {
            "X-Lab": "1",
            "User-Agent": "mangler",
        }

    def test_parse_headers_keeps_colons_in_value(self) -> None:
        assert cli.parse_headers(["Referer: http://x:3000/"]) == {
            "Referer": "http://x:3000/"
        }

    @pytest.mark.parametrize("raw", ["no-colon", ": value", "   "])
    def test_parse_headers_rejects_invalid(self, raw: str) -> None:
        with pytest.raises(ValueError, match="invalid header"):
            cli.parse_headers([raw])

    def test_parse_cookies(self) -> None:
        assert cli.parse_cookies("a=1; b=2") == {"a": "1", "b": "2"}

    def test_parse_cookies_with_none(self) -> None:
        assert cli.parse_cookies(None) == {}

    def test_parse_cookies_skips_empty_chunks(self) -> None:
        assert cli.parse_cookies("a=1;; ") == {"a": "1"}

    @pytest.mark.parametrize("raw", ["novalue", "=1"])
    def test_parse_cookies_rejects_invalid(self, raw: str) -> None:
        with pytest.raises(ValueError, match="invalid cookie"):
            cli.parse_cookies(raw)

    def test_load_payloads_skips_comments_and_blanks(self, payload_file: Path) -> None:
        assert cli.load_payloads(payload_file) == ["' OR 1=1--", "' UNION SELECT NULL--"]

    def test_load_payloads_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            cli.load_payloads(tmp_path / "nope.txt")


class TestMutationSelection:
    def test_all_returns_every_mutator(self) -> None:
        names = cli.resolve_mutations("all")
        assert names == list(cli.MUTATIONS)
        assert len(names) == 10

    def test_all_is_case_insensitive(self) -> None:
        assert cli.resolve_mutations("ALL") == list(cli.MUTATIONS)

    def test_explicit_list_is_parsed_and_deduplicated(self) -> None:
        assert cli.resolve_mutations("url_encode, null_byte ,url_encode") == [
            "url_encode",
            "null_byte",
        ]

    def test_unknown_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown mutation"):
            cli.resolve_mutations("url_encode,nope")

    def test_empty_list_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one mutator"):
            cli.resolve_mutations(" , ")


class TestCandidates:
    def test_one_candidate_per_mutator(self) -> None:
        candidates = cli.build_candidates(["' OR 1=1--"], ["null_byte"], False)
        assert [item.mutated for item in candidates] == ["%00' OR 1=1--"]
        assert candidates[0].chain == ("null_byte",)
        assert candidates[0].original == "' OR 1=1--"

    def test_duplicate_mutants_are_dropped(self) -> None:
        candidates = cli.build_candidates(["abc"], ["url_encode", "double_url_encode"], False)
        assert [item.mutated for item in candidates] == ["abc"]

    def test_duplicates_are_dropped_across_payloads(self) -> None:
        candidates = cli.build_candidates(["abc", "abc"], ["url_encode"], False)
        assert len(candidates) == 1

    def test_order_is_preserved(self) -> None:
        candidates = cli.build_candidates(["b", "a"], ["url_encode"], False)
        assert [item.original for item in candidates] == ["b", "a"]

    def test_pairwise_chains_are_ordered_pairs(self) -> None:
        pairs = cli.pairwise_chains(["a", "b"])
        assert pairs == [("a", "b"), ("b", "a")]

    def test_pairwise_chains_skip_identical_names(self) -> None:
        assert cli.pairwise_chains(["a"]) == []

    def test_chain_flag_adds_pairs(self) -> None:
        candidates = cli.build_candidates(["'a'"], ["url_encode", "null_byte"], True)
        assert [item.chain for item in candidates] == [
            ("url_encode",),
            ("null_byte",),
            ("null_byte", "url_encode"),
        ]

    def test_chain_drops_pairs_that_reproduce_a_single_mutant(self) -> None:
        candidates = cli.build_candidates(["'a'"], ["url_encode", "null_byte"], True)
        chains = [item.chain for item in candidates]
        assert ("url_encode", "null_byte") not in chains

    def test_chain_uses_apply_chain_order(self) -> None:
        candidates = cli.build_candidates(["'a'"], ["null_byte", "url_encode"], True)
        chained = [
            item for item in candidates if item.chain == ("null_byte", "url_encode")
        ]
        assert chained[0].mutated == "%2500%27a%2500%27"
        assert chained[0].original == "'a'"

    def test_chain_does_not_duplicate_single_mutants(self) -> None:
        candidates = cli.build_candidates(["'a'"], ["url_encode", "null_byte"], True)
        mutated = [item.mutated for item in candidates]
        assert len(mutated) == len(set(mutated))


class TestSummarize:
    @staticmethod
    def finding(**overrides: object) -> Finding:
        """Build a finding with the verdict/backend flags under test."""
        defaults: dict[str, object] = {
            "original_payload": "p",
            "mutated_payload": "p",
            "verdict": Verdict.PASSTHROUGH,
            "oracle_true": False,
            "backend": False,
        }
        defaults.update(overrides)
        return Finding(**defaults)  # type: ignore[arg-type]

    def test_counts_every_bucket(self) -> None:
        findings = [
            self.finding(verdict=Verdict.BLOCKED),
            self.finding(verdict=Verdict.PASSTHROUGH, oracle_true=True),
            self.finding(backend=True),
            self.finding(verdict=Verdict.ERROR),
        ]
        assert cli.summarize(findings) == (
            "4 mutants: 1 blocked, 1 passthrough+oracle, 1 reached_backend, 1 error"
        )

    def test_passthrough_without_oracle_is_not_counted_as_hit(self) -> None:
        assert "0 passthrough+oracle" in cli.summarize([self.finding()])

    def test_empty_run(self) -> None:
        assert cli.summarize([]) == (
            "0 mutants: 0 blocked, 0 passthrough+oracle, 0 reached_backend, 0 error"
        )


class TestScopeGuard:
    def test_out_of_scope_target_exits_with_2(self, capsys) -> None:
        code = cli.main(["--url", "http://8.8.8.8/?q={{PAYLOAD}}", "--payload", "1"])
        assert code == cli.EXIT_USAGE == 2
        assert "out of scope" in capsys.readouterr().err

    def test_scope_error_names_the_rejected_host(self, capsys) -> None:
        cli.main(["--url", "http://192.168.1.1/?q={{PAYLOAD}}", "--payload", "1"])
        assert "192.168.1.1" in capsys.readouterr().err

    def test_scope_is_checked_before_any_request(self, monkeypatch) -> None:
        def no_network(*args: object, **kwargs: object) -> object:
            raise AssertionError("a session was created before the scope check")

        monkeypatch.setattr(cli.requests, "Session", no_network)
        assert cli.main(["--url", "http://8.8.8.8/?q={{PAYLOAD}}", "--payload", "1"]) == 2

    def test_loopback_passes_the_guard(self, lab_url: str) -> None:
        code = cli.main(
            ["--url", f"{lab_url}/s?q={{{{PAYLOAD}}}}", "--payload", "1", *BASE_ARGS]
        )
        assert code == cli.EXIT_OK


class TestValidationErrors:
    def test_missing_marker_exits_with_2(self, capsys) -> None:
        code = cli.main(["--url", "http://127.0.0.1:1/static", "--payload", "1"])
        assert code == 2
        assert "{{PAYLOAD}}" in capsys.readouterr().err

    def test_marker_in_body_only_is_accepted(self, lab_url: str) -> None:
        code = cli.main(
            ["--url", f"{lab_url}/login", "--method", "POST", "--data", "q={{PAYLOAD}}",
             "--payload", "1", *BASE_ARGS]
        )
        assert code == cli.EXIT_OK

    def test_unknown_mutation_exits_with_2(self, capsys) -> None:
        code = cli.main(
            ["--url", "http://127.0.0.1:1/?q={{PAYLOAD}}", "--payload", "1",
             "--mutations", "bogus"]
        )
        assert code == 2
        assert "unknown mutation" in capsys.readouterr().err

    def test_invalid_oracle_regex_exits_with_2(self, capsys) -> None:
        code = cli.main(
            ["--url", "http://127.0.0.1:1/?q={{PAYLOAD}}", "--payload", "1",
             "--oracle-regex", "(unclosed"]
        )
        assert code == 2
        assert capsys.readouterr().err.startswith("error:")

    def test_missing_payload_file_exits_with_2(self, tmp_path: Path, capsys) -> None:
        code = cli.main(
            ["--url", "http://127.0.0.1:1/?q={{PAYLOAD}}",
             "--payload-file", str(tmp_path / "nope.txt")]
        )
        assert code == 2
        assert capsys.readouterr().err

    def test_empty_payload_file_exits_with_2(self, tmp_path: Path, capsys) -> None:
        empty = tmp_path / "empty.txt"
        empty.write_text("# nothing here\n\n", encoding="utf-8")
        code = cli.main(
            ["--url", "http://127.0.0.1:1/?q={{PAYLOAD}}", "--payload-file", str(empty)]
        )
        assert code == 2
        assert "no payloads" in capsys.readouterr().err


class TestEndToEnd:
    def test_runs_batch_and_prints_table(
        self, lab_url: str, payload_file: Path, capsys
    ) -> None:
        code = cli.main(
            ["--url", f"{lab_url}/s?q={{{{PAYLOAD}}}}", "--payload-file", str(payload_file),
             "--mutations", "url_encode", *BASE_ARGS]
        )
        assert code == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "CHAIN" in out
        assert "BACKEND" in out
        assert "mutants:" in out

    def test_backend_column_marks_database_errors(self, lab_url: str, capsys) -> None:
        cli.main(
            ["--url", f"{lab_url}/s?q={{{{PAYLOAD}}}}", "--payload", "' OR 1=1--",
             "--mutations", "url_encode", *BASE_ARGS]
        )
        rows = [line for line in capsys.readouterr().out.splitlines() if "url_encode" in line]
        assert rows and "yes" in rows[0]

    def test_summary_counts_blocked_payload(self, lab_url: str, capsys) -> None:
        cli.main(
            ["--url", f"{lab_url}/s?q={{{{PAYLOAD}}}}", "--payload", "' UNION SELECT NULL--",
             "--mutations", "url_encode", *BASE_ARGS]
        )
        assert "1 blocked" in capsys.readouterr().out

    def test_report_json_is_written(
        self, lab_url: str, payload_file: Path, tmp_path: Path, capsys
    ) -> None:
        report = tmp_path / "out" / "findings.json"
        code = cli.main(
            ["--url", f"{lab_url}/s?q={{{{PAYLOAD}}}}", "--payload-file", str(payload_file),
             "--mutations", "url_encode,null_byte", "--report", str(report), *BASE_ARGS]
        )
        assert code == cli.EXIT_OK
        entries = json.loads(report.read_text(encoding="utf-8"))
        assert len(entries) == 4
        assert {"backend", "verdict", "oracle_true"} <= set(entries[0])
        assert "report:" in capsys.readouterr().out

    def test_oracle_flag_marks_passthrough_hits(self, lab_url: str, capsys) -> None:
        cli.main(
            ["--url", f"{lab_url}/s?q={{{{PAYLOAD}}}}", "--payload", "hidden",
             "--mutations", "url_encode", "--oracle-regex", "secret panel",
             *BASE_ARGS]
        )
        assert "1 passthrough+oracle" in capsys.readouterr().out

    def test_post_body_is_actually_sent(self, lab_url: str, capsys) -> None:
        cli.main(
            ["--url", f"{lab_url}/login", "--method", "POST", "--data", "q={{PAYLOAD}}",
             "--payload", "hidden", "--mutations", "url_encode",
             "--oracle-regex", "secret panel", *BASE_ARGS]
        )
        assert "1 passthrough+oracle" in capsys.readouterr().out
        assert ("POST", "q=hidden") in LabHandler.seen
        assert not any(method == "GET" and "hidden" in raw for method, raw in LabHandler.seen)

    def test_chain_mode_adds_pairwise_mutants(
        self, lab_url: str, tmp_path: Path, capsys
    ) -> None:
        def mutant_count(extra: list[str], name: str) -> int:
            report = tmp_path / f"{name}.json"
            cli.main(
                ["--url", f"{lab_url}/s?q={{{{PAYLOAD}}}}", "--payload", "'a'",
                 "--mutations", "url_encode,null_byte", "--report", str(report),
                 *extra, *BASE_ARGS]
            )
            capsys.readouterr()
            return len(json.loads(report.read_text(encoding="utf-8")))

        assert mutant_count([], "singles") == 2
        assert mutant_count(["--chain"], "chained") == 3

    def test_no_color_output_has_no_ansi(self, lab_url: str, capsys) -> None:
        cli.main(
            ["--url", f"{lab_url}/s?q={{{{PAYLOAD}}}}", "--payload", "' UNION SELECT NULL--",
             "--mutations", "url_encode", *BASE_ARGS]
        )
        assert "\033[" not in capsys.readouterr().out
