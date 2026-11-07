"""Unit tests for :mod:`payload_mangler.report`.

Covers the two deliverables: a round-trippable JSON export and an aligned
console table whose colouring is opt-out. Table assertions run with
``use_color=False`` for exact text, plus explicit checks that the ANSI codes
appear/disappear as documented.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from payload_mangler.detector import Verdict
from payload_mangler.report import (
    GREEN,
    HEADERS,
    RED,
    RESET,
    Finding,
    color_enabled,
    print_table,
    save_json,
)


def make_finding(**overrides: object) -> Finding:
    """Build a finding with sane defaults, overridable per test."""
    defaults: dict[str, object] = {
        "original_payload": "' OR 1=1--",
        "mutated_payload": "%27%20OR%201=1--",
        "chain": ["url_encode"],
        "verdict": Verdict.PASSTHROUGH,
        "status_code": 200,
        "size": 142,
        "elapsed_ms": 12.34,
        "oracle_true": False,
    }
    defaults.update(overrides)
    return Finding(**defaults)  # type: ignore[arg-type]


class TestFinding:
    def test_holds_all_requested_fields(self) -> None:
        finding = make_finding()
        assert finding.original_payload == "' OR 1=1--"
        assert finding.mutated_payload == "%27%20OR%201=1--"
        assert finding.chain == ["url_encode"]
        assert finding.verdict is Verdict.PASSTHROUGH
        assert finding.status_code == 200
        assert finding.size == 142
        assert finding.elapsed_ms == 12.34
        assert finding.oracle_true is False

    def test_chain_defaults_to_empty_list(self) -> None:
        finding = Finding(original_payload="a", mutated_payload="a")
        assert finding.chain == []

    def test_chain_label_joins_with_plus(self) -> None:
        finding = make_finding(chain=["url_encode", "null_byte"])
        assert finding.chain_label == "url_encode+null_byte"

    def test_chain_label_is_dash_when_chain_is_empty(self) -> None:
        assert make_finding(chain=[]).chain_label == "-"

    def test_chain_label_truncates_long_chains(self) -> None:
        finding = make_finding(
            chain=["comment_inject", "whitespace_swap", "null_byte"]
        )
        assert len(finding.chain_label) <= 28
        assert finding.chain_label.endswith("...")

    def test_to_dict_flattens_verdict_to_string(self) -> None:
        payload = make_finding().to_dict()
        assert payload["verdict"] == "passthrough"
        assert isinstance(payload["verdict"], str)
        assert payload["chain"] == ["url_encode"]

    def test_to_dict_is_json_serialisable(self) -> None:
        text = json.dumps(make_finding().to_dict())
        assert json.loads(text)["status_code"] == 200


class TestSaveJson:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        target = tmp_path / "findings.json"
        save_json([make_finding()], target)
        assert json.loads(target.read_text(encoding="utf-8"))[0]["verdict"] == (
            "passthrough"
        )

    def test_round_trips_all_fields(self, tmp_path: Path) -> None:
        finding = make_finding(verdict=Verdict.BLOCKED, oracle_true=True)
        target = tmp_path / "round.json"
        save_json([finding], target)
        restored = json.loads(target.read_text(encoding="utf-8"))[0]
        assert restored == finding.to_dict()

    def test_preserves_order_of_findings(self, tmp_path: Path) -> None:
        findings = [make_finding(original_payload=f"p{index}") for index in range(3)]
        target = tmp_path / "order.json"
        save_json(findings, target)
        restored = json.loads(target.read_text(encoding="utf-8"))
        assert [item["original_payload"] for item in restored] == ["p0", "p1", "p2"]

    def test_uses_indentation(self, tmp_path: Path) -> None:
        target = tmp_path / "indent.json"
        save_json([make_finding()], target)
        text = target.read_text(encoding="utf-8")
        assert '\n    "original_payload"' in text
        assert text.startswith("[\n  {")

    def test_empty_list_writes_empty_array(self, tmp_path: Path) -> None:
        target = tmp_path / "empty.json"
        save_json([], target)
        assert json.loads(target.read_text(encoding="utf-8")) == []

    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "dir" / "run.json"
        save_json([make_finding()], target)
        assert target.exists()

    def test_returns_the_written_path(self, tmp_path: Path) -> None:
        target = tmp_path / "returned.json"
        assert save_json([], target) == target

    def test_keeps_non_ascii_readable(self, tmp_path: Path) -> None:
        target = tmp_path / "utf8.json"
        save_json([make_finding(original_payload="' OR 1=1-- /* тест */")], target)
        text = target.read_text(encoding="utf-8")
        assert "тест" in text
        assert json.loads(text)[0]["original_payload"].endswith("тест */")


class TestColorEnabled:
    def test_explicit_false_disables_colour(self) -> None:
        assert color_enabled(False) is False

    def test_explicit_true_enables_colour(self) -> None:
        assert color_enabled(True) is True

    def test_explicit_value_wins_over_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NO_COLOR", "1")
        assert color_enabled(True) is True

    def test_no_color_environment_variable_disables_colour(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NO_COLOR", "1")
        assert color_enabled(None) is False

    def test_colour_is_on_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NO_COLOR", raising=False)
        assert color_enabled(None) is True


class TestPrintTable:
    def test_prints_all_requested_columns(self, capsys) -> None:
        print_table([make_finding()], use_color=False)
        header = capsys.readouterr().out.splitlines()[0]
        for column in ("CHAIN", "VERDICT", "ORACLE", "STATUS", "SIZE", "MS"):
            assert column in header
        assert header.split()[0] == HEADERS[0]

    def test_prints_one_row_per_finding(self, capsys) -> None:
        findings = [make_finding(), make_finding(), make_finding()]
        print_table(findings, use_color=False)
        lines = capsys.readouterr().out.strip().splitlines()
        assert len(lines) == 2 + len(findings)

    def test_rows_are_numbered(self, capsys) -> None:
        print_table([make_finding(), make_finding()], use_color=False)
        body = capsys.readouterr().out.strip().splitlines()[2:]
        assert body[0].split()[0] == "1"
        assert body[1].split()[0] == "2"

    def test_columns_stay_aligned(self, capsys) -> None:
        findings = [
            make_finding(chain=["url_encode"]),
            make_finding(chain=["comment_inject", "null_byte"], size=7),
        ]
        print_table(findings, use_color=False)
        lines = capsys.readouterr().out.strip().splitlines()
        positions = {line.index("PASSTHROUGH") for line in lines[2:]}
        assert len(positions) == 1

    def test_blocked_row_is_red(self, capsys) -> None:
        print_table([make_finding(verdict=Verdict.BLOCKED)], use_color=True)
        assert f"{RED}BLOCKED{RESET}" in capsys.readouterr().out

    def test_passthrough_with_oracle_is_green(self, capsys) -> None:
        finding = make_finding(verdict=Verdict.PASSTHROUGH, oracle_true=True)
        print_table([finding], use_color=True)
        assert GREEN in capsys.readouterr().out

    def test_passthrough_without_oracle_is_not_green(self, capsys) -> None:
        print_table([make_finding(oracle_true=False)], use_color=True)
        assert GREEN not in capsys.readouterr().out

    def test_no_color_flag_emits_no_ansi_codes(self, capsys) -> None:
        findings = [
            make_finding(verdict=Verdict.BLOCKED),
            make_finding(oracle_true=True),
        ]
        print_table(findings, use_color=False)
        assert "\033[" not in capsys.readouterr().out

    def test_no_color_output_is_identical_to_stripped_colour(self, capsys) -> None:
        findings = [make_finding(verdict=Verdict.BLOCKED), make_finding()]
        print_table(findings, use_color=False)
        plain = capsys.readouterr().out
        print_table(findings, use_color=True)
        coloured = capsys.readouterr().out
        assert coloured.replace(RED, "").replace(GREEN, "").replace(
            RESET, ""
        ).replace("\033[2m", "").replace("\033[1m", "") == plain

    def test_error_status_is_rendered_as_err(self, capsys) -> None:
        print_table([make_finding(status_code=None)], use_color=False)
        assert "ERR" in capsys.readouterr().out

    def test_empty_list_still_prints_header(self, capsys) -> None:
        print_table([], use_color=False)
        lines = capsys.readouterr().out.strip().splitlines()
        assert len(lines) == 2
        assert "VERDICT" in lines[0]

    def test_oracle_column_shows_yes_and_no(self, capsys) -> None:
        print_table(
            [make_finding(oracle_true=True), make_finding(oracle_true=False)],
            use_color=False,
        )
        body = capsys.readouterr().out.strip().splitlines()[2:]
        assert body[0].split()[3] == "yes"
        assert body[1].split()[3] == "no"

    def test_elapsed_is_rendered_with_one_decimal(self, capsys) -> None:
        print_table([make_finding(elapsed_ms=12.345)], use_color=False)
        assert "12.3" in capsys.readouterr().out
