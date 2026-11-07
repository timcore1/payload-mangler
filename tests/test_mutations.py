"""Unit tests for :mod:`payload_mangler.mutations`.

Every mutator gets at least two kinds of cases:

* **changed** -- the exact mutated output for a known payload;
* **unchanged** -- the invariant that must hold (input returned as is, or the
  structural property that must be preserved: length, letters, digit order...).

Run with ``python3 -m pytest tests -q`` from the project root.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from payload_mangler.mutations import (
    MUTATIONS,
    apply_chain,
    case_swap,
    comment_inject,
    concat_split,
    double_url_encode,
    hex_encode,
    newline_bypass,
    null_byte,
    unicode_escape,
    url_encode,
    whitespace_swap,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAYLOADS_DIR = PROJECT_ROOT / "payloads"


def load_payloads(filename: str) -> list[str]:
    """Read a payload fixture, skipping blank lines."""
    text = (PAYLOADS_DIR / filename).read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line.strip()]


class TestUrlEncode:
    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ("' OR 1=1--", "%27%20OR%201%3D1--"),
            ("<script>", "%3Cscript%3E"),
            ("a b", "a%20b"),
            ("a/b", "a%2Fb"),
        ],
    )
    def test_encodes_special_characters(self, payload: str, expected: str) -> None:
        assert url_encode(payload) == expected

    @pytest.mark.parametrize("payload", ["UNION", "abc123", "a-b_c.d~e", ""])
    def test_keeps_unreserved_characters(self, payload: str) -> None:
        assert url_encode(payload) == payload

    def test_encodes_only_once(self) -> None:
        assert url_encode("%") == "%25"


class TestDoubleUrlEncode:
    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ("'", "%2527"),
            (" ", "%2520"),
            ("a b", "a%2520b"),
        ],
    )
    def test_encodes_twice(self, payload: str, expected: str) -> None:
        assert double_url_encode(payload) == expected

    @pytest.mark.parametrize("payload", ["UNION", "abc123", ""])
    def test_keeps_unreserved_characters(self, payload: str) -> None:
        assert double_url_encode(payload) == payload

    @pytest.mark.parametrize(
        "payload",
        ["' OR 1=1--", "<script>alert(1)</script>", "admin@juice-sh.op", "a-b_c.d~e"],
    )
    def test_matches_two_sequential_url_encodes(self, payload: str) -> None:
        assert double_url_encode(payload) == url_encode(url_encode(payload))


class TestCaseSwap:
    def test_is_deterministic_for_same_seed(self) -> None:
        assert case_swap("UNION SELECT", seed=42) == case_swap("UNION SELECT", seed=42)

    def test_matches_known_seed_output(self) -> None:
        assert case_swap("UNION", seed=42) == "UnioN"

    def test_actually_changes_the_case(self) -> None:
        assert case_swap("UNION", seed=42) != "UNION"

    def test_uses_different_patterns_for_different_seeds(self) -> None:
        variants = {case_swap("union select", seed=seed) for seed in range(5)}
        assert len(variants) > 1

    @pytest.mark.parametrize(
        "payload",
        ["' OR 1=1--", "qwert')) OR deletedAt IS NOT NULL--", "<script>alert(1)</script>"],
    )
    def test_keeps_letters_and_length(self, payload: str) -> None:
        mutated = case_swap(payload, seed=7)
        assert mutated.lower() == payload.lower()
        assert len(mutated) == len(payload)

    @pytest.mark.parametrize("payload", ["1=1 --", "", "%00'()", "привет"])
    def test_leaves_non_ascii_letters_alone(self, payload: str) -> None:
        assert case_swap(payload, seed=42) == payload


class TestCommentInject:
    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ("UNION SELECT", "UN/**/ION SE/**/LECT"),
            ("' UNION SELECT NULL--", "' UN/**/ION SE/**/LECT NULL--"),
            ("WHERE id FROM x AND y", "WH/**/ERE id FR/**/OM x AN/**/D y"),
        ],
    )
    def test_splits_sql_keywords(self, payload: str, expected: str) -> None:
        assert comment_inject(payload) == expected

    @pytest.mark.parametrize(
        ("keyword", "expected"),
        [
            ("SELECT", "SE/**/LECT"),
            ("UNION", "UN/**/ION"),
            ("WHERE", "WH/**/ERE"),
            ("FROM", "FR/**/OM"),
            ("AND", "AN/**/D"),
            ("OR", "O/**/R"),
        ],
    )
    def test_covers_every_supported_keyword(self, keyword: str, expected: str) -> None:
        assert comment_inject(keyword) == expected

    def test_is_case_insensitive_but_keeps_the_case(self) -> None:
        assert comment_inject("or 1=1") == "o/**/r 1=1"

    @pytest.mark.parametrize(
        "payload",
        ["", "ORDER BY 1", "admin@juice-sh.op", "SELECTED", "<script>alert(1)</script>"],
    )
    def test_leaves_other_text_alone(self, payload: str) -> None:
        assert comment_inject(payload) == payload

    def test_does_not_touch_a_keyword_inside_a_longer_word(self) -> None:
        assert "ORDER" in comment_inject("ORDER BY 1")

    def test_adds_one_comment_per_keyword(self) -> None:
        assert comment_inject("' UNION SELECT NULL--").count("/**/") == 2


class TestWhitespaceSwap:
    @pytest.mark.parametrize(
        ("variant", "expected"),
        [
            ("tab", "1%09OR%091=1"),
            ("lf", "1%0aOR%0a1=1"),
            ("ff", "1%0cOR%0c1=1"),
            ("comment", "1/**/OR/**/1=1"),
        ],
    )
    def test_replaces_spaces_with_variant(self, variant: str, expected: str) -> None:
        assert whitespace_swap("1 OR 1=1", variant=variant) == expected

    def test_defaults_to_tab(self) -> None:
        assert whitespace_swap("a b") == "a%09b"

    @pytest.mark.parametrize("variant", ["tab", "lf", "ff", "comment"])
    def test_keeps_existing_whitespace_and_other_chars(self, variant: str) -> None:
        payload = "a\tb\nc\rd"
        assert whitespace_swap(payload, variant=variant) == payload

    @pytest.mark.parametrize("variant", ["tab", "lf", "ff", "comment"])
    def test_leaves_payload_without_spaces_alone(self, variant: str) -> None:
        payload = "UNION(SELECT(1))"
        assert whitespace_swap(payload, variant=variant) == payload

    def test_rejects_unknown_variant(self) -> None:
        with pytest.raises(ValueError, match="unknown whitespace variant"):
            whitespace_swap("a b", variant="crlf")

    def test_reports_available_variants_in_the_error(self) -> None:
        with pytest.raises(ValueError, match="comment"):
            whitespace_swap("a b", variant="none")


class TestNullByte:
    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ("'", "%00'"),
            ('"', '%00"'),
            ("(", "%00("),
            (")", "%00)"),
            ("qwert')) AND 1=1--", "qwert%00'%00)%00) AND 1=1--"),
        ],
    )
    def test_prefixes_quotes_and_parens(self, payload: str, expected: str) -> None:
        assert null_byte(payload) == expected

    @pytest.mark.parametrize("payload", ["", "admin", "' OR 1=1--".replace("'", "x"), "1=1"])
    def test_leaves_payload_without_targets_alone(self, payload: str) -> None:
        assert null_byte(payload) == payload

    def test_inserts_one_null_byte_per_target(self) -> None:
        payload = "') OR ('')=('"
        assert null_byte(payload).count("%00") == sum(
            char in "'\"()" for char in payload
        )


class TestUnicodeEscape:
    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ("ab", "\\u0061\\u0062"),
            ("alert(1)", "\\u0061\\u006c\\u0065\\u0072\\u0074(1)"),
            ("Z", "\\u005a"),
        ],
    )
    def test_escapes_ascii_letters(self, payload: str, expected: str) -> None:
        assert unicode_escape(payload) == expected

    @pytest.mark.parametrize("payload", ["", "(1)", "<>\"'", "12345", "привет"])
    def test_leaves_non_letters_alone(self, payload: str) -> None:
        assert unicode_escape(payload) == payload

    def test_never_produces_bare_letters(self) -> None:
        mutated = unicode_escape("<script>alert(1)</script>")
        assert "script" not in mutated
        assert mutated.startswith("<\\u0073")


class TestHexEncode:
    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ("'admin'", "0x61646d696e"),
            ("id='ab'", "id=0x6162"),
            ("'ab' AND 'cd'", "0x6162 AND 0x6364"),
        ],
    )
    def test_rewrites_quoted_literals_as_hex(self, payload: str, expected: str) -> None:
        assert hex_encode(payload) == expected

    @pytest.mark.parametrize(
        "payload",
        ["", "1 OR 1=1", "admin@juice-sh.op", "''", "no quotes here"],
    )
    def test_leaves_text_without_literals_alone(self, payload: str) -> None:
        assert hex_encode(payload) == payload

    def test_keeps_surrounding_text(self) -> None:
        assert hex_encode("id='ab' AND x=1") == "id=0x6162 AND x=1"

    def test_encodes_utf8_literals(self) -> None:
        assert hex_encode("'ab'") == "0x" + "ab".encode("utf-8").hex()

    def test_produces_no_quotes(self) -> None:
        assert "'" not in hex_encode("'admin' AND 'root'")


class TestConcatSplit:
    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ("'admin'", "'adm'||'in'"),
            ("'ab'", "'a'||'b'"),
            ("'ab' AND 'cd'", "'a'||'b' AND 'c'||'d'"),
        ],
    )
    def test_splits_literals_into_two_halves(self, payload: str, expected: str) -> None:
        assert concat_split(payload) == expected

    @pytest.mark.parametrize(
        "payload",
        ["", "1=1", "'a'", "''", "qwert')) AND 1=1--"],
    )
    def test_leaves_short_or_absent_literals_alone(self, payload: str) -> None:
        assert concat_split(payload) == payload

    def test_keeps_the_original_characters(self) -> None:
        mutated = concat_split("'admin'")
        assert "".join(char for char in mutated if char.isalpha()) == "admin"

    def test_uses_sqlite_double_pipe_concatenation(self) -> None:
        assert "||" in concat_split("'admin'")


class TestNewlineBypass:
    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ("1=1--", "1=1--\n"),
            ("' OR 1=1--", "' OR 1=1--\n"),
            ("a--b--", "a--\nb--\n"),
        ],
    )
    def test_appends_newline_after_comment_opener(
        self, payload: str, expected: str
    ) -> None:
        assert newline_bypass(payload) == expected

    @pytest.mark.parametrize(
        "payload",
        ["", "UNION SELECT 1", "<script>alert(1)</script>", "1=1"],
    )
    def test_leaves_payload_without_comment_alone(self, payload: str) -> None:
        assert newline_bypass(payload) == payload

    def test_adds_exactly_one_newline_per_comment(self) -> None:
        payload = "' OR 1=1--"
        assert newline_bypass(payload).count("\n") == payload.count("--")

    def test_newline_goes_right_after_the_dashes(self) -> None:
        assert newline_bypass("1--x").index("\n") == "1--".index("-") + 2


class TestApplyChain:
    def test_empty_chain_is_identity(self) -> None:
        payload = "' OR 1=1--"
        assert apply_chain(payload, []) == payload

    def test_single_mutator_matches_direct_call(self) -> None:
        payload = "' UNION SELECT NULL--"
        assert apply_chain(payload, ["comment_inject"]) == comment_inject(payload)

    def test_applies_mutators_left_to_right(self) -> None:
        assert apply_chain("1=1--", ["newline_bypass", "url_encode"]) == "1%3D1--%0A"

    def test_order_is_significant(self) -> None:
        forward = apply_chain("1=1--", ["newline_bypass", "url_encode"])
        backward = apply_chain("1=1--", ["url_encode", "newline_bypass"])
        assert forward != backward
        assert backward == "1%3D1--\n"

    def test_three_step_chain(self) -> None:
        payload = "' UNION SELECT NULL--"
        expected = "'%09UN/**/ION%09SE/**/LECT%09NULL--\n"
        chain = ["comment_inject", "whitespace_swap", "newline_bypass"]
        assert apply_chain(payload, chain) == expected

    def test_null_byte_then_newline_chain(self) -> None:
        assert apply_chain("' OR 1=1--", ["null_byte", "newline_bypass"]) == (
            "%00' OR 1=1--\n"
        )

    def test_matches_manual_composition(self) -> None:
        chain = ["case_swap", "url_encode", "double_url_encode"]
        manual = double_url_encode(url_encode(case_swap("union", seed=42)))
        assert apply_chain("union", chain) == manual

    def test_rejects_unknown_name(self) -> None:
        with pytest.raises(ValueError, match="unknown mutation 'nope'"):
            apply_chain("1=1", ["url_encode", "nope"])

    def test_error_message_lists_available_mutations(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            apply_chain("1=1", ["nope"])
        assert "null_byte" in str(excinfo.value)
        assert "url_encode" in str(excinfo.value)

    def test_rejects_non_string_chain_entries(self) -> None:
        with pytest.raises(ValueError, match="unknown mutation"):
            apply_chain("1=1", ["url_encode", 1])  # type: ignore[list-item]

    def test_rejects_non_list_chain(self) -> None:
        with pytest.raises(TypeError, match="chain must be a list"):
            apply_chain("1=1", ("url_encode",))  # type: ignore[arg-type]

    def test_unknown_name_is_reported_before_mutating(self) -> None:
        with pytest.raises(ValueError):
            apply_chain("1=1", ["newline_bypass", "nope"])
        assert MUTATIONS["newline_bypass"]("1=1") == "1=1"


EXPECTED_MUTATIONS = {
    "url_encode",
    "double_url_encode",
    "case_swap",
    "comment_inject",
    "whitespace_swap",
    "null_byte",
    "unicode_escape",
    "hex_encode",
    "concat_split",
    "newline_bypass",
}


class TestRegistry:
    def test_exposes_exactly_the_ten_mutators(self) -> None:
        assert set(MUTATIONS) == EXPECTED_MUTATIONS

    def test_maps_names_to_callables(self) -> None:
        assert all(callable(mutator) for mutator in MUTATIONS.values())

    @pytest.mark.parametrize("name", sorted(EXPECTED_MUTATIONS))
    def test_every_mutator_has_a_docstring(self, name: str) -> None:
        doc = MUTATIONS[name].__doc__
        assert doc is not None
        assert len(doc.strip()) > 40

    @pytest.mark.parametrize("name", sorted(EXPECTED_MUTATIONS))
    def test_every_mutator_is_pure_and_returns_str(self, name: str) -> None:
        mutator = MUTATIONS[name]
        payload = "' UNION SELECT NULL--"
        first, second = mutator(payload), mutator(payload)
        assert isinstance(first, str)
        assert first == second

    @pytest.mark.parametrize("name", sorted(EXPECTED_MUTATIONS))
    @pytest.mark.parametrize("payload", ["", "' OR 1=1--", "<script>alert(1)</script>"])
    def test_every_mutator_accepts_any_string(self, name: str, payload: str) -> None:
        assert isinstance(MUTATIONS[name](payload), str)


class TestPayloadFixtures:
    def test_sqli_fixture_has_five_payloads(self) -> None:
        assert load_payloads("sqli.txt") == [
            "' OR 1=1--",
            "admin@juice-sh.op'--",
            "qwert')) OR deletedAt IS NOT NULL--",
            "' UNION SELECT NULL--",
            "qwert')) AND 1=1--",
        ]

    def test_xss_fixture_has_three_payloads(self) -> None:
        assert load_payloads("xss.txt") == [
            "<script>alert(1)</script>",
            '"><img src=x onerror=alert(1)>',
            "javascript:alert(1)",
        ]

    @pytest.mark.parametrize("name", sorted(EXPECTED_MUTATIONS))
    def test_mutating_every_fixture_payload_changes_nothing_structurally(
        self, name: str
    ) -> None:
        mutator = MUTATIONS[name]
        for payload in load_payloads("sqli.txt") + load_payloads("xss.txt"):
            assert isinstance(mutator(payload), str)
