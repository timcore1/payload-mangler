"""Deterministic payload mutators for local lab WAF/filter bypass testing.

Design rules of this module:

* every public mutator is a *pure* function ``str -> str`` -- no I/O, no
  global state, no ``eval``/``exec``;
* the input string is never modified in place (strings are immutable, and no
  mutator keeps a reference to it), so chains stay reproducible;
* randomness exists only in :func:`case_swap` and is driven by an explicit
  ``seed`` argument;
* names are stable strings, so a chain can be stored and replayed later via
  :func:`apply_chain`.

Intended use: attacking your own lab / CTF targets only.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from re import Match
from urllib.parse import quote

__all__ = [
    "MUTATIONS",
    "apply_chain",
    "case_swap",
    "comment_inject",
    "concat_split",
    "double_url_encode",
    "hex_encode",
    "newline_bypass",
    "null_byte",
    "unicode_escape",
    "url_encode",
    "whitespace_swap",
]

# Characters that RFC 3986 leaves unencoded: A-Z a-z 0-9 - . _ ~ (always safe
# for ``urllib.parse.quote``). An empty ``safe`` set therefore means "encode
# every special character, keep alphanumerics and - . _ ~ as is".
_SAFE_CHARS = ""


# --------------------------------------------------------------------------- #
# 1-2. URL encoding
# --------------------------------------------------------------------------- #
def url_encode(s: str) -> str:
    """Percent-encode special characters of ``s`` exactly once.

    Alphanumerics and ``-._~`` are kept, everything else (spaces, quotes,
    angle brackets, slashes...) becomes ``%XX``.

    >>> url_encode("' OR 1=1--")
    '%27%20OR%201%3D1--'
    """
    return quote(s, safe=_SAFE_CHARS)


def double_url_encode(s: str) -> str:
    """Percent-encode ``s`` twice, so the ``%`` itself becomes ``%25``.

    Useful against filters that decode the input once before matching, and
    against filters that only decode once after matching.
    """
    return quote(quote(s, safe=_SAFE_CHARS), safe=_SAFE_CHARS)


# --------------------------------------------------------------------------- #
# 3. Case randomisation
# --------------------------------------------------------------------------- #
def case_swap(s: str, seed: int = 42) -> str:
    """Randomly swap the case of ASCII letters, deterministically per ``seed``.

    Same ``seed`` always yields the same output (chains stay reproducible).
    At least one letter is flipped when the input contains letters, so the
    result never silently equals the input. Non-letter characters are copied
    verbatim and the length never changes.

    >>> case_swap("UNION", seed=42)
    'UnioN'
    """
    rng = random.Random(seed)
    chars = list(s)
    letters = [i for i, ch in enumerate(chars) if ch.isascii() and ch.isalpha()]
    if not letters:
        return s
    flipped = False
    for index in letters:
        if rng.random() < 0.5:
            chars[index] = chars[index].swapcase()
            flipped = True
    if not flipped:
        index = rng.choice(letters)
        chars[index] = chars[index].swapcase()
    return "".join(chars)


# --------------------------------------------------------------------------- #
# 4. SQL comment injection
# --------------------------------------------------------------------------- #
# Longest keywords first so the alternation cannot shadow a longer match.
_SQL_KEYWORDS: tuple[str, ...] = ("SELECT", "UNION", "WHERE", "FROM", "AND", "OR")
_KEYWORD_RE = re.compile(
    r"\b(?:%s)\b" % "|".join(_SQL_KEYWORDS), flags=re.IGNORECASE
)


def _split_keyword(match: Match[str]) -> str:
    """Split a matched keyword with ``/**/`` after its second letter."""
    word = match.group(0)
    pivot = 2 if len(word) > 2 else 1
    return f"{word[:pivot]}/**/{word[pivot:]}"


def comment_inject(s: str) -> str:
    """Break SQL keywords apart with an inline comment (``UN/**/ION``).

    Handled keywords: ``SELECT``, ``UNION``, ``WHERE``, ``FROM``, ``AND``,
    ``OR``. They are matched case-insensitively as whole words and split after
    the second letter (``SE/**/LECT``, ``UN/**/ION``) or after the first one
    for two-letter keywords (``O/**/R``). Everything else is untouched, and a
    keyword hidden inside a longer word (``OR`` in ``ORDER``) is not split.

    >>> comment_inject("' UNION SELECT NULL--")
    "' UN/**/ION SE/**/LECT NULL--"
    """
    return _KEYWORD_RE.sub(_split_keyword, s)


# --------------------------------------------------------------------------- #
# 5. Whitespace substitution
# --------------------------------------------------------------------------- #
_WHITESPACE_VARIANTS: dict[str, str] = {
    "tab": "%09",
    "lf": "%0a",
    "ff": "%0c",
    "comment": "/**/",
}


def whitespace_swap(s: str, variant: str = "tab") -> str:
    """Replace plain spaces with a WAF-friendly separator.

    Supported ``variant`` values: ``tab`` -> ``%09``, ``lf`` -> ``%0a``,
    ``ff`` -> ``%0c``, ``comment`` -> ``/**/``. Existing tabs/newlines are
    left in place; only ``" "`` (U+0020) is substituted.

    Raises:
        ValueError: if ``variant`` is not one of the supported names.

    >>> whitespace_swap("1 OR 1", variant="comment")
    '1/**/OR/**/1'
    """
    try:
        replacement = _WHITESPACE_VARIANTS[variant]
    except KeyError:
        raise ValueError(
            f"unknown whitespace variant {variant!r}; "
            f"expected one of {sorted(_WHITESPACE_VARIANTS)}"
        ) from None
    return s.replace(" ", replacement)


# --------------------------------------------------------------------------- #
# 6. Null byte prefixing
# --------------------------------------------------------------------------- #
_NULL_BYTE_TARGETS = frozenset("'\"()")


def null_byte(s: str) -> str:
    """Insert a ``%00`` null byte *before* every quote and parenthesis.

    Terminators are a classic way to cut off whatever a server appends to the
    value (e.g. a trailing quote or ``) LIMIT 1``).

    >>> null_byte("' OR 1=1")
    "%00' OR 1=1"
    """
    out: list[str] = []
    for char in s:
        if char in _NULL_BYTE_TARGETS:
            out.append("%00")
        out.append(char)
    return "".join(out)


# --------------------------------------------------------------------------- #
# 7. JS unicode escapes
# --------------------------------------------------------------------------- #
def unicode_escape(s: str) -> str:
    r"""Turn every ASCII letter into a ``\uXXXX`` JavaScript escape.

    Handy for XSS contexts where filters look for literal Latin letters;
    digits, punctuation and non-ASCII characters are copied verbatim.

    >>> unicode_escape("ab")
    '\\u0061\\u0062'
    """
    return "".join(
        f"\\u{ord(char):04x}" if char.isascii() and char.isalpha() else char
        for char in s
    )


# --------------------------------------------------------------------------- #
# 8-9. Single-quoted SQL string literals
# --------------------------------------------------------------------------- #
_STRING_LITERAL_RE = re.compile(r"'([^']*)'")


def _literal_to_hex(match: Match[str]) -> str:
    """Rewrite one ``'literal'`` match as a ``0x`` blob (empty stays as is)."""
    inner = match.group(1)
    if not inner:
        return match.group(0)
    return "0x" + inner.encode("utf-8").hex()


def hex_encode(s: str) -> str:
    """Rewrite single-quoted SQL string literals into ``0x...`` hex literals.

    ``'admin'`` -> ``0x61646d696e``. Empty literals (``''``) and text outside
    of quotes are left untouched, since ``0x`` alone is not a valid literal.

    >>> hex_encode("id='ab'")
    'id=0x6162'
    """
    return _STRING_LITERAL_RE.sub(_literal_to_hex, s)


def _literal_to_concat(match: Match[str]) -> str:
    """Split one ``'literal'`` match into two ``||``-concatenated halves."""
    inner = match.group(1)
    if len(inner) < 2:
        return match.group(0)
    pivot = (len(inner) + 1) // 2
    return f"'{inner[:pivot]}'||'{inner[pivot:]}'"


def concat_split(s: str) -> str:
    """Split single-quoted SQL literals into two ``||``-concatenated halves.

    Uses SQLite string concatenation: ``'admin'`` -> ``'adm'||'in'``. Literals
    of length 0 or 1 are left alone, text outside quotes is not touched.

    >>> concat_split("name='admin'")
    "name='adm'||'in'"
    """
    return _STRING_LITERAL_RE.sub(_literal_to_concat, s)


# --------------------------------------------------------------------------- #
# 10. Line-comment newline bypass
# --------------------------------------------------------------------------- #
def newline_bypass(s: str) -> str:
    """Insert a line feed right after every ``--`` SQL comment opener.

    Some filters strip everything after ``--`` while the backend still parses
    the comment; adding a newline keeps the rest of the payload on the next
    line. ``s`` without ``--`` is returned unchanged.

    >>> newline_bypass("1=1--")
    '1=1--\\n'
    """
    return s.replace("--", "--\n")


# --------------------------------------------------------------------------- #
# Registry and chain application
# --------------------------------------------------------------------------- #
MUTATIONS: dict[str, Callable[[str], str]] = {
    "url_encode": url_encode,
    "double_url_encode": double_url_encode,
    "case_swap": case_swap,
    "comment_inject": comment_inject,
    "whitespace_swap": whitespace_swap,
    "null_byte": null_byte,
    "unicode_escape": unicode_escape,
    "hex_encode": hex_encode,
    "concat_split": concat_split,
    "newline_bypass": newline_bypass,
}


def apply_chain(s: str, chain: list[str]) -> str:
    """Apply mutators named in ``chain`` to ``s``, left to right.

    The order is significant, because mutators do not commute -- encoding a
    payload before vs. after injecting a newline gives different bytes. An
    empty chain is the identity.

    >>> apply_chain("1=1--", ["newline_bypass", "url_encode"])
    '1%3D1--%0A'
    >>> apply_chain("1=1--", ["url_encode", "newline_bypass"])
    '1%3D1--\\n'

    Raises:
        ValueError: if a name in ``chain`` is not registered in ``MUTATIONS``.
        TypeError: if ``chain`` is not a list of strings.
    """
    if not isinstance(chain, list):
        raise TypeError("chain must be a list of mutation names")
    result = s
    for name in chain:
        try:
            mutator = MUTATIONS[name]
        except (KeyError, TypeError):
            available = ", ".join(sorted(MUTATIONS))
            raise ValueError(
                f"unknown mutation {name!r}; available: {available}"
            ) from None
        result = mutator(result)
    return result
