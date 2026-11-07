"""Console table and JSON export for scan findings.

The report layer is pure presentation: it never sends requests and never
decides verdicts. It takes :class:`Finding` objects produced by the scan stage
and renders them either as an aligned coloured table or as JSON on disk.

Colour is opt-out, never opt-in: the CLI maps ``--no-color`` to
``use_color=False`` and the ``NO_COLOR`` environment variable is honoured too.
Alignment is computed on plain text, so turning colour off cannot shift
columns.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .detector import Verdict

__all__ = ["Finding", "color_enabled", "print_table", "save_json"]

RED = "\033[31m"
GREEN = "\033[32m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

#: Column order of :func:`print_table`.
HEADERS: tuple[str, ...] = (
    "№",
    "CHAIN",
    "VERDICT",
    "ORACLE",
    "BACKEND",
    "STATUS",
    "SIZE",
    "MS",
)

#: Columns rendered right-aligned (numbers); the rest are left-aligned.
_RIGHT_ALIGNED = frozenset({0, 5, 6, 7})

#: Chain label wider than this gets truncated with an ellipsis.
_CHAIN_WIDTH = 28


@dataclass(slots=True)
class Finding:
    """One payload/chain result as it should appear in a report."""

    original_payload: str
    mutated_payload: str
    chain: list[str] = field(default_factory=list)
    verdict: Verdict = Verdict.PASSTHROUGH
    status_code: int | None = None
    size: int = 0
    elapsed_ms: float = 0.0
    oracle_true: bool = False
    backend: bool = False

    @property
    def chain_label(self) -> str:
        """Chain rendered for the table: ``a+b``, truncated, ``-`` if empty."""
        if not self.chain:
            return "-"
        label = "+".join(self.chain)
        if len(label) > _CHAIN_WIDTH:
            return label[: _CHAIN_WIDTH - 3] + "..."
        return label

    def to_dict(self) -> dict[str, object]:
        """JSON-ready mapping (the verdict is flattened to its string value)."""
        return {
            "original_payload": self.original_payload,
            "mutated_payload": self.mutated_payload,
            "chain": list(self.chain),
            "verdict": str(self.verdict),
            "status_code": self.status_code,
            "size": self.size,
            "elapsed_ms": self.elapsed_ms,
            "oracle_true": self.oracle_true,
            "backend": self.backend,
        }


def save_json(findings: Sequence[Finding], path: str | Path) -> Path:
    """Write ``findings`` as indented UTF-8 JSON and return the written path.

    Parent directories are created on demand, so ``--json out/run1.json`` works
    without a prior ``mkdir``.

    >>> save_json([], "out.json")  # doctest: +SKIP
    PosixPath('out.json')
    """
    destination = Path(path)
    if destination.parent != Path(""):
        destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        [finding.to_dict() for finding in findings], indent=2, ensure_ascii=False
    )
    destination.write_text(f"{text}\n", encoding="utf-8")
    return destination


def color_enabled(explicit: bool | None = None) -> bool:
    """Decide whether ANSI colouring is emitted.

    An explicit value wins -- the CLI maps ``--no-color`` to ``False``.
    Otherwise ``NO_COLOR`` (https://no-color.org) disables colour when set.

    >>> color_enabled(False)
    False
    """
    if explicit is not None:
        return explicit
    return not os.environ.get("NO_COLOR")


def _paint(text: str, color: str | None, enabled: bool) -> str:
    """Wrap ``text`` in an ANSI colour when colouring is on and one is set."""
    if not enabled or color is None:
        return text
    return f"{color}{text}{RESET}"


def _row_color(finding: Finding) -> str | None:
    """Red for a block, green for "passed the filter and the oracle fired"."""
    if finding.verdict is Verdict.BLOCKED:
        return RED
    if finding.verdict is Verdict.PASSTHROUGH and finding.oracle_true:
        return GREEN
    return None


def _cells(index: int, finding: Finding) -> tuple[str, ...]:
    """Render one finding as plain (uncoloured) table cells."""
    status = "ERR" if finding.status_code is None else str(finding.status_code)
    return (
        str(index),
        finding.chain_label,
        finding.verdict.name,
        "yes" if finding.oracle_true else "no",
        "yes" if finding.backend else "no",
        status,
        str(finding.size),
        f"{finding.elapsed_ms:.1f}",
    )


def _widths(rows: Sequence[tuple[str, ...]]) -> list[int]:
    """Per-column widths: max of the header and every cell length."""
    widths = [len(header) for header in HEADERS]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    return widths


def _format_row(
    cells: Sequence[str], widths: Sequence[int], enabled: bool, color: str | None
) -> str:
    """Pad and optionally colour one row, keeping the columns aligned."""
    parts: list[str] = []
    for position, (cell, width) in enumerate(zip(cells, widths)):
        padded = cell.rjust(width) if position in _RIGHT_ALIGNED else cell.ljust(width)
        parts.append(_paint(padded, color, enabled))
    return " ".join(parts).rstrip()


def print_table(findings: Sequence[Finding], use_color: bool | None = None) -> None:
    """Print an aligned findings table to stdout.

    Columns: ``№ | CHAIN | VERDICT | ORACLE | BACKEND | STATUS | SIZE | MS``.
    ``BLOCKED`` rows are red; ``PASSTHROUGH`` rows whose oracle fired are
    green. Pass ``use_color=False`` (the CLI's ``--no-color``) to emit plain
    text -- padding is applied before colouring, so columns stay aligned either
    way. An empty sequence still prints the header.

    Example::

        print_table(findings, use_color=False)
    """
    enabled = color_enabled(use_color)
    rows = [_cells(index, finding) for index, finding in enumerate(findings, 1)]
    widths = _widths(rows)
    print(_format_row(HEADERS, widths, enabled, BOLD))
    print(_format_row(tuple("-" * width for width in widths), widths, enabled, DIM))
    for finding, row in zip(findings, rows):
        print(_format_row(row, widths, enabled, _row_color(finding)))
