"""Command line interface: mutate lab payloads, replay them, classify replies.

Usage flow, in this order:

1. :func:`~payload_mangler.scope.check_scope` runs **before** anything touches
   the network -- an out-of-scope target ends the run with exit code ``2``;
2. payloads are read from ``--payload-file`` (one per line, ``#`` comments and
   blank lines skipped) or taken from ``--payload``;
3. every payload is mutated by each selected mutator, optionally plus every
   ordered pair of mutators (``--chain``); identical mutated strings are sent
   only once, keeping the first occurrence;
4. a baseline request (empty payload) is sent, then the batch runs through the
   thread pool from :mod:`payload_mangler.engine`;
5. each reply becomes a :class:`~payload_mangler.report.Finding` carrying the
   filter verdict, the oracle result and the ``reached_backend`` flag;
6. the table is printed, ``--report`` dumps JSON, and a one-line tally closes
   the run.

The tool never leaves the local lab: there is no flag to widen the scope.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import requests

from .detector import Verdict, analyze, baseline, url_decode
from .engine import MAX_THREADS, PAYLOAD_MARKER, Response, run_batch
from .mutations import MUTATIONS, apply_chain
from .oracle import Oracle, make_oracle
from .report import Finding, print_table, save_json
from .scope import ScopeError, check_scope

__all__ = [
    "Candidate",
    "build_candidates",
    "build_parser",
    "collect_findings",
    "load_payloads",
    "main",
    "pairwise_chains",
    "parse_cookies",
    "parse_headers",
    "resolve_mutations",
    "run",
    "summarize",
]

#: Everything went fine.
EXIT_OK = 0
#: Bad usage, out-of-scope target or unreadable input.
EXIT_USAGE = 2

DESCRIPTION = (
    "Mutate security payloads and replay them against a local lab stand, "
    "classifying every reply. Loopback targets only."
)

EPILOG = """\
example:
  python3 -m payload_mangler \\
    --url 'http://127.0.0.1:3000/rest/products/search?q={{PAYLOAD}}' \\
    --payload-file payloads/sqli.txt --mutations all --chain \\
    --oracle-regex '"data":\\[\\{' --report out.json

verdicts: BLOCKED (filter said no), REFLECTED (payload echoed),
PASSTHROUGH (clean reply), ERROR (no HTTP response).
BACKEND=yes means a database error surfaced: the filter was bypassed.
"""


def _threads(value: str) -> int:
    """argparse type: worker count, clamped by validation to ``1..16``."""
    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}") from None
    if not 1 <= count <= MAX_THREADS:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_THREADS}")
    return count


def _delay(value: str) -> float:
    """argparse type: non-negative per-thread delay in seconds."""
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid float value: {value!r}") from None
    if seconds < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return seconds


def _add_target_args(parser: argparse.ArgumentParser) -> None:
    """Register URL / method / body / header / cookie options."""
    parser.add_argument(
        "--url", required=True, metavar="URL",
        help=f"target URL containing {PAYLOAD_MARKER}",
    )
    parser.add_argument(
        "--method", default="GET", type=str.upper, choices=("GET", "POST"),
        help="HTTP method (default: GET)",
    )
    parser.add_argument(
        "--data", metavar="BODY",
        help=f"POST body template, may contain {PAYLOAD_MARKER}",
    )
    parser.add_argument(
        "-H", "--header", action="append", default=[], metavar="'Key: Value'",
        help="extra request header, repeatable",
    )
    parser.add_argument("--cookie", metavar="'k=v; k2=v2'", help="cookies to send")


def _add_input_args(parser: argparse.ArgumentParser) -> None:
    """Register payload source, mutator selection and oracle options."""
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--payload-file", metavar="PATH",
        help="file with one payload per line (blank lines and # comments skipped)",
    )
    source.add_argument("--payload", metavar="PAYLOAD", help="a single payload")
    parser.add_argument(
        "--mutations", default="all", metavar="LIST",
        help="comma-separated mutator names, or 'all' (default: all)",
    )
    parser.add_argument(
        "--chain", action="store_true",
        help="also try every ordered pair of mutators",
    )
    parser.add_argument(
        "--oracle-regex", metavar="REGEX",
        help="success marker searched in the body (default: always true)",
    )


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    """Register transport tuning and reporting options."""
    parser.add_argument(
        "--proxy", metavar="URL", help="proxy, e.g. http://127.0.0.1:8080"
    )
    parser.add_argument(
        "--delay", type=_delay, default=0.1, metavar="SECONDS",
        help="delay between requests per thread (default: 0.1)",
    )
    parser.add_argument(
        "--threads", type=_threads, default=4, metavar="N",
        help=f"worker threads, 1..{MAX_THREADS} (default: 4)",
    )
    parser.add_argument("--report", metavar="PATH", help="write findings as JSON to PATH")
    parser.add_argument(
        "--no-color", action="store_true", help="disable ANSI colours in the table"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser backing ``--help``."""
    parser = argparse.ArgumentParser(
        prog="python3 -m payload_mangler",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_target_args(parser)
    _add_input_args(parser)
    _add_output_args(parser)
    return parser


def parse_headers(values: Sequence[str]) -> dict[str, str]:
    """Parse repeated ``-H 'Key: Value'`` entries into a header dict.

    Raises:
        ValueError: if an entry has no colon or an empty name.
    """
    headers: dict[str, str] = {}
    for raw in values:
        name, separator, value = raw.partition(":")
        if not separator or not name.strip():
            raise ValueError(f"invalid header {raw!r}; expected 'Key: Value'")
        headers[name.strip()] = value.strip()
    return headers


def parse_cookies(value: str | None) -> dict[str, str]:
    """Parse a ``k=v; k2=v2`` cookie string into a dict.

    Raises:
        ValueError: if a pair has no ``=`` or an empty name.
    """
    if not value:
        return {}
    cookies: dict[str, str] = {}
    for chunk in value.split(";"):
        name, separator, val = chunk.strip().partition("=")
        if not chunk.strip():
            continue
        if not separator or not name.strip():
            raise ValueError(f"invalid cookie {chunk.strip()!r}; expected 'name=value'")
        cookies[name.strip()] = val.strip()
    return cookies


def load_payloads(path: str | Path) -> list[str]:
    """Read payloads from ``path``: one per line, ``#`` comments and blanks out.

    Lines are kept verbatim (no stripping) so leading/trailing whitespace, which
    may be part of the payload, survives.

    Raises:
        OSError: if the file cannot be read.
    """
    text = Path(path).read_text(encoding="utf-8")
    return [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def resolve_mutations(spec: str) -> list[str]:
    """Turn ``--mutations`` (``all`` or ``a,b,c``) into validated mutator names.

    Raises:
        ValueError: if the list is empty or names an unknown mutator.
    """
    if spec.strip().lower() == "all":
        return list(MUTATIONS)
    names = list(dict.fromkeys(part.strip() for part in spec.split(",") if part.strip()))
    if not names:
        raise ValueError("--mutations must list at least one mutator")
    unknown = [name for name in names if name not in MUTATIONS]
    if unknown:
        raise ValueError(
            f"unknown mutation(s): {', '.join(unknown)}; "
            f"available: {', '.join(sorted(MUTATIONS))}"
        )
    return names


def pairwise_chains(names: Sequence[str]) -> list[tuple[str, str]]:
    """All ordered pairs of distinct mutators -- order matters when chaining."""
    return [(first, second) for first in names for second in names if first != second]


@dataclass(frozen=True, slots=True)
class Candidate:
    """One request to send: an original payload, its mutant and the chain used."""

    original: str
    mutated: str
    chain: tuple[str, ...] = ()


def build_candidates(
    payloads: Sequence[str], names: Sequence[str], chained: bool
) -> list[Candidate]:
    """Expand payloads into single-mutator (optionally paired) candidates.

    Identical mutated strings are dropped across the whole run while keeping the
    first occurrence, so a mutator that leaves a payload untouched does not cost
    a duplicate request.
    """
    chains: list[tuple[str, ...]] = [(name,) for name in names]
    if chained:
        chains.extend(pairwise_chains(names))
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for payload in payloads:
        for chain in chains:
            mutated = apply_chain(payload, list(chain))
            if mutated in seen:
                continue
            seen.add(mutated)
            candidates.append(Candidate(payload, mutated, chain))
    return candidates


def collect_findings(
    candidates: Sequence[Candidate],
    responses: Sequence[Response],
    baseline_response: Response,
    oracle: Oracle,
) -> list[Finding]:
    """Pair every candidate with its reply and build the report rows.

    Each row carries the filter verdict (:func:`~payload_mangler.detector.analyze`),
    the oracle outcome and the ``reached_backend`` flag, so a ``PASSTHROUGH``
    row with ``BACKEND=yes`` stands out as "the filter was bypassed".
    """
    findings: list[Finding] = []
    for candidate, response in zip(candidates, responses):
        analysis = analyze(
            response,
            baseline_response,
            candidate.mutated,
            url_decode(candidate.mutated),
        )
        findings.append(
            Finding(
                original_payload=candidate.original,
                mutated_payload=candidate.mutated,
                chain=list(candidate.chain),
                verdict=analysis.verdict,
                status_code=response.status_code,
                size=response.size,
                elapsed_ms=response.elapsed_ms,
                oracle_true=bool(oracle(response)),
                backend=analysis.reached_backend,
            )
        )
    return findings


def summarize(findings: Sequence[Finding]) -> str:
    """One-line tally, e.g. ``42 mutants: 12 blocked, 3 passthrough+oracle, ...``."""
    blocked = sum(1 for item in findings if item.verdict is Verdict.BLOCKED)
    passthrough_hits = sum(
        1
        for item in findings
        if item.verdict is Verdict.PASSTHROUGH and item.oracle_true
    )
    backend = sum(1 for item in findings if item.backend)
    errors = sum(1 for item in findings if item.verdict is Verdict.ERROR)
    return (
        f"{len(findings)} mutants: {blocked} blocked, "
        f"{passthrough_hits} passthrough+oracle, "
        f"{backend} reached_backend, {errors} error"
    )


def _resolve_payloads(args: argparse.Namespace) -> list[str]:
    """Payload list from ``--payload`` or ``--payload-file``.

    Raises:
        ValueError: on an empty single payload or a file without payloads.
        OSError: when the payload file cannot be read.
    """
    if args.payload is not None:
        if not args.payload.strip():
            raise ValueError("--payload must not be empty")
        return [args.payload]
    payloads = load_payloads(args.payload_file)
    if not payloads:
        raise ValueError(f"no payloads found in {args.payload_file!r}")
    return payloads


def _validate_target(args: argparse.Namespace) -> str | None:
    """Return a usage error message when the payload marker is missing."""
    if PAYLOAD_MARKER in args.url:
        return None
    if args.data and PAYLOAD_MARKER in args.data:
        return None
    return f"--url or --data must contain the {PAYLOAD_MARKER} marker"


def _execute(
    args: argparse.Namespace, candidates: Sequence[Candidate], oracle: Oracle
) -> list[Finding]:
    """Send the baseline plus the whole batch and convert replies to findings."""
    headers = parse_headers(args.header)
    cookies = parse_cookies(args.cookie)
    session = requests.Session()
    base = baseline(
        session, args.url, args.method, headers, cookies, args.data
    )
    responses = run_batch(
        args.url,
        args.method,
        [candidate.mutated for candidate in candidates],
        headers=headers,
        cookies=cookies,
        delay=args.delay,
        threads=args.threads,
        proxy=args.proxy,
        body_template=args.data,
    )
    return collect_findings(candidates, responses, base, oracle)


def _prepare(args: argparse.Namespace) -> tuple[list[Candidate], Oracle]:
    """Validate the target, build the oracle and expand payloads into candidates.

    Raises:
        ValueError: bad marker, mutator list, payload input or oracle regex.
        OSError: unreadable payload file.
    """
    problem = _validate_target(args)
    if problem:
        raise ValueError(problem)
    oracle = make_oracle(args.oracle_regex)
    names = resolve_mutations(args.mutations)
    payloads = _resolve_payloads(args)
    return build_candidates(payloads, names, args.chain), oracle


def run(args: argparse.Namespace) -> int:
    """Execute the scan described by ``args`` and return the process exit code."""
    try:
        check_scope(args.url)
    except ScopeError as exc:
        print(f"scope error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        candidates, oracle = _prepare(args)
    except (ValueError, OSError, re.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    findings = _execute(args, candidates, oracle)
    print()
    print_table(findings, use_color=not args.no_color)
    if args.report:
        print(f"report: {save_json(findings, args.report)}")
    print(summarize(findings))
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: parse ``argv`` (default ``sys.argv``) and run."""
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
