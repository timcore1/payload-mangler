"""payload-mangler -- deterministic mutation chains for security payloads.

Lab-only helper: mutate a known-bad payload in reproducible ways to check how
a local WAF / filter responds. Nothing here performs network I/O; the CLI and
the HTTP layer are separate (later) modules.

>>> from payload_mangler import apply_chain
>>> apply_chain("' OR 1=1--", ["null_byte", "newline_bypass"])
"%00' OR 1=1--\\n"
"""

from __future__ import annotations

from .detector import (
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
from .engine import (
    MAX_THREADS,
    PAYLOAD_MARKER,
    Request,
    Response,
    build_request,
    run_batch,
    send,
)
from .oracle import Oracle, make_oracle
from .report import Finding, color_enabled, print_table, save_json
from .scope import ALLOWED, ScopeError, check_scope
from .mutations import (
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

__version__ = "0.4.0"

__all__ = [
    "ALLOWED",
    "BACKEND_SIGNALS",
    "BLOCK_SIGNATURES",
    "BLOCK_STATUS_CODES",
    "MAX_THREADS",
    "MUTATIONS",
    "PAYLOAD_MARKER",
    "Analysis",
    "Finding",
    "Oracle",
    "Request",
    "Response",
    "ScopeError",
    "Verdict",
    "__version__",
    "analyze",
    "apply_chain",
    "backend_signals_in",
    "baseline",
    "build_request",
    "case_swap",
    "check_scope",
    "classify",
    "color_enabled",
    "comment_inject",
    "concat_split",
    "double_url_encode",
    "hex_encode",
    "make_oracle",
    "newline_bypass",
    "null_byte",
    "print_table",
    "reached_backend",
    "reflection_forms",
    "run_batch",
    "save_json",
    "send",
    "unicode_escape",
    "url_decode",
    "url_encode",
    "whitespace_swap",
]
