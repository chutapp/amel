"""Guard for deterministic seeding across runs.

Every `run_*.py` script derives per-condition seeds from Python's built-in
`hash()` of a string key like ``"{domain}|{polarity}|{ctx_len}|{item.id}|{rep}"``.
By default Python randomizes `hash()` for str/bytes on every interpreter start
(PEP 456), so a resumed or re-launched run will produce *different* seeds for
the same condition keys. This was the root cause of the Qwen3 30B duplicate-row
incident documented in App.~\\ref{app:dedup}.

Calling :func:`require_hashseed` at process start aborts if
``PYTHONHASHSEED`` is not pinned to ``"0"``, so a future re-run cannot silently
re-introduce the same issue.
"""
from __future__ import annotations

import os
import sys


def require_hashseed(expected: str = "0") -> None:
    """Abort with a clear message if PYTHONHASHSEED is not pinned.

    Exits with status 2 (not 1) so callers can distinguish from runtime errors.
    """
    actual = os.environ.get("PYTHONHASHSEED")
    if actual != expected:
        sys.stderr.write(
            f"\nFATAL: PYTHONHASHSEED must be '{expected}' for reproducible per-condition seeds.\n"
            f"  Currently: PYTHONHASHSEED={actual!r}\n"
            f"  Re-run as: PYTHONHASHSEED={expected} python {' '.join(sys.argv)}\n"
            "  (See App. on dataset deduplication for why this matters.)\n\n"
        )
        sys.exit(2)
