"""Re-parse the four binary side-experiment datasets with parser_v2.

Each row already has `raw_response`. We overwrite `parsed_response` in place.
Logprobs experiment skipped: it uses raw first-token logprobs, not the
binary parser.

Run from repo root:  python scripts/reparse_side_experiments_v2.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.parser_v2 import parse_yes_no  # noqa: E402

TARGETS = [
    Path("data/flipped/results.jsonl"),
    Path("data/positional/results.jsonl"),
    Path("data/mitigation/results.jsonl"),
    Path("data/temperature/results.jsonl"),
]


def reparse(path: Path) -> None:
    transitions: Counter[tuple[str | None, str | None]] = Counter()
    out_lines: list[str] = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            old = r.get("parsed_response")
            new = parse_yes_no(r.get("raw_response", ""))
            transitions[(old, new)] += 1
            r["parsed_response"] = new
            out_lines.append(json.dumps(r, ensure_ascii=False) + "\n")
    path.write_text("".join(out_lines))
    n_total = sum(transitions.values())
    n_changed = sum(c for (o, n), c in transitions.items() if o != n)
    print(f"  {path}: {n_total} rows, {n_changed} parse changes")
    for (o, n), c in sorted(transitions.items(), key=lambda x: -x[1]):
        if o != n:
            print(f"    {str(o):>10s} -> {str(n):<10s}  {c}")


def main() -> None:
    print("Re-parsing 4 side-experiment datasets with parser_v2...")
    for p in TARGETS:
        if p.exists():
            reparse(p)
        else:
            print(f"  SKIP (not found): {p}")


if __name__ == "__main__":
    main()
