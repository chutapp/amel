"""Re-parse all_results.jsonl with the symmetric v2 parser.

Replaces the `parsed_response` field in every row, leaves all other fields
untouched. Writes a new file `data/all_results_v2_parser.jsonl`. The
original file `data/all_results.jsonl` is left in place for comparison.

Also prints a diff summary: count of parses that changed yes->no, no->yes,
yes->None, no->None, None->yes, None->no.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser_v2 import parse_yes_no  # noqa: E402

SRC = Path("data/all_results.jsonl")
DST = Path("data/all_results_v2_parser.jsonl")


def main() -> None:
    transitions: Counter[tuple[str | None, str | None]] = Counter()
    per_polarity_unparse: dict[str, Counter[str]] = {}
    out_lines: list[str] = []

    with SRC.open() as f:
        for line in f:
            r = json.loads(line)
            old = r.get("parsed_response")
            new = parse_yes_no(r.get("raw_response", ""))
            transitions[(old, new)] += 1
            pol = r["polarity"]
            d = per_polarity_unparse.setdefault(pol, Counter())
            d["total"] += 1
            if new is None:
                d["unparseable"] += 1
            r["parsed_response"] = new
            out_lines.append(json.dumps(r, ensure_ascii=False) + "\n")

    DST.write_text("".join(out_lines))

    print(f"Wrote {len(out_lines)} rows to {DST}")
    print("\nTransitions (old -> new):")
    for (o, n), c in sorted(transitions.items(), key=lambda x: -x[1]):
        if o == n:
            continue
        print(f"  {str(o):>10s} -> {str(n):<10s}  {c}")

    print("\nUnchanged labels:")
    for (o, n), c in sorted(transitions.items(), key=lambda x: -x[1]):
        if o == n:
            print(f"  {str(o):>10s} -> {str(n):<10s}  {c}")

    print("\nUnparseable rate by polarity (v2 parser):")
    for pol, d in sorted(per_polarity_unparse.items()):
        rate = d["unparseable"] / max(d["total"], 1)
        print(f"  {pol:>15s}  n={d['total']:>6d}  unparse={d['unparseable']:>5d}  rate={rate:.4f}")


if __name__ == "__main__":
    main()
