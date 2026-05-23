"""Dedup qwen3:30b rows in data/all_results.jsonl.

A concurrent-resume bug after a disk-full crash on 2026-03-16 caused two
Python processes to append duplicate-condition rows for qwen3:30b
(2,186 extras). We keep the FIRST occurrence per
(domain, polarity, context_length, test_item_id, repetition) key, in file
order, for that model only. All other models are passed through unchanged.

Run:  python scripts/dedupe_qwen30b.py
"""
from __future__ import annotations

import json
from pathlib import Path

SRC = Path("data/all_results.jsonl")
DST = Path("data/all_results.jsonl")  # in-place; backup is .pre_dedup_backup.jsonl

KEY_FIELDS = ("domain", "polarity", "context_length", "test_item_id", "repetition")


def main() -> None:
    seen: set[tuple] = set()
    kept = 0
    dropped = 0
    out_lines: list[str] = []

    with SRC.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("model") == "qwen3:30b":
                key = tuple(r[k] for k in KEY_FIELDS)
                if key in seen:
                    dropped += 1
                    continue
                seen.add(key)
            kept += 1
            out_lines.append(line)

    DST.write_text("".join(out_lines))
    print(f"Kept {kept} rows, dropped {dropped} qwen3:30b duplicates.")


if __name__ == "__main__":
    main()
