"""Ingest the DeepSeek-V4 Flash run into the headline dataset.

Steps:
  1. Snapshot data/all_results.jsonl -> data/all_results.pre_deepseek_backup.jsonl
     (idempotent: if backup already exists, leave it).
  2. Re-parse data/deepseek-v3/results.jsonl with the v2 symmetric parser
     (same parser applied to every other model in all_results.jsonl).
  3. Append the re-parsed DeepSeek rows to data/all_results.jsonl.
  4. Print a small summary (row count by model, parse-rate for DeepSeek).

Run: python -m scripts.ingest_deepseek
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.parser_v2 import parse_yes_no  # noqa: E402

ALL = REPO / "data" / "all_results.jsonl"
BACKUP = REPO / "data" / "all_results.pre_deepseek_backup.jsonl"
DS = REPO / "data" / "deepseek-v3" / "results.jsonl"


def main() -> None:
    if not DS.exists():
        sys.exit(f"missing {DS}")

    # 1. snapshot
    if not BACKUP.exists():
        shutil.copy(ALL, BACKUP)
        print(f"snapshot: {BACKUP}")
    else:
        print(f"snapshot already present: {BACKUP}")

    # Drop any prior DeepSeek rows from ALL (in case of re-run)
    print("removing any prior deepseek-chat rows from all_results.jsonl ...")
    tmp = ALL.with_suffix(".tmp")
    kept = 0
    dropped = 0
    with ALL.open() as fin, tmp.open("w") as fout:
        for line in fin:
            r = json.loads(line)
            if r.get("model", "").startswith("deepseek"):
                dropped += 1
                continue
            fout.write(line)
            kept += 1
    tmp.replace(ALL)
    print(f"  kept {kept:,} rows; dropped {dropped:,} prior deepseek rows")

    # 2 + 3. re-parse with v2 and append
    print(f"re-parsing + appending DeepSeek rows from {DS} ...")
    parse_diff = Counter()
    appended = 0
    with DS.open() as fin, ALL.open("a") as fout:
        for line in fin:
            r = json.loads(line)
            old = r.get("parsed_response")
            new = parse_yes_no(r.get("raw_response", ""))
            r["parsed_response"] = new
            parse_diff[(old, new)] += 1
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            appended += 1
    print(f"  appended {appended:,} DeepSeek rows")

    parse_changes = {k: v for k, v in parse_diff.items() if k[0] != k[1]}
    if parse_changes:
        print(f"  parse changes (old -> new : count):")
        for (o, n), c in sorted(parse_changes.items(), key=lambda x: -x[1]):
            print(f"    {o} -> {n}: {c:,}")

    # 4. final summary
    c = Counter()
    parse_c = Counter()
    with ALL.open() as f:
        for line in f:
            r = json.loads(line)
            c[r["model"]] += 1
            if r["model"].startswith("deepseek"):
                parse_c[r["parsed_response"]] += 1
    print("\nFinal all_results.jsonl row counts by model:")
    for m, n in sorted(c.items(), key=lambda x: -x[1]):
        print(f"  {n:>6,}  {m}")
    print("\nDeepSeek parse distribution:")
    for p, n in sorted(parse_c.items(), key=lambda x: -x[1]):
        print(f"  {n:>6,}  {p}")


if __name__ == "__main__":
    main()
