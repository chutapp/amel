"""Re-run paper_statistics with majority-vote-adjudicated item labels.

8 items currently labelled "ambiguous" by the author were re-labelled
"clear_positive" by ≥3 of 4 external annotators. We adopt the majority
label for those, keep the author label everywhere else (including the
7 no-consensus ties), and re-compute the per-category statistics.

This produces a v2 paper_statistics file (`paper_statistics_v2.json`)
that we can compare to v1 (the original author-coded labels) without
overwriting anything.

Run:  python -m analysis.paper_statistics_adjudicated
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AGG_PATH = REPO / "data" / "annotators" / "aggregate_labels.json"
MAP_PATH = REPO / "data" / "annotator_id_mapping.json"
ITEM_FILE = REPO / "data" / "all_results.jsonl"


def build_relabel_map() -> dict[str, str]:
    """Return original_item_id -> adjudicated_category for items where the
    author label disagrees with a non-tie majority vote.

    For tie items and items where author already agrees with majority, we
    return nothing (keep author label).
    """
    agg = json.loads(AGG_PATH.read_text())
    id_map = json.loads(MAP_PATH.read_text())  # anon_id -> original_test_item_id

    relabel: dict[str, str] = {}
    for anon_id, info in agg["per_item"].items():
        author = info.get("author_label")
        majority = info.get("label")
        if info.get("tie") or majority is None or author is None:
            continue
        if majority == author:
            continue
        original_id = id_map.get(anon_id)
        if not original_id:
            continue
        relabel[original_id] = majority
    return relabel


def rewrite_dataset(relabel: dict[str, str]) -> Path:
    """Make an adjudicated copy of all_results.jsonl with updated
    `test_item_category`."""
    out = REPO / "data" / "all_results.adjudicated.jsonl"
    n_rewritten = 0
    with ITEM_FILE.open() as fin, out.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            iid = row.get("test_item_id")
            if iid in relabel:
                row["test_item_category"] = relabel[iid]
                n_rewritten += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Rewrote {n_rewritten:,} rows across {len(relabel)} relabelled items -> {out}")
    return out


def main() -> None:
    relabel = build_relabel_map()
    print(f"Adjudicated relabelling map ({len(relabel)} items):")
    for k, v in sorted(relabel.items()):
        print(f"  {k}: author=ambiguous -> majority={v}")

    # Rewrite dataset
    adj_path = rewrite_dataset(relabel)

    # Re-run paper_statistics as a subprocess with the adjudicated file
    # set via env var. paper_statistics.py writes to a fixed path; we
    # rename the output afterwards.
    import os
    import shutil
    import subprocess

    primary_out = REPO / "results" / "paper_statistics.json"
    v1_backup = REPO / "results" / "paper_statistics_v1.json"
    v2_out = REPO / "results" / "paper_statistics_v2.json"

    # Save v1 if not already snapshotted.
    if primary_out.exists() and not v1_backup.exists():
        shutil.copy(primary_out, v1_backup)
        print(f"v1 snapshot saved: {v1_backup}")

    env = {**os.environ, "AMEL_DATA_FILE": str(adj_path)}
    print(f"\nRunning paper_statistics on adjudicated dataset...")
    r = subprocess.run(
        ["python3", "-m", "analysis.paper_statistics"],
        env=env, cwd=str(REPO),
        capture_output=True, text=True
    )
    print(r.stdout.splitlines()[-3] if r.stdout else "(no stdout)")
    if r.returncode != 0:
        print("STDERR:", r.stderr[-500:])
        return

    # Move the new output aside.
    shutil.move(primary_out, v2_out)
    # Restore the v1 file as the primary so other scripts don't break.
    shutil.copy(v1_backup, primary_out)
    print(f"Wrote adjudicated stats: {v2_out}")
    print(f"v1 stats (author labels) preserved at: {v1_backup}")


if __name__ == "__main__":
    main()
