"""Dedup-sensitivity check for Qwen3 30B (B5).

The Qwen3 30B run had 2,186 duplicate-condition rows from a concurrent
resume. The published dataset keeps the FIRST occurrence per condition.
This script re-runs the Qwen3:30b headline statistics under the LAST-
occurrence choice, and under random selection, to confirm that the dedup
strategy does not change the direction or significance of the per-model
result.

Outputs:
    results/qwen30b_dedup_sensitivity.json
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# Allow running from repo root via -m
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import t as student_t

from src.parser_v2 import parse_yes_no

BACKUP = Path("data/all_results.pre_dedup_backup.jsonl")
KEY_FIELDS = ("domain", "polarity", "context_length", "test_item_id", "repetition")


def collect_qwen30b_rows() -> list[dict]:
    rows = []
    with BACKUP.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("model") == "qwen3:30b":
                # Re-parse with v2 parser, since the backup carries the OLD label
                r["parsed_response"] = parse_yes_no(r.get("raw_response", ""))
                rows.append(r)
    return rows


def dedupe(rows: list[dict], strategy: str) -> list[dict]:
    """strategy: 'first', 'last', or 'random'."""
    if strategy == "first":
        seen: set[tuple] = set()
        out = []
        for r in rows:
            k = tuple(r[f] for f in KEY_FIELDS)
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
        return out
    if strategy == "last":
        idx: dict[tuple, dict] = {}
        for r in rows:
            k = tuple(r[f] for f in KEY_FIELDS)
            idx[k] = r
        return list(idx.values())
    if strategy == "random":
        rng = random.Random(20260520)
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows:
            groups[tuple(r[f] for f in KEY_FIELDS)].append(r)
        return [rng.choice(v) for v in groups.values()]
    raise ValueError(strategy)


def compute_bias_scores(rows: list[dict]) -> list[float]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["parsed_response"] is None:
            continue
        base_key = f"{r['domain']}|{r['test_item_id']}"
        if r["polarity"] == "baseline":
            groups[f"{base_key}|baseline"].append(r)
        else:
            key = f"{base_key}|{r['polarity']}|{r['context_length']}"
            groups[key].append(r)

    scores: list[float] = []
    for key, group in groups.items():
        parts = key.split("|")
        if parts[-1] == "baseline":
            continue
        domain, item_id, polarity, ctx_len = parts
        baseline_key = f"{domain}|{item_id}|baseline"
        baseline = groups.get(baseline_key, [])
        if not baseline:
            continue
        target = "no" if polarity in ("no_saturated", "neutral") else "yes"
        bl_rate = sum(1 for r in baseline if r["parsed_response"] == target) / len(baseline)
        tx_rate = sum(1 for r in group if r["parsed_response"] == target) / len(group)
        scores.append(tx_rate - bl_rate)
    return scores


def summarize(scores: list[float]) -> dict:
    a = np.array(scores)
    n = len(a)
    if n < 2:
        return {"n": n, "mean": float(a.mean()) if n else 0.0, "cohens_d": 0.0, "p": 1.0}
    mean = float(a.mean())
    std = float(a.std(ddof=1))
    t_stat = mean / (std / np.sqrt(n)) if std > 0 else 0.0
    p = 2 * (1 - student_t.cdf(abs(t_stat), df=n - 1))
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "cohens_d": float(mean / std) if std > 0 else 0.0,
        "t": float(t_stat),
        "p": float(p),
    }


def main() -> None:
    rows = collect_qwen30b_rows()
    print(f"Loaded {len(rows)} raw Qwen3 30B rows from backup")

    out: dict[str, dict] = {}
    for strat in ("first", "last", "random"):
        deduped = dedupe(rows, strat)
        scores = compute_bias_scores(deduped)
        summary = summarize(scores)
        summary["n_rows_after_dedup"] = len(deduped)
        out[strat] = summary
        print(f"  {strat:6s}  n_rows={len(deduped):>5d}  n_bs={summary['n']:>4d}  mean={summary['mean']:+.4f}  d={summary['cohens_d']:+.3f}  p={summary['p']:.3g}")

    Path("results/qwen30b_dedup_sensitivity.json").write_text(json.dumps(out, indent=2))
    print("\nSaved: results/qwen30b_dedup_sensitivity.json")


if __name__ == "__main__":
    main()
