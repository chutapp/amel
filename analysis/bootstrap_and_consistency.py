"""Bootstrap CIs for headline statistics + consistency-rate metric.

Two additions the council flagged as 2025-standard for LLM-judge papers:

1. **Bootstrap CIs (1000 resamples over items)** for the headline d.
   The existing 95% CIs are normal-approximation; bootstrap is more
   robust given the 10-rep quantisation of per-cell BS.

2. **Consistency rate (CR)** alongside Cohen's d. CR is the fraction
   of (item, model, polarity) cells where the model's modal response
   under treatment matches the model's modal response under baseline.
   Reported as a percentage; cross-paper comparable with the LLM-judge
   literature (Shi et al. 2024, Wang et al. 2024).

Outputs:
    results/bootstrap_cis.json
    results/consistency_rate.json

Run:  python -m analysis.bootstrap_and_consistency
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.utils import compute_bias_scores, load_results

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results"
N_BOOT = 1000
SEED = 20260525


# --------------------------- bootstrap CIs -----------------------------


def bootstrap_d(bs_values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Bootstrap CI for Cohen's d (unclustered row-level). Retained for backward
    comparison only; the published CI uses ``bootstrap_d_clustered`` below."""
    if len(bs_values) < 2:
        return {"n": int(len(bs_values)), "d": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    n = len(bs_values)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(bs_values, size=n, replace=True)
        sd = sample.std(ddof=1)
        ds[i] = sample.mean() / sd if sd > 0 else 0.0
    return {
        "n": int(n),
        "d": float(bs_values.mean() / bs_values.std(ddof=1)) if bs_values.std(ddof=1) > 0 else 0.0,
        "ci95": [round(float(np.percentile(ds, 2.5)), 4),
                 round(float(np.percentile(ds, 97.5)), 4)],
    }


def bootstrap_d_clustered(records: list[dict], n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Item-clustered bootstrap CI for Cohen's d.

    The 63 test items are fixed stimuli repeated across all (model, polarity,
    context_length) cells; rows therefore share within-item correlation that
    a row-level resample treats as independent. Following the council audit
    (2026-05-26) we resample item IDs with replacement and include every row
    for each sampled item -- the standard block-bootstrap for clustered data.
    """
    if not records:
        return {"n": 0, "n_items": 0, "d": None, "ci95": [None, None]}
    by_item: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_item[r["item_id"]].append(r["bias_score"])
    item_ids = list(by_item.keys())
    full = np.array([v for vals in by_item.values() for v in vals])
    point_d = float(full.mean() / full.std(ddof=1)) if full.std(ddof=1) > 0 else 0.0

    rng = np.random.default_rng(seed)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(item_ids, size=len(item_ids), replace=True)
        vals = np.concatenate([by_item[iid] for iid in sampled])
        sd = vals.std(ddof=1)
        ds[i] = vals.mean() / sd if sd > 0 else 0.0
    return {
        "n": int(len(full)),
        "n_items": int(len(item_ids)),
        "d": point_d,
        "ci95": [round(float(np.percentile(ds, 2.5)), 4),
                 round(float(np.percentile(ds, 97.5)), 4)],
    }


def stratified_bootstrap(records: list[dict], group_key: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r[group_key]].append(r)
    for grp, recs in groups.items():
        out[grp] = bootstrap_d_clustered(recs)
    # overall
    out["__overall__"] = bootstrap_d_clustered(records)
    return out


# --------------------------- consistency rate --------------------------


def consistency_rate(rows: list[dict]) -> dict:
    """For each (model, item, polarity, context_length) cell, compute the
    modal binary response (yes / no / tie) from 10 reps, then compare with
    the model's modal baseline response for that item. CR = fraction of
    cells where modal_treat == modal_base.

    A 100% CR means treatment never shifted the binary outcome.
    Lower CR = more bias-induced flipping.
    """
    # group rows
    base_modal: dict[tuple[str, str], str] = {}
    treat_modal: dict[tuple[str, str, str, int], str] = {}

    base_cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    treat_cells: dict[tuple[str, str, str, int], list[str]] = defaultdict(list)
    for r in rows:
        pol = r["polarity"]
        if r["parsed_response"] not in ("yes", "no"):
            continue
        if pol == "baseline":
            base_cells[(r["model"], r["test_item_id"])].append(r["parsed_response"])
        else:
            treat_cells[(r["model"], r["test_item_id"], pol, r["context_length"])].append(r["parsed_response"])

    def modal(labels: list[str]) -> str:
        c = {l: labels.count(l) for l in set(labels)}
        if len(c) == 1:
            return next(iter(c))
        items = sorted(c.items(), key=lambda x: -x[1])
        return "tie" if len(items) > 1 and items[0][1] == items[1][1] else items[0][0]

    base_modal = {k: modal(v) for k, v in base_cells.items()}
    treat_modal = {k: modal(v) for k, v in treat_cells.items()}

    # Per-polarity + per-domain CR
    per_polarity: dict[str, dict] = defaultdict(lambda: {"matched": 0, "n": 0})
    per_domain: dict[str, dict] = defaultdict(lambda: {"matched": 0, "n": 0})
    overall = {"matched": 0, "n": 0}

    # need domain per item
    item_domain = {r["test_item_id"]: r["domain"] for r in rows}

    for (m, iid, pol, ctx), tmodal in treat_modal.items():
        bm = base_modal.get((m, iid))
        if bm is None:
            continue
        matched = (bm == tmodal) and bm in ("yes", "no")
        per_polarity[pol]["n"] += 1
        per_polarity[pol]["matched"] += int(matched)
        per_domain[item_domain[iid]]["n"] += 1
        per_domain[item_domain[iid]]["matched"] += int(matched)
        overall["n"] += 1
        overall["matched"] += int(matched)

    def rate(c):
        return round(c["matched"] / c["n"], 4) if c["n"] else 0.0

    return {
        "overall": {**overall, "consistency_rate": rate(overall)},
        "per_polarity": {p: {**c, "consistency_rate": rate(c)} for p, c in per_polarity.items()},
        "per_domain": {d: {**c, "consistency_rate": rate(c)} for d, c in per_domain.items()},
        "interpretation": "Lower CR = more bias-induced flipping. CR=1.0 means treatment never shifted the binary outcome. AMEL predicts CR < 1.0, especially on ambiguous items and under no-saturated context.",
    }


# --------------------------- main --------------------------------------


def main() -> None:
    print(f"Loading dataset: {os.environ.get('AMEL_DATA_FILE', 'data/all_results.jsonl')}")
    rows = load_results()
    bs = compute_bias_scores(rows)
    print(f"  rows: {len(rows):,}  bias-score cells: {len(bs):,}")

    # Bootstrap CIs (item-clustered)
    print("\nBootstrap CIs (1000 item-clustered resamples)...")
    boot = {
        "n_resamples": N_BOOT,
        "seed": SEED,
        "cluster_unit": "item_id",
        "method": "block-bootstrap: resample item IDs with replacement, include all rows per sampled item",
        "headline_overall": bootstrap_d_clustered(bs),
        "per_polarity": stratified_bootstrap(bs, "polarity"),
        "per_domain": stratified_bootstrap(bs, "domain"),
        "per_category": stratified_bootstrap(bs, "category"),
    }
    (RESULTS_DIR / "bootstrap_cis.json").write_text(json.dumps(boot, indent=2))
    print(f"  overall d = {boot['headline_overall']['d']:+.3f}  "
          f"95% CI [{boot['headline_overall']['ci95'][0]:+.3f}, {boot['headline_overall']['ci95'][1]:+.3f}]")
    for d, info in boot["per_domain"].items():
        if d == "__overall__":
            continue
        print(f"  {d:25s} d = {info['d']:+.3f}  CI [{info['ci95'][0]:+.3f}, {info['ci95'][1]:+.3f}]  (n={info['n']})")
    print(f"\nSaved: {RESULTS_DIR / 'bootstrap_cis.json'}")

    # Consistency rate
    print("\nConsistency rate (modal treatment matches modal baseline)...")
    cr = consistency_rate(rows)
    (RESULTS_DIR / "consistency_rate.json").write_text(json.dumps(cr, indent=2))
    print(f"  overall CR = {cr['overall']['consistency_rate']:.1%}  (matched {cr['overall']['matched']:,}/{cr['overall']['n']:,} cells)")
    for p, info in cr["per_polarity"].items():
        print(f"  {p:15s} CR = {info['consistency_rate']:.1%}  (n={info['n']})")
    for d, info in cr["per_domain"].items():
        print(f"  {d:25s} CR = {info['consistency_rate']:.1%}  (n={info['n']})")
    print(f"\nSaved: {RESULTS_DIR / 'consistency_rate.json'}")


if __name__ == "__main__":
    main()
