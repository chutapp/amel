"""Logprobs analysis (Phase 1).

Analyzes first-token probability distributions across conditions.
Shows that P(Yes) shifts continuously — not just binary flips.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from analysis.utils import save_json


def load_logprobs_results(path=None):
    """Load logprobs experiment results."""
    path = path or Path("data/logprobs/results.jsonl")
    if not path.exists():
        print(f"WARNING: {path} not found. Run run_logprobs.py first.")
        return []
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def main():
    print("Loading logprobs results...")
    results = load_logprobs_results()
    if not results:
        print("No logprobs data found. Run run_logprobs.py first.")
        save_json({"status": "awaiting_data"}, "results/logprobs_analysis.json")
        return

    print(f"  Loaded {len(results)} results")

    # Group by condition
    conditions = [
        ("baseline", 0),
        ("no_saturated", 5),
        ("yes_saturated", 5),
        ("no_saturated", 50),
        ("yes_saturated", 50),
    ]

    condition_probs = {}
    for polarity, ctx_len in conditions:
        label = f"{polarity}@{ctx_len}" if polarity != "baseline" else "baseline"
        p_yes_vals = []
        p_no_vals = []

        for r in results:
            if r["polarity"] == polarity and r["context_length"] == ctx_len:
                if r.get("p_yes") is not None and r.get("p_no") is not None:
                    p_yes_vals.append(r["p_yes"])
                    p_no_vals.append(r["p_no"])

        if p_yes_vals:
            arr_yes = np.array(p_yes_vals)
            arr_no = np.array(p_no_vals)
            condition_probs[label] = {
                "n": len(arr_yes),
                "p_yes_mean": round(float(np.mean(arr_yes)), 6),
                "p_yes_median": round(float(np.median(arr_yes)), 6),
                "p_yes_std": round(float(np.std(arr_yes, ddof=1)), 6),
                "p_no_mean": round(float(np.mean(arr_no)), 6),
                "p_no_median": round(float(np.median(arr_no)), 6),
                "p_no_std": round(float(np.std(arr_no, ddof=1)), 6),
                "p_yes_values": [round(v, 6) for v in arr_yes.tolist()],
                "p_no_values": [round(v, 6) for v in arr_no.tolist()],
            }

    # Statistical tests: compare P(Yes) across conditions
    tests = {}

    # Baseline vs no_sat@5
    if "baseline" in condition_probs and "no_saturated@5" in condition_probs:
        bl = condition_probs["baseline"]["p_yes_values"]
        ns5 = condition_probs["no_saturated@5"]["p_yes_values"]
        u, p = stats.mannwhitneyu(bl, ns5, alternative="two-sided")
        tests["baseline_vs_no_sat_5"] = {
            "U": round(float(u), 2),
            "p": float(p),
            "direction": "P(Yes) drops under no-saturated context" if np.mean(ns5) < np.mean(bl) else "P(Yes) rises",
            "mean_diff": round(float(np.mean(ns5) - np.mean(bl)), 6),
        }

    # Baseline vs yes_sat@5
    if "baseline" in condition_probs and "yes_saturated@5" in condition_probs:
        bl = condition_probs["baseline"]["p_yes_values"]
        ys5 = condition_probs["yes_saturated@5"]["p_yes_values"]
        u, p = stats.mannwhitneyu(bl, ys5, alternative="two-sided")
        tests["baseline_vs_yes_sat_5"] = {
            "U": round(float(u), 2),
            "p": float(p),
            "direction": "P(Yes) rises under yes-saturated context" if np.mean(ys5) > np.mean(bl) else "P(Yes) drops",
            "mean_diff": round(float(np.mean(ys5) - np.mean(bl)), 6),
        }

    # no_sat@5 vs no_sat@50 (saturation test)
    if "no_saturated@5" in condition_probs and "no_saturated@50" in condition_probs:
        ns5 = condition_probs["no_saturated@5"]["p_yes_values"]
        ns50 = condition_probs["no_saturated@50"]["p_yes_values"]
        u, p = stats.mannwhitneyu(ns5, ns50, alternative="two-sided")
        tests["no_sat_5_vs_50"] = {
            "U": round(float(u), 2),
            "p": float(p),
            "direction": "No significant difference (saturation)" if p > 0.05 else "Significant difference",
            "mean_diff": round(float(np.mean(ns50) - np.mean(ns5)), 6),
        }

    # yes_sat@5 vs yes_sat@50
    if "yes_saturated@5" in condition_probs and "yes_saturated@50" in condition_probs:
        ys5 = condition_probs["yes_saturated@5"]["p_yes_values"]
        ys50 = condition_probs["yes_saturated@50"]["p_yes_values"]
        u, p = stats.mannwhitneyu(ys5, ys50, alternative="two-sided")
        tests["yes_sat_5_vs_50"] = {
            "U": round(float(u), 2),
            "p": float(p),
            "direction": "No significant difference (saturation)" if p > 0.05 else "Significant difference",
            "mean_diff": round(float(np.mean(ys50) - np.mean(ys5)), 6),
        }

    # Per-item analysis: does every item shift in the same direction?
    item_shifts = defaultdict(dict)
    for r in results:
        if r.get("p_yes") is not None:
            item_id = r["test_item_id"]
            cond = f"{r['polarity']}@{r['context_length']}" if r["polarity"] != "baseline" else "baseline"
            item_shifts[item_id].setdefault(cond, []).append(r["p_yes"])

    per_item = {}
    for item_id, cond_vals in item_shifts.items():
        bl_mean = np.mean(cond_vals.get("baseline", [0]))
        item_data = {"baseline_p_yes": round(float(bl_mean), 4)}
        for cond in ["no_saturated@5", "yes_saturated@5", "no_saturated@50", "yes_saturated@50"]:
            if cond in cond_vals:
                cond_mean = np.mean(cond_vals[cond])
                item_data[f"{cond}_p_yes"] = round(float(cond_mean), 4)
                item_data[f"{cond}_shift"] = round(float(cond_mean - bl_mean), 4)
        per_item[item_id] = item_data

    # Check polarity separation: for each item, is P(Yes|yes_sat) > P(Yes|no_sat)?
    # This is the key comparison: do the two polarities create different distributions?
    polarity_separation = {}
    if "no_saturated@5" in condition_probs and "yes_saturated@5" in condition_probs:
        n_separated = 0
        n_reversed = 0
        n_items = 0
        for item_id, d in per_item.items():
            if "no_saturated@5_p_yes" in d and "yes_saturated@5_p_yes" in d:
                n_items += 1
                if d["yes_saturated@5_p_yes"] > d["no_saturated@5_p_yes"]:
                    n_separated += 1
                elif d["yes_saturated@5_p_yes"] < d["no_saturated@5_p_yes"]:
                    n_reversed += 1
        polarity_separation["at_5_turns"] = {
            "n_items": n_items,
            "n_yes_sat_higher": n_separated,
            "n_no_sat_higher": n_reversed,
            "fraction_separated": round(n_separated / n_items, 3) if n_items > 0 else 0,
        }

    # Check context shift from baseline: does context generally shift P(Yes)?
    context_shift = {}
    for cond in ["no_saturated@5", "yes_saturated@5"]:
        shifts = [v.get(f"{cond}_shift", 0) for v in per_item.values() if f"{cond}_shift" in v]
        if shifts:
            n_up = sum(1 for s in shifts if s > 0)
            n_down = sum(1 for s in shifts if s < 0)
            context_shift[cond] = {
                "n_items": len(shifts),
                "n_shifted_up": n_up,
                "n_shifted_down": n_down,
                "mean_shift": round(float(np.mean(shifts)), 4),
            }

    # Strip raw values from condition_probs for the output (keep summary stats)
    condition_summary = {}
    for label, data in condition_probs.items():
        condition_summary[label] = {k: v for k, v in data.items() if k not in ("p_yes_values", "p_no_values")}

    output = {
        "conditions": condition_summary,
        "statistical_tests": tests,
        "polarity_separation": polarity_separation,
        "context_shift": context_shift,
        "per_item_shifts": per_item,
    }

    save_json(output, "results/logprobs_analysis.json")

    print("\nSummary:")
    for label, data in condition_summary.items():
        print(f"  {label}: P(Yes)={data['p_yes_mean']:.4f} ± {data['p_yes_std']:.4f}, "
              f"P(No)={data['p_no_mean']:.4f} ± {data['p_no_std']:.4f}")
    print(f"\n  Polarity separation: {polarity_separation}")
    print(f"  Context shift: {context_shift}")


if __name__ == "__main__":
    main()
