"""Mitigation experiment analysis.

Compares fresh (baseline) vs sequential_fixed vs sequential_balanced.
Tracks position-dependent bias within sequential batches.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from analysis.utils import load_results as load_main_results, save_json


def load_mitigation_results(path=None):
    """Load mitigation experiment results."""
    path = path or Path("data/mitigation/results.jsonl")
    if not path.exists():
        print(f"WARNING: {path} not found. Run run_mitigation.py first.")
        return []
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def main():
    print("Loading mitigation results...")
    mit_results = load_mitigation_results()
    if not mit_results:
        print("No mitigation data found. Creating placeholder output.")
        save_json({"status": "awaiting_data", "note": "Run run_mitigation.py first"}, "results/mitigation_analysis.json")
        return

    print(f"  Loaded {len(mit_results)} mitigation results")

    # Also load main experiment baseline for comparison
    print("Loading main experiment baseline...")
    main_results = load_main_results()
    baseline_results = [r for r in main_results if r["polarity"] == "baseline" and r["parsed_response"] is not None]

    # Group baseline by model x domain x item
    baseline_rates = defaultdict(list)
    for r in baseline_results:
        key = f"{r['domain']}|{r['model']}|{r['test_item_id']}"
        baseline_rates[key].append(r["parsed_response"])

    # Compute P(no) for baseline
    baseline_pno = {}
    for key, responses in baseline_rates.items():
        baseline_pno[key] = sum(1 for r in responses if r == "no") / len(responses)

    # Analyze sequential results
    # Group by condition x model x domain x item
    seq_groups = defaultdict(list)
    for r in mit_results:
        if r["parsed_response"] is None:
            continue
        key = f"{r['condition']}|{r['domain']}|{r['model']}|{r['test_item_id']}"
        seq_groups[key].append(r)

    # Compute bias scores: P(no|sequential) - P(no|baseline)
    condition_scores = defaultdict(list)
    for key, group in seq_groups.items():
        condition, domain, model, item_id = key.split("|")
        bl_key = f"{domain}|{model}|{item_id}"
        if bl_key not in baseline_pno:
            continue

        pno_seq = sum(1 for r in group if r["parsed_response"] == "no") / len(group)
        pno_bl = baseline_pno[bl_key]
        bs = pno_seq - pno_bl

        condition_scores[condition].append({
            "bias_score": bs,
            "model": model,
            "domain": domain,
            "item_id": item_id,
            "category": group[0]["test_item_category"],
        })

    # Per-condition stats
    output = {"conditions": {}}
    for cond in ["sequential_fixed", "sequential_balanced"]:
        scores = [s["bias_score"] for s in condition_scores.get(cond, [])]
        if scores:
            arr = np.array(scores)
            t, p = stats.ttest_1samp(arr, 0)
            d = float(np.mean(arr) / np.std(arr, ddof=1)) if np.std(arr, ddof=1) > 0 else 0
            output["conditions"][cond] = {
                "n": len(arr),
                "mean": round(float(np.mean(arr)), 6),
                "std": round(float(np.std(arr, ddof=1)), 6),
                "cohens_d": round(d, 4),
                "t": round(float(t), 4),
                "p": float(p),
                "significant": float(p) < 0.05,
            }

    # Compare fixed vs balanced
    fixed_bs = [s["bias_score"] for s in condition_scores.get("sequential_fixed", [])]
    balanced_bs = [s["bias_score"] for s in condition_scores.get("sequential_balanced", [])]
    if fixed_bs and balanced_bs:
        t_comp, p_comp = stats.ttest_ind(fixed_bs, balanced_bs, equal_var=False)
        output["fixed_vs_balanced"] = {
            "t": round(float(t_comp), 4),
            "p": float(p_comp),
            "significant": float(p_comp) < 0.05,
            "fixed_mean_abs": round(float(np.mean(np.abs(fixed_bs))), 4),
            "balanced_mean_abs": round(float(np.mean(np.abs(balanced_bs))), 4),
        }

    # Position-dependent bias within sequential batches
    position_bias = defaultdict(lambda: defaultdict(list))
    for r in mit_results:
        if r["parsed_response"] is None:
            continue
        position_bias[r["condition"]][r["position"]].append(
            1 if r["parsed_response"] == "no" else 0
        )

    output["position_analysis"] = {}
    for cond in ["sequential_fixed", "sequential_balanced"]:
        positions = sorted(position_bias.get(cond, {}).keys())
        if positions:
            means = [np.mean(position_bias[cond][p]) for p in positions]
            r_sp, p_sp = stats.spearmanr(positions, means) if len(positions) > 2 else (0, 1)
            output["position_analysis"][cond] = {
                "positions": positions,
                "pno_by_position": [round(m, 4) for m in means],
                "spearman_r": round(float(r_sp), 4),
                "spearman_p": float(p_sp),
                "drift_detected": float(p_sp) < 0.05,
            }

    save_json(output, "results/mitigation_analysis.json")
    print("\nSummary:")
    for cond, stats_dict in output.get("conditions", {}).items():
        print(f"  {cond}: d={stats_dict['cohens_d']}, p={stats_dict['p']:.2e}")


if __name__ == "__main__":
    main()
