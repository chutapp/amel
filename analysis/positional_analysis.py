"""Positional placement analysis (Phase 3).

Compares bias across START, END, and SPREAD placements.
Tests primacy vs recency vs any-signal-suffices hypotheses.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from analysis.utils import load_results as load_main_results, save_json


def load_positional_results(path=None):
    """Load positional experiment results."""
    path = path or Path("data/positional/results.jsonl")
    if not path.exists():
        print(f"WARNING: {path} not found. Run run_positional.py first.")
        return []
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def main():
    print("Loading positional results...")
    pos_results = load_positional_results()
    if not pos_results:
        print("No positional data found. Run run_positional.py first.")
        save_json({"status": "awaiting_data"}, "results/positional_analysis.json")
        return

    print(f"  Loaded {len(pos_results)} positional results")

    # Also load main experiment baseline data for the same models/domain
    print("Loading main experiment for baselines...")
    main_results = load_main_results()

    models = set(r["model"] for r in pos_results)
    domain = "code_review"

    # Compute baseline P(no) from main experiment
    baseline_groups = defaultdict(list)
    for r in main_results:
        if (r["polarity"] == "baseline" and r["parsed_response"] is not None
                and r["model"] in models and r["domain"] == domain):
            key = f"{r['model']}|{r['test_item_id']}"
            baseline_groups[key].append(r["parsed_response"])

    baseline_pno = {}
    for key, responses in baseline_groups.items():
        baseline_pno[key] = sum(1 for r in responses if r == "no") / len(responses)

    # Also get FULL_50 data (no_saturated@50) from main experiment
    full50_groups = defaultdict(list)
    for r in main_results:
        if (r["polarity"] == "no_saturated" and r["context_length"] == 50
                and r["parsed_response"] is not None
                and r["model"] in models and r["domain"] == domain):
            key = f"{r['model']}|{r['test_item_id']}"
            full50_groups[key].append(r["parsed_response"])

    # And CONTROL_5 data (no_saturated@5)
    ctrl5_groups = defaultdict(list)
    for r in main_results:
        if (r["polarity"] == "no_saturated" and r["context_length"] == 5
                and r["parsed_response"] is not None
                and r["model"] in models and r["domain"] == domain):
            key = f"{r['model']}|{r['test_item_id']}"
            ctrl5_groups[key].append(r["parsed_response"])

    # Compute bias scores for each placement
    placements = ["START", "END", "SPREAD"]
    placement_bs = defaultdict(lambda: defaultdict(list))  # model → placement → [bs]

    for placement in placements:
        placement_items = defaultdict(list)
        for r in pos_results:
            if r["placement"] == placement and r["parsed_response"] is not None:
                key = f"{r['model']}|{r['test_item_id']}"
                placement_items[key].append(r)

        for key, group in placement_items.items():
            if key not in baseline_pno:
                continue
            bl_pno = baseline_pno[key]
            tx_pno = sum(1 for r in group if r["parsed_response"] == "no") / len(group)
            bs = tx_pno - bl_pno
            model = key.split("|")[0]
            placement_bs[model][placement].append(bs)

    # Also compute BS for FULL_50 and CONTROL_5 from main experiment
    for key, group in full50_groups.items():
        if key not in baseline_pno:
            continue
        bl_pno = baseline_pno[key]
        tx_pno = sum(1 for r in group if r == "no") / len(group)
        bs = tx_pno - bl_pno
        model = key.split("|")[0]
        placement_bs[model]["FULL_50"].append(bs)

    for key, group in ctrl5_groups.items():
        if key not in baseline_pno:
            continue
        bl_pno = baseline_pno[key]
        tx_pno = sum(1 for r in group if r == "no") / len(group)
        bs = tx_pno - bl_pno
        model = key.split("|")[0]
        placement_bs[model]["CONTROL_5"].append(bs)

    # Per-model analysis
    output = {"per_model": {}, "combined": {}}

    all_condition_scores = defaultdict(list)
    for model, conditions in placement_bs.items():
        model_output = {}
        for cond, scores in conditions.items():
            arr = np.array(scores)
            t_stat, p_val = stats.ttest_1samp(arr, 0) if len(arr) > 1 else (0, 1)
            d = float(np.mean(arr) / np.std(arr, ddof=1)) if np.std(arr, ddof=1) > 0 else 0
            model_output[cond] = {
                "n": len(arr),
                "mean_bs": round(float(np.mean(arr)), 4),
                "std_bs": round(float(np.std(arr, ddof=1)), 4) if len(arr) > 1 else 0,
                "cohens_d": round(d, 4),
                "t": round(float(t_stat), 4),
                "p": float(p_val),
            }
            all_condition_scores[cond].extend(scores)

        # Kruskal-Wallis across placements for this model
        kw_groups = [np.array(conditions[p]) for p in placements if p in conditions and len(conditions[p]) > 1]
        if len(kw_groups) >= 2:
            h_stat, p_kw = stats.kruskal(*kw_groups)
            model_output["kruskal_wallis_placements"] = {
                "H": round(float(h_stat), 4),
                "p": float(p_kw),
                "significant": float(p_kw) < 0.05,
            }

        output["per_model"][model] = model_output

    # Combined analysis across models
    for cond, scores in all_condition_scores.items():
        arr = np.array(scores)
        t_stat, p_val = stats.ttest_1samp(arr, 0) if len(arr) > 1 else (0, 1)
        d = float(np.mean(arr) / np.std(arr, ddof=1)) if np.std(arr, ddof=1) > 0 else 0
        output["combined"][cond] = {
            "n": len(arr),
            "mean_bs": round(float(np.mean(arr)), 4),
            "std_bs": round(float(np.std(arr, ddof=1)), 4) if len(arr) > 1 else 0,
            "cohens_d": round(d, 4),
            "t": round(float(t_stat), 4),
            "p": float(p_val),
        }

    # Kruskal-Wallis across all placements (combined)
    kw_combined = [np.array(all_condition_scores[p]) for p in placements if p in all_condition_scores and len(all_condition_scores[p]) > 1]
    if len(kw_combined) >= 2:
        h_stat, p_kw = stats.kruskal(*kw_combined)
        output["kruskal_wallis_combined"] = {
            "H": round(float(h_stat), 4),
            "p": float(p_kw),
            "significant": float(p_kw) < 0.05,
        }

    # Pairwise comparisons
    pairwise = {}
    for i, p1 in enumerate(placements):
        for p2 in placements[i+1:]:
            if p1 in all_condition_scores and p2 in all_condition_scores:
                arr1 = np.array(all_condition_scores[p1])
                arr2 = np.array(all_condition_scores[p2])
                if len(arr1) > 1 and len(arr2) > 1:
                    u, p = stats.mannwhitneyu(arr1, arr2, alternative="two-sided")
                    pairwise[f"{p1}_vs_{p2}"] = {
                        "U": round(float(u), 2),
                        "p": float(p),
                        "mean_diff": round(float(np.mean(arr1) - np.mean(arr2)), 4),
                    }
    output["pairwise"] = pairwise

    # Interpretation
    means = {cond: data["mean_bs"] for cond, data in output["combined"].items() if cond in placements}
    if means:
        sorted_placements = sorted(means.items(), key=lambda x: abs(x[1]), reverse=True)
        kw_sig = output.get("kruskal_wallis_combined", {}).get("significant", False)

        if not kw_sig:
            interpretation = "Position does not matter: START ≈ END ≈ SPREAD. Any signal suffices."
        elif sorted_placements[0][0] == "START":
            interpretation = "Primacy effect: biased turns at the START have stronger influence."
        elif sorted_placements[0][0] == "END":
            interpretation = "Recency effect: biased turns at the END have stronger influence."
        else:
            interpretation = "SPREAD placement shows strongest effect."

        output["interpretation"] = interpretation

    save_json(output, "results/positional_analysis.json")

    print("\nCombined results:")
    for cond in ["CONTROL_5", "START", "END", "SPREAD", "FULL_50"]:
        if cond in output["combined"]:
            d = output["combined"][cond]
            print(f"  {cond:>10}: BS={d['mean_bs']:+.4f}, d={d['cohens_d']:.3f}, p={d['p']:.2e}")

    if "kruskal_wallis_combined" in output:
        kw = output["kruskal_wallis_combined"]
        print(f"\n  Kruskal-Wallis (placements): H={kw['H']}, p={kw['p']:.4f}")
    if "interpretation" in output:
        print(f"\n  → {output['interpretation']}")


if __name__ == "__main__":
    main()
