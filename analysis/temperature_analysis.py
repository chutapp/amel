"""Temperature sensitivity analysis.

Compares bias scores at T=0.3, T=0.7, and T=1.0 (from main experiment).
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from analysis.utils import load_results as load_main_results, save_json


def load_temperature_results(path=None):
    """Load temperature experiment results."""
    path = path or Path("data/temperature/results.jsonl")
    if not path.exists():
        print(f"WARNING: {path} not found. Run run_temperature.py first.")
        return []
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_bs_for_temp(results, baseline_pno):
    """Compute bias scores for a set of temperature results."""
    groups = defaultdict(list)
    for r in results:
        if r["parsed_response"] is None:
            continue
        if r["polarity"] == "baseline":
            continue
        key = f"{r['domain']}|{r['model']}|{r['test_item_id']}|{r['polarity']}"
        groups[key].append(r)

    scores = []
    for key, group in groups.items():
        domain, model, item_id, polarity = key.split("|")
        bl_key = f"{domain}|{model}|{item_id}"
        if bl_key not in baseline_pno:
            continue
        target = "no" if polarity in ("no_saturated", "neutral") else "yes"
        tx_rate = sum(1 for r in group if r["parsed_response"] == target) / len(group)
        bl_rate = baseline_pno[bl_key] if target == "no" else (1 - baseline_pno[bl_key])
        bs = tx_rate - bl_rate
        scores.append(bs)
    return scores


def main():
    print("Loading temperature results...")
    temp_results = load_temperature_results()
    if not temp_results:
        print("No temperature data found. Creating placeholder output.")
        save_json({"status": "awaiting_data", "note": "Run run_temperature.py first"}, "results/temperature_analysis.json")
        return

    print(f"  Loaded {len(temp_results)} temperature results")

    # Load main experiment for T=1.0 comparison
    print("Loading main experiment data for T=1.0 comparison...")
    main_results = load_main_results()

    # Filter to same model and domain as temperature experiment
    model = "gpt-4.1-nano"
    domain = "code_review"

    # Compute baseline P(no) from temperature experiment baselines
    temp_baselines = defaultdict(lambda: defaultdict(list))
    for r in temp_results:
        if r["polarity"] == "baseline" and r["parsed_response"] is not None:
            temp = r.get("temperature", 1.0)
            key = f"{r['domain']}|{r['model']}|{r['test_item_id']}"
            temp_baselines[temp][key].append(r["parsed_response"])

    # Compute baseline P(no) from main experiment
    main_baselines = defaultdict(list)
    for r in main_results:
        if r["polarity"] == "baseline" and r["parsed_response"] is not None and r["model"] == model and r["domain"] == domain:
            key = f"{r['domain']}|{r['model']}|{r['test_item_id']}"
            main_baselines[key].append(r["parsed_response"])

    main_pno = {k: sum(1 for r in v if r == "no") / len(v) for k, v in main_baselines.items()}

    # Compute BS for each temperature
    output = {"temperatures": {}}

    for temp in [0.3, 0.7]:
        temp_subset = [r for r in temp_results if r.get("temperature") == temp]

        # Use temp-specific baseline if available
        pno = {}
        for key, responses in temp_baselines.get(temp, {}).items():
            pno[key] = sum(1 for r in responses if r == "no") / len(responses)

        if not pno:
            pno = main_pno  # fallback

        bs_scores = compute_bs_for_temp(temp_subset, pno)
        if bs_scores:
            arr = np.array(bs_scores)
            t, p = stats.ttest_1samp(arr, 0)
            d = float(np.mean(arr) / np.std(arr, ddof=1)) if np.std(arr, ddof=1) > 0 else 0
            output["temperatures"][str(temp)] = {
                "n": len(arr),
                "mean": round(float(np.mean(arr)), 6),
                "std": round(float(np.std(arr, ddof=1)), 6),
                "cohens_d": round(d, 4),
                "t": round(float(t), 4),
                "p": float(p),
                "significant": float(p) < 0.05,
            }

    # T=1.0 from main experiment (same model, same domain)
    main_subset = [r for r in main_results if r["model"] == model and r["domain"] == domain]
    bs_main = compute_bs_for_temp(main_subset, main_pno)
    if bs_main:
        arr = np.array(bs_main)
        t, p = stats.ttest_1samp(arr, 0)
        d = float(np.mean(arr) / np.std(arr, ddof=1)) if np.std(arr, ddof=1) > 0 else 0
        output["temperatures"]["1.0"] = {
            "n": len(arr),
            "mean": round(float(np.mean(arr)), 6),
            "std": round(float(np.std(arr, ddof=1)), 6),
            "cohens_d": round(d, 4),
            "t": round(float(t), 4),
            "p": float(p),
            "significant": float(p) < 0.05,
            "source": "main_experiment",
        }

    # Kruskal-Wallis test across temperatures
    all_temp_scores = []
    all_temp_labels = []
    for temp_str, temp_data in output.get("temperatures", {}).items():
        if temp_str == "1.0":
            all_temp_scores.append(bs_main)
        else:
            temp_val = float(temp_str)
            temp_subset = [r for r in temp_results if r.get("temperature") == temp_val]
            pno = {}
            for key, responses in temp_baselines.get(temp_val, {}).items():
                pno[key] = sum(1 for r in responses if r == "no") / len(responses)
            if not pno:
                pno = main_pno
            scores = compute_bs_for_temp(temp_subset, pno)
            all_temp_scores.append(scores)
        all_temp_labels.append(temp_str)

    valid_groups = [s for s in all_temp_scores if len(s) > 1]
    if len(valid_groups) >= 2:
        h_stat, p_kw = stats.kruskal(*valid_groups)
        output["kruskal_wallis"] = {
            "H": round(float(h_stat), 4),
            "p": float(p_kw),
            "significant": float(p_kw) < 0.05,
            "interpretation": "Temperature significantly affects bias magnitude" if p_kw < 0.05 else "No significant temperature effect",
        }

    save_json(output, "results/temperature_analysis.json")
    print("\nSummary:")
    for temp, data in output.get("temperatures", {}).items():
        print(f"  T={temp}: d={data.get('cohens_d', 'N/A')}, p={data.get('p', 'N/A')}")


if __name__ == "__main__":
    main()
