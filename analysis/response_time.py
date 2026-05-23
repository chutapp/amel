"""Response time analysis.

Compares response_time_ms between baseline and treatment conditions,
controlling for context length. Also correlates per-item response time
variability with |BS|.
"""

from collections import defaultdict

import numpy as np
from scipy import stats

from analysis.utils import load_results, compute_bias_scores, save_json


def main():
    print("Loading data...")
    results = load_results()
    scores = compute_bias_scores(results)

    # Separate baseline and treatment response times
    baseline_times = []
    treatment_times = defaultdict(list)  # keyed by context_length

    for r in results:
        if r["parsed_response"] is None:
            continue
        rt = r.get("response_time_ms")
        if rt is None or rt <= 0:
            continue
        if r["polarity"] == "baseline":
            baseline_times.append(rt)
        else:
            treatment_times[r["context_length"]].append(rt)

    print(f"Baseline response times: n={len(baseline_times)}")

    # Overall baseline vs treatment comparison
    all_treatment = []
    for times in treatment_times.values():
        all_treatment.extend(times)

    bl_arr = np.array(baseline_times)
    tx_arr = np.array(all_treatment)

    t_overall, p_overall = stats.ttest_ind(bl_arr, tx_arr, equal_var=False)
    d_overall = (np.mean(tx_arr) - np.mean(bl_arr)) / np.sqrt((np.std(bl_arr)**2 + np.std(tx_arr)**2) / 2)

    print(f"Baseline: mean={np.mean(bl_arr):.0f}ms, Treatment: mean={np.mean(tx_arr):.0f}ms")
    print(f"t={t_overall:.4f}, p={p_overall:.2e}, d={d_overall:.4f}")

    # Per context length
    per_length = {}
    for cl in sorted(treatment_times.keys()):
        tx = np.array(treatment_times[cl])
        t, p = stats.ttest_ind(bl_arr, tx, equal_var=False)
        per_length[str(cl)] = {
            "n_treatment": len(tx),
            "mean_treatment_ms": round(float(np.mean(tx)), 1),
            "mean_baseline_ms": round(float(np.mean(bl_arr)), 1),
            "t": round(float(t), 4),
            "p": float(p),
        }

    # Per-item response time variability vs |BS|
    # Group response times by item x model
    item_rt_var = defaultdict(list)
    for r in results:
        if r["parsed_response"] is None:
            continue
        rt = r.get("response_time_ms")
        if rt is None or rt <= 0:
            continue
        if r["polarity"] != "baseline":
            key = f"{r['domain']}|{r['model']}|{r['test_item_id']}|{r['polarity']}|{r['context_length']}"
            item_rt_var[key].append(rt)

    # Build score-to-key map
    score_map = {}
    for s in scores:
        key = f"{s['domain']}|{s['model']}|{s['item_id']}|{s['polarity']}|{s['context_length']}"
        score_map[key] = abs(s["bias_score"])

    rt_cvs = []
    abs_bs_vals = []
    for key, times in item_rt_var.items():
        if key in score_map and len(times) >= 3:
            mean_rt = np.mean(times)
            if mean_rt > 0:
                cv = np.std(times) / mean_rt  # coefficient of variation
                rt_cvs.append(cv)
                abs_bs_vals.append(score_map[key])

    if len(rt_cvs) > 2:
        r_rt, p_rt = stats.spearmanr(rt_cvs, abs_bs_vals)
    else:
        r_rt, p_rt = 0, 1

    output = {
        "baseline_vs_treatment": {
            "n_baseline": len(bl_arr),
            "n_treatment": len(tx_arr),
            "mean_baseline_ms": round(float(np.mean(bl_arr)), 1),
            "mean_treatment_ms": round(float(np.mean(tx_arr)), 1),
            "t": round(float(t_overall), 4),
            "p": float(p_overall),
            "cohens_d": round(float(d_overall), 4),
            "significant": float(p_overall) < 0.05,
        },
        "per_context_length": per_length,
        "rt_variability_vs_bias": {
            "n_pairs": len(rt_cvs),
            "spearman_r": round(float(r_rt), 4),
            "spearman_p": float(p_rt),
            "significant": float(p_rt) < 0.05,
            "interpretation": "Higher RT variability associated with larger bias" if r_rt > 0 and p_rt < 0.05 else "No significant relationship",
        },
    }

    save_json(output, "results/response_time.json")


if __name__ == "__main__":
    main()
