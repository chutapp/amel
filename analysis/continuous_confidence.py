"""Continuous confidence analysis.

Computes baseline entropy per item x model as a proxy for model confidence,
then tests whether low-confidence items (high entropy) show larger bias.
"""

from collections import defaultdict

import numpy as np
from scipy import stats

from analysis.utils import load_results, compute_bias_scores, save_json


def compute_baseline_entropy(results):
    """Compute binary entropy of baseline responses per item x model.

    H = -p*log2(p) - (1-p)*log2(1-p) where p = P(yes|baseline).
    """
    baseline_groups = defaultdict(list)
    for r in results:
        if r["polarity"] == "baseline" and r["parsed_response"] is not None:
            key = f"{r['domain']}|{r['model']}|{r['test_item_id']}"
            baseline_groups[key].append(r["parsed_response"])

    entropy_map = {}
    for key, responses in baseline_groups.items():
        if len(responses) < 2:
            continue
        p_yes = sum(1 for r in responses if r == "yes") / len(responses)
        # Binary entropy
        if p_yes == 0 or p_yes == 1:
            h = 0.0
        else:
            h = -p_yes * np.log2(p_yes) - (1 - p_yes) * np.log2(1 - p_yes)
        entropy_map[key] = {
            "entropy": h,
            "p_yes": p_yes,
            "n": len(responses),
        }
    return entropy_map


def main():
    print("Loading data...")
    results = load_results()
    scores = compute_bias_scores(results)

    print("Computing baseline entropy...")
    entropy_map = compute_baseline_entropy(results)

    # Match entropy to bias scores
    entropies = []
    abs_bias = []
    raw_bias = []
    categories = []

    for s in scores:
        key = f"{s['domain']}|{s['model']}|{s['item_id']}"
        if key in entropy_map:
            entropies.append(entropy_map[key]["entropy"])
            abs_bias.append(abs(s["bias_score"]))
            raw_bias.append(s["bias_score"])
            categories.append(s["category"])

    ent_arr = np.array(entropies)
    abs_arr = np.array(abs_bias)

    print(f"Matched {len(ent_arr)} score-entropy pairs")

    # Overall correlation
    r_sp, p_sp = stats.spearmanr(ent_arr, abs_arr)
    r_pe, p_pe = stats.pearsonr(ent_arr, abs_arr)
    print(f"Spearman: r={r_sp:.4f}, p={p_sp:.2e}")
    print(f"Pearson:  r={r_pe:.4f}, p={p_pe:.2e}")

    # Bin by entropy level
    bins = {"zero_entropy": [], "low_entropy": [], "high_entropy": []}
    for e, b in zip(ent_arr, abs_arr):
        if e == 0:
            bins["zero_entropy"].append(b)
        elif e < 0.5:
            bins["low_entropy"].append(b)
        else:
            bins["high_entropy"].append(b)

    bin_stats = {}
    for bname, bvals in bins.items():
        if bvals:
            bin_stats[bname] = {
                "n": len(bvals),
                "mean_abs_bs": round(float(np.mean(bvals)), 6),
                "std": round(float(np.std(bvals, ddof=1)), 6),
            }

    # Per-category
    per_category = {}
    for cat in ["clear_positive", "ambiguous", "clear_negative"]:
        cat_ent = [e for e, c in zip(ent_arr, categories) if c == cat]
        cat_abs = [b for b, c in zip(abs_arr, categories) if c == cat]
        if len(cat_ent) > 2:
            r, p = stats.spearmanr(cat_ent, cat_abs)
            per_category[cat] = {
                "n": len(cat_ent),
                "mean_entropy": round(float(np.mean(cat_ent)), 4),
                "spearman_r": round(float(r), 4),
                "spearman_p": float(p),
            }

    output = {
        "n_pairs": len(ent_arr),
        "overall_spearman": {
            "r": round(float(r_sp), 6),
            "p": float(p_sp),
            "significant": float(p_sp) < 0.05,
        },
        "overall_pearson": {
            "r": round(float(r_pe), 6),
            "p": float(p_pe),
            "significant": float(p_pe) < 0.05,
        },
        "entropy_bins": bin_stats,
        "per_category": per_category,
    }

    save_json(output, "results/continuous_confidence.json")


if __name__ == "__main__":
    main()
