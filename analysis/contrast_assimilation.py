"""Contrast vs. Assimilation analysis.

Classifies each bias score as congruent or incongruent based on whether
the context polarity aligns with the item's ground truth, then tests
whether the effect is assimilation (shift toward context) or contrast
(shift away from context for incongruent items).
"""

import numpy as np
from scipy import stats

from analysis.utils import load_results, compute_bias_scores, save_json


def classify_congruence(score):
    """Determine if context polarity is congruent with item ground truth.

    Congruent: no_saturated + ground_truth=no, or yes_saturated + ground_truth=yes
    Incongruent: no_saturated + ground_truth=yes, or yes_saturated + ground_truth=no
    Neutral polarity is excluded (no clear directionality).
    """
    polarity = score["polarity"]
    gt = score["ground_truth"]

    if polarity == "neutral":
        return "neutral"
    if polarity == "no_saturated":
        return "congruent" if gt == "no" else "incongruent"
    if polarity == "yes_saturated":
        return "congruent" if gt == "yes" else "incongruent"
    return "unknown"


def main():
    print("Loading data...")
    results = load_results()
    scores = compute_bias_scores(results)

    # Classify each score
    congruent_bs = []
    incongruent_bs = []

    # Per-item paired aggregation: each item contributes one mean BS in each
    # group, allowing a paired test that respects the within-item dependency
    # (the same 63 stimuli appear in both groups under different polarities).
    cong_by_item: dict = {}
    incong_by_item: dict = {}

    for s in scores:
        label = classify_congruence(s)
        if label == "congruent":
            congruent_bs.append(s["bias_score"])
            cong_by_item.setdefault(s["item_id"], []).append(s["bias_score"])
        elif label == "incongruent":
            incongruent_bs.append(s["bias_score"])
            incong_by_item.setdefault(s["item_id"], []).append(s["bias_score"])

    cong = np.array(congruent_bs)
    incong = np.array(incongruent_bs)

    print(f"Congruent: n={len(cong)}, mean={np.mean(cong):.4f}")
    print(f"Incongruent: n={len(incong)}, mean={np.mean(incong):.4f}")

    # Unpaired test (retained for transparency / backward comparison)
    t_stat, p_val = stats.ttest_ind(cong, incong, equal_var=False)
    pooled_std = np.sqrt((np.std(cong, ddof=1)**2 + np.std(incong, ddof=1)**2) / 2)
    d = (np.mean(cong) - np.mean(incong)) / pooled_std if pooled_std > 0 else 0

    # Paired-by-item test (council audit 2026-05-26): the 63 items are fixed
    # stimuli; each contributes one (mean congruent, mean incongruent) pair.
    paired_items = sorted(set(cong_by_item) & set(incong_by_item))
    cong_per_item = np.array([np.mean(cong_by_item[iid]) for iid in paired_items])
    incong_per_item = np.array([np.mean(incong_by_item[iid]) for iid in paired_items])
    diff = cong_per_item - incong_per_item
    paired_t, paired_p = stats.ttest_rel(cong_per_item, incong_per_item)
    paired_d = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0
    paired_ci = stats.t.interval(0.95, len(diff) - 1, loc=np.mean(diff), scale=stats.sem(diff))

    # Both positive BS = assimilation (shifting toward context polarity)
    # Congruent > Incongruent = stronger assimilation when context matches ground truth
    both_positive = np.mean(cong) > 0 and np.mean(incong) > 0
    interpretation = "assimilation" if both_positive else "mixed"
    if np.mean(incong) < 0:
        interpretation = "contrast_for_incongruent"

    # Per-category breakdown
    per_category = {}
    for cat in ["clear_positive", "ambiguous", "clear_negative"]:
        cat_cong = [s["bias_score"] for s in scores if classify_congruence(s) == "congruent" and s["category"] == cat]
        cat_incong = [s["bias_score"] for s in scores if classify_congruence(s) == "incongruent" and s["category"] == cat]
        if cat_cong and cat_incong:
            t_c, p_c = stats.ttest_ind(cat_cong, cat_incong, equal_var=False)
            per_category[cat] = {
                "congruent_mean": round(float(np.mean(cat_cong)), 6),
                "congruent_n": len(cat_cong),
                "incongruent_mean": round(float(np.mean(cat_incong)), 6),
                "incongruent_n": len(cat_incong),
                "t": round(float(t_c), 4),
                "p": float(p_c),
            }

    output = {
        "congruent": {
            "n": len(cong),
            "mean": round(float(np.mean(cong)), 6),
            "std": round(float(np.std(cong, ddof=1)), 6),
            "ci_lower": round(float(np.mean(cong) - 1.96 * np.std(cong) / np.sqrt(len(cong))), 6),
            "ci_upper": round(float(np.mean(cong) + 1.96 * np.std(cong) / np.sqrt(len(cong))), 6),
        },
        "incongruent": {
            "n": len(incong),
            "mean": round(float(np.mean(incong)), 6),
            "std": round(float(np.std(incong, ddof=1)), 6),
            "ci_lower": round(float(np.mean(incong) - 1.96 * np.std(incong) / np.sqrt(len(incong))), 6),
            "ci_upper": round(float(np.mean(incong) + 1.96 * np.std(incong) / np.sqrt(len(incong))), 6),
        },
        "t_test_unpaired": {
            "t": round(float(t_stat), 4),
            "p": float(p_val),
            "cohens_d": round(float(d), 4),
            "significant": float(p_val) < 0.05,
            "note": "Unpaired Welch test retained for backward comparison only. The published estimate is t_test_paired below (council audit 2026-05-26: 63 items are fixed stimuli shared across both groups).",
        },
        "t_test_paired": {
            "n_items": int(len(paired_items)),
            "mean_diff": round(float(np.mean(diff)), 6),
            "ci95_diff": [round(float(paired_ci[0]), 6), round(float(paired_ci[1]), 6)],
            "t": round(float(paired_t), 4),
            "p": float(paired_p),
            "cohens_d": round(paired_d, 4),
            "significant": float(paired_p) < 0.05,
        },
        "interpretation": interpretation,
        "per_category": per_category,
    }

    print(f"\nInterpretation: {interpretation}")
    print(f"unpaired: t={t_stat:.4f}, p={p_val:.2e}, d={d:.4f}")
    print(f"paired-by-item (n={len(paired_items)}): t={paired_t:.4f}, p={paired_p:.2e}, d={paired_d:.4f}")

    save_json(output, "results/contrast_assimilation.json")


if __name__ == "__main__":
    main()
