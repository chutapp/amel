"""Phase 0: Baseline tendency analysis (zero cost).

Tests whether items with a higher baseline P(no) also show stronger
negativity asymmetry. If positive Spearman r → negative context reinforces
a pre-existing 'no' lean.

Uses existing data only — no API calls.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from analysis.utils import load_results, save_json


def main():
    print("Loading main experiment data...")
    results = load_results()
    print(f"  Loaded {len(results)} results")

    # Compute baseline P(no) per item × model
    baseline_groups = defaultdict(list)
    for r in results:
        if r["polarity"] == "baseline" and r["parsed_response"] is not None:
            key = f"{r['domain']}|{r['model']}|{r['test_item_id']}"
            baseline_groups[key].append(r["parsed_response"])

    baseline_pno = {}
    for key, responses in baseline_groups.items():
        if len(responses) >= 2:
            baseline_pno[key] = sum(1 for r in responses if r == "no") / len(responses)

    # Compute per-item × model asymmetry: |BS(no_sat)| - |BS(yes_sat)|
    # Group treatment results
    treatment_groups = defaultdict(list)
    for r in results:
        if r["polarity"] in ("no_saturated", "yes_saturated") and r["parsed_response"] is not None:
            key = f"{r['domain']}|{r['model']}|{r['test_item_id']}|{r['polarity']}"
            treatment_groups[key].append(r)

    # Compute bias scores per item × model × polarity
    item_bs = defaultdict(dict)  # key → {polarity → bs}
    for key, group in treatment_groups.items():
        parts = key.split("|")
        domain, model, item_id, polarity = parts
        base_key = f"{domain}|{model}|{item_id}"
        if base_key not in baseline_pno:
            continue

        bl_pno = baseline_pno[base_key]
        if polarity == "no_saturated":
            tx_rate = sum(1 for r in group if r["parsed_response"] == "no") / len(group)
            bs = tx_rate - bl_pno
        else:  # yes_saturated
            tx_pyes = sum(1 for r in group if r["parsed_response"] == "yes") / len(group)
            bl_pyes = 1 - bl_pno
            bs = tx_pyes - bl_pyes

        item_bs[base_key][polarity] = bs

    # Compute asymmetry magnitude and correlate with baseline P(no)
    pno_values = []
    asymmetry_values = []
    item_details = []

    for base_key, pol_scores in item_bs.items():
        if "no_saturated" not in pol_scores or "yes_saturated" not in pol_scores:
            continue
        if base_key not in baseline_pno:
            continue

        asym = abs(pol_scores["no_saturated"]) - abs(pol_scores["yes_saturated"])
        pno = baseline_pno[base_key]
        pno_values.append(pno)
        asymmetry_values.append(asym)
        item_details.append({
            "key": base_key,
            "baseline_pno": round(pno, 4),
            "bs_no_sat": round(pol_scores["no_saturated"], 4),
            "bs_yes_sat": round(pol_scores["yes_saturated"], 4),
            "asymmetry": round(asym, 4),
        })

    pno_arr = np.array(pno_values)
    asym_arr = np.array(asymmetry_values)

    # Spearman correlation
    r_spearman, p_spearman = stats.spearmanr(pno_arr, asym_arr)

    # Pearson for comparison
    r_pearson, p_pearson = stats.pearsonr(pno_arr, asym_arr)

    # Binned analysis: split by baseline P(no) quartiles
    # Use terciles for cleaner binning (P(no) is often bimodal: 0 or 1)
    sorted_pno = np.sort(np.unique(pno_arr))
    terciles = np.percentile(pno_arr, [33, 67])
    bins = [
        ("Low P(no)", pno_arr <= terciles[0]),
        ("Mid P(no)", (pno_arr > terciles[0]) & (pno_arr <= terciles[1])),
        ("High P(no)", pno_arr > terciles[1]),
    ]

    binned = {}
    for label, mask in bins:
        if mask.sum() == 0:
            continue
        binned[label] = {
            "n": int(mask.sum()),
            "mean_pno": round(float(np.mean(pno_arr[mask])), 4),
            "mean_asymmetry": round(float(np.mean(asym_arr[mask])), 4),
            "std_asymmetry": round(float(np.std(asym_arr[mask], ddof=1)), 4) if mask.sum() > 1 else 0,
        }

    output = {
        "n_pairs": len(pno_values),
        "spearman_r": round(float(r_spearman), 4),
        "spearman_p": float(p_spearman),
        "pearson_r": round(float(r_pearson), 4),
        "pearson_p": float(p_pearson),
        "interpretation": (
            "Positive correlation: items with higher baseline P(no) show stronger negativity asymmetry, "
            "suggesting negative context reinforces a pre-existing tendency."
            if r_spearman > 0 and p_spearman < 0.05
            else "No significant correlation between baseline P(no) and asymmetry magnitude."
            if p_spearman >= 0.05
            else "Negative correlation: items with higher baseline P(no) show WEAKER negativity asymmetry."
        ),
        "quartile_analysis": binned,
        "mean_baseline_pno": round(float(np.mean(pno_arr)), 4),
        "mean_asymmetry": round(float(np.mean(asym_arr)), 4),
    }

    save_json(output, "results/asymmetry_baseline_corr.json")

    print(f"\nResults:")
    print(f"  N pairs: {len(pno_values)}")
    print(f"  Spearman r = {r_spearman:.4f}, p = {p_spearman:.2e}")
    print(f"  Pearson  r = {r_pearson:.4f}, p = {p_pearson:.2e}")
    print(f"  Mean baseline P(no) = {np.mean(pno_arr):.3f}")
    print(f"  Mean asymmetry = {np.mean(asym_arr):.3f}")
    print(f"\n  Quartile breakdown:")
    for label, data in binned.items():
        print(f"    {label}: n={data['n']}, mean_asym={data['mean_asymmetry']:.3f}")


if __name__ == "__main__":
    main()
