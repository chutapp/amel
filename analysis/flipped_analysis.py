"""Flipped framing analysis (Phase 2).

Compares asymmetry ratios between original and flipped framing.
- If asymmetry flips (yes_sat now stronger because yes=rejection): RLHF hypothesis
- If asymmetry stays (no_sat still stronger): token hypothesis
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from analysis.utils import load_results as load_main_results, save_json


def load_flipped_results(path=None):
    """Load flipped experiment results."""
    path = path or Path("data/flipped/results.jsonl")
    if not path.exists():
        print(f"WARNING: {path} not found. Run run_flipped.py first.")
        return []
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_asymmetry(results):
    """Compute |BS(no_sat)| and |BS(yes_sat)| per model.

    Returns dict of model → {no_sat_abs_bs, yes_sat_abs_bs, ratio, ...}
    """
    # Group baselines
    baselines = defaultdict(list)
    treatments = defaultdict(list)

    for r in results:
        if r["parsed_response"] is None:
            continue
        key = f"{r['model']}|{r['test_item_id']}"
        if r["polarity"] == "baseline":
            baselines[key].append(r)
        else:
            treatments[f"{key}|{r['polarity']}"].append(r)

    # Compute baseline P(no) per item × model
    baseline_pno = {}
    for key, group in baselines.items():
        baseline_pno[key] = sum(1 for r in group if r["parsed_response"] == "no") / len(group)

    # Compute BS per polarity
    model_bs = defaultdict(lambda: defaultdict(list))
    for key, group in treatments.items():
        parts = key.split("|")
        model, item_id, polarity = parts[0], parts[1], parts[2]
        base_key = f"{model}|{item_id}"
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

        model_bs[model][polarity].append(abs(bs))

    # Compute per-model asymmetry
    model_asymmetry = {}
    for model, pol_scores in model_bs.items():
        no_abs = np.array(pol_scores.get("no_saturated", []))
        yes_abs = np.array(pol_scores.get("yes_saturated", []))

        if len(no_abs) == 0 or len(yes_abs) == 0:
            continue

        # Paired test if same number of items
        if len(no_abs) == len(yes_abs):
            t_stat, p_val = stats.ttest_rel(no_abs, yes_abs)
        else:
            t_stat, p_val = stats.ttest_ind(no_abs, yes_abs)

        ratio = float(np.mean(no_abs) / np.mean(yes_abs)) if np.mean(yes_abs) > 0 else float("inf")

        model_asymmetry[model] = {
            "mean_abs_bs_no_sat": round(float(np.mean(no_abs)), 4),
            "mean_abs_bs_yes_sat": round(float(np.mean(yes_abs)), 4),
            "ratio_no_over_yes": round(ratio, 2),
            "n_no_sat": len(no_abs),
            "n_yes_sat": len(yes_abs),
            "t_stat": round(float(t_stat), 4),
            "p_value": float(p_val),
            "stronger_polarity": "no_saturated" if np.mean(no_abs) > np.mean(yes_abs) else "yes_saturated",
        }

    return model_asymmetry


def main():
    print("Loading flipped experiment results...")
    flipped_results = load_flipped_results()
    if not flipped_results:
        print("No flipped data found. Run run_flipped.py first.")
        save_json({"status": "awaiting_data"}, "results/flipped_analysis.json")
        return

    print(f"  Loaded {len(flipped_results)} flipped results")

    # Compute flipped asymmetry
    flipped_asymmetry = compute_asymmetry(flipped_results)

    # Load and compute original asymmetry for the same models
    print("Loading main experiment for comparison...")
    main_results = load_main_results()

    # Filter to same models and code_review domain
    flipped_models = set(flipped_asymmetry.keys())
    original_code_review = [
        r for r in main_results
        if r["model"] in flipped_models and r["domain"] == "code_review"
    ]

    original_asymmetry = compute_asymmetry(original_code_review)

    # Compare
    comparison = {}
    for model in flipped_models:
        orig = original_asymmetry.get(model, {})
        flip = flipped_asymmetry.get(model, {})

        if not orig or not flip:
            continue

        orig_stronger = orig.get("stronger_polarity", "unknown")
        flip_stronger = flip.get("stronger_polarity", "unknown")

        # Key question: did the asymmetry flip?
        # In original: no_saturated context → model says more "no" (rejection)
        # In flipped:  no_saturated context → model says more "no" (approval!)
        #
        # If RLHF hypothesis (rejection is sticky):
        #   Original: no_sat stronger (rejection context → more rejection)
        #   Flipped:  yes_sat stronger (yes=rejection context → more rejection)
        #   → The stronger polarity FLIPS
        #
        # If token hypothesis ("no" token is sticky):
        #   Original: no_sat stronger ("no" token → more "no")
        #   Flipped:  no_sat still stronger ("no" token → more "no")
        #   → The stronger polarity STAYS

        asymmetry_flipped = (orig_stronger != flip_stronger)

        comparison[model] = {
            "original_stronger": orig_stronger,
            "original_ratio": orig.get("ratio_no_over_yes", 0),
            "flipped_stronger": flip_stronger,
            "flipped_ratio": flip.get("ratio_no_over_yes", 0),
            "asymmetry_flipped": asymmetry_flipped,
            "hypothesis_supported": "RLHF (rejection frame)" if asymmetry_flipped else "Token ('no' is stickier)",
        }

    # Overall verdict
    n_flipped = sum(1 for v in comparison.values() if v["asymmetry_flipped"])
    n_stayed = sum(1 for v in comparison.values() if not v["asymmetry_flipped"])

    output = {
        "flipped_asymmetry": flipped_asymmetry,
        "original_asymmetry": original_asymmetry,
        "comparison": comparison,
        "verdict": {
            "n_models_tested": len(comparison),
            "n_asymmetry_flipped": n_flipped,
            "n_asymmetry_stayed": n_stayed,
            "conclusion": (
                "RLHF hypothesis: negativity follows the rejection semantic frame, not the 'no' token"
                if n_flipped > n_stayed
                else "Token hypothesis: the 'no' token is inherently stickier regardless of semantic frame"
                if n_stayed > n_flipped
                else "Inconclusive: mixed results across models"
            ),
        },
    }

    save_json(output, "results/flipped_analysis.json")

    print("\nComparison:")
    for model, data in comparison.items():
        print(f"  {model}:")
        print(f"    Original: {data['original_stronger']} stronger (ratio={data['original_ratio']})")
        print(f"    Flipped:  {data['flipped_stronger']} stronger (ratio={data['flipped_ratio']})")
        print(f"    → {data['hypothesis_supported']}")
    print(f"\nVerdict: {output['verdict']['conclusion']}")


if __name__ == "__main__":
    main()
