"""Comprehensive statistical analysis for the AMEL paper.

Produces results/paper_statistics.json with all statistics needed for the paper.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from analysis.utils import load_results, compute_bias_scores, N_COMPARISONS


def test_group(values, label=""):
    """One-sample t-test against 0 with effect size and CI."""
    if not values or len(values) < 2:
        return {"label": label, "n": len(values), "skip": True}
    arr = np.array(values)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    ci = 1.96 * std / np.sqrt(len(arr))
    d = mean / std if std > 0 else 0
    t, p = stats.ttest_1samp(arr, 0)
    p_corr = min(float(p) * N_COMPARISONS, 1.0)
    return {
        "label": label, "n": len(arr),
        "mean": round(mean, 6), "std": round(std, 6),
        "ci_lower": round(mean - ci, 6), "ci_upper": round(mean + ci, 6),
        "cohens_d": round(float(d), 4),
        "t": round(float(t), 4), "p_raw": float(p), "p_corrected": float(p_corr),
        "sig_raw": float(p) < 0.05, "sig_corrected": float(p_corr) < 0.05,
    }


def main():
    data_file = Path("data/all_results.jsonl")
    out_file = Path("results/paper_statistics.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {data_file}...")
    results = load_results(data_file)
    print(f"  Total results: {len(results)}")

    # ── 1. Data Summary ──
    print("\n1. Data summary...")
    n_total = len(results)
    n_parsed = sum(1 for r in results if r["parsed_response"] is not None)
    n_unparsed = n_total - n_parsed
    models = sorted(set(r["model"] for r in results))
    domains = sorted(set(r["domain"] for r in results))

    per_model = {m: sum(1 for r in results if r["model"] == m) for m in models}
    per_domain = {d: sum(1 for r in results if r["domain"] == d) for d in domains}
    per_polarity = defaultdict(int)
    for r in results:
        per_polarity[r["polarity"]] += 1

    summary = {
        "total_results": n_total,
        "parsed": n_parsed,
        "unparseable": n_unparsed,
        "unparseable_rate": round(n_unparsed / n_total, 4),
        "n_models": len(models),
        "models": models,
        "n_domains": len(domains),
        "per_model": per_model,
        "per_domain": dict(per_domain),
        "per_polarity": dict(per_polarity),
    }
    print(f"  Parsed: {n_parsed}, Unparseable: {n_unparsed} ({summary['unparseable_rate']*100:.1f}%)")

    # ── Compute bias scores ──
    print("\nComputing bias scores...")
    scores = compute_bias_scores(results)
    print(f"  {len(scores)} bias score observations")

    vals_all = [s["bias_score"] for s in scores]

    # ── 2. Overall Bias ──
    print("\n2. Overall bias...")
    overall = test_group(vals_all, "overall")
    print(f"  mean={overall['mean']:.4f}, d={overall['cohens_d']:.4f}, p={overall['p_raw']:.2e}")

    # ── 3. Per-Model ──
    print("\n3. Per-model bias...")
    per_model_stats = {}
    for m in models:
        vals = [s["bias_score"] for s in scores if s["model"] == m]
        per_model_stats[m] = test_group(vals, m)
        d = per_model_stats[m].get("cohens_d", "N/A")
        print(f"  {m}: n={len(vals)}, d={d}")

    # ── 4. Per-Polarity ──
    print("\n4. Per-polarity bias...")
    per_polarity_stats = {}
    for pol in ["no_saturated", "yes_saturated", "neutral"]:
        vals = [s["bias_score"] for s in scores if s["polarity"] == pol]
        per_polarity_stats[pol] = test_group(vals, pol)
        print(f"  {pol}: n={len(vals)}, mean={np.mean(vals):.4f}, d={per_polarity_stats[pol]['cohens_d']}")

    # ── 5. Per-Domain ──
    print("\n5. Per-domain bias...")
    per_domain_stats = {}
    for dom in domains:
        vals = [s["bias_score"] for s in scores if s["domain"] == dom]
        per_domain_stats[dom] = test_group(vals, dom)
        print(f"  {dom}: n={len(vals)}, d={per_domain_stats[dom]['cohens_d']}")

    # ── 6. Per-Category ──
    print("\n6. Per-category bias...")
    per_category_stats = {}
    for cat in ["clear_positive", "ambiguous", "clear_negative"]:
        vals = [s["bias_score"] for s in scores if s["category"] == cat]
        per_category_stats[cat] = test_group(vals, cat)
        print(f"  {cat}: n={len(vals)}, d={per_category_stats[cat]['cohens_d']}")

    # ── 7. Asymmetry Test ──
    print("\n7. Asymmetry test (|no_sat| vs |yes_sat|)...")
    # Match items across no_sat and yes_sat
    no_sat_map = {}
    yes_sat_map = {}
    for s in scores:
        key = f"{s['domain']}|{s['model']}|{s['item_id']}|{s['context_length']}"
        if s["polarity"] == "no_saturated":
            no_sat_map[key] = abs(s["bias_score"])
        elif s["polarity"] == "yes_saturated":
            yes_sat_map[key] = abs(s["bias_score"])

    common_keys = sorted(set(no_sat_map) & set(yes_sat_map))
    abs_no = [no_sat_map[k] for k in common_keys]
    abs_yes = [yes_sat_map[k] for k in common_keys]

    if len(common_keys) > 1:
        t_asym, p_asym = stats.ttest_rel(abs_no, abs_yes)
        asym_ratio = np.mean(abs_no) / np.mean(abs_yes) if np.mean(abs_yes) > 0 else float("inf")
        asymmetry = {
            "n_pairs": len(common_keys),
            "mean_abs_no_saturated": round(float(np.mean(abs_no)), 6),
            "mean_abs_yes_saturated": round(float(np.mean(abs_yes)), 6),
            "ratio": round(float(asym_ratio), 2),
            "t": round(float(t_asym), 4),
            "p": float(p_asym),
            "significant": float(p_asym) < 0.05,
            "direction": "no-bias stronger" if np.mean(abs_no) > np.mean(abs_yes) else "yes-bias stronger",
        }
    else:
        asymmetry = {"error": "insufficient paired data"}
    print(f"  Ratio: {asymmetry.get('ratio', 'N/A')}x, p={asymmetry.get('p', 'N/A')}")

    # ── 8. Accumulation Test ──
    print("\n8. Accumulation test...")
    accumulation = {}
    # Overall
    all_lens = [s["context_length"] for s in scores]
    all_bs = [s["bias_score"] for s in scores]
    r_sp, p_sp = stats.spearmanr(all_lens, all_bs)
    r_pe, p_pe = stats.pearsonr(all_lens, all_bs)
    accumulation["overall"] = {
        "spearman_r": round(float(r_sp), 6), "spearman_p": float(p_sp),
        "pearson_r": round(float(r_pe), 6), "pearson_p": float(p_pe),
        "significant": float(p_sp) < 0.05,
    }
    # Per polarity
    for pol in ["no_saturated", "yes_saturated", "neutral"]:
        subset = [(s["context_length"], s["bias_score"]) for s in scores if s["polarity"] == pol]
        if len(subset) > 2:
            lens, bs = zip(*subset)
            r, p = stats.spearmanr(lens, bs)
            accumulation[pol] = {"spearman_r": round(float(r), 6), "spearman_p": float(p), "significant": float(p) < 0.05}
    # Per category
    for cat in ["clear_positive", "ambiguous", "clear_negative"]:
        subset = [(s["context_length"], s["bias_score"]) for s in scores if s["category"] == cat]
        if len(subset) > 2:
            lens, bs = zip(*subset)
            r, p = stats.spearmanr(lens, bs)
            accumulation[cat] = {"spearman_r": round(float(r), 6), "spearman_p": float(p), "significant": float(p) < 0.05}
    # Per context length means
    for cl in [5, 10, 20, 50]:
        vals = [s["bias_score"] for s in scores if s["context_length"] == cl]
        accumulation[f"ctx_{cl}"] = {"mean": round(float(np.mean(vals)), 6), "std": round(float(np.std(vals)), 6), "n": len(vals)}
    print(f"  Overall: r={accumulation['overall']['spearman_r']:.4f}, p={accumulation['overall']['spearman_p']:.4f}")

    # ── 9. Per-Model × Per-Polarity ──
    print("\n9. Model x polarity...")
    model_polarity = {}
    for m in models:
        model_polarity[m] = {}
        for pol in ["no_saturated", "yes_saturated", "neutral"]:
            vals = [s["bias_score"] for s in scores if s["model"] == m and s["polarity"] == pol]
            if vals:
                model_polarity[m][pol] = {
                    "n": len(vals),
                    "mean": round(float(np.mean(vals)), 6),
                    "std": round(float(np.std(vals)), 6),
                }
    print(f"  Computed for {len(models)} models x 3 polarities")

    # ── 10. McNemar's Test ──
    print("\n10. McNemar's test (baseline vs treatment response concordance)...")
    mcnemar_results = {}
    for m in models:
        # For each model, count items that flip under no_saturated
        flips = {"yes_to_no": 0, "no_to_yes": 0, "same_yes": 0, "same_no": 0}
        for s in scores:
            if s["model"] == m and s["polarity"] == "no_saturated":
                # bl_rate and tx_rate are P(no)
                bl_majority = "no" if s["bl_rate"] > 0.5 else "yes"
                tx_majority = "no" if s["tx_rate"] > 0.5 else "yes"
                if bl_majority == "yes" and tx_majority == "no":
                    flips["yes_to_no"] += 1
                elif bl_majority == "no" and tx_majority == "yes":
                    flips["no_to_yes"] += 1
                elif bl_majority == "yes":
                    flips["same_yes"] += 1
                else:
                    flips["same_no"] += 1
        b = flips["yes_to_no"]
        c = flips["no_to_yes"]
        n_discordant = b + c
        if n_discordant > 0:
            # McNemar's chi-squared
            chi2 = (b - c) ** 2 / (b + c)
            p_mcn = float(stats.chi2.sf(chi2, 1))
        else:
            chi2 = 0.0
            p_mcn = 1.0
        mcnemar_results[m] = {
            "yes_to_no": b, "no_to_yes": c,
            "same_yes": flips["same_yes"], "same_no": flips["same_no"],
            "chi2": round(chi2, 4), "p": p_mcn,
            "significant": p_mcn < 0.05,
        }
    print(f"  Computed for {len(models)} models")

    # ── 11. Cross-tabulations ──
    print("\n11. Cross-tabulations...")
    # Model x Category
    model_category = {}
    for m in models:
        model_category[m] = {}
        for cat in ["clear_positive", "ambiguous", "clear_negative"]:
            vals = [s["bias_score"] for s in scores if s["model"] == m and s["category"] == cat]
            if vals:
                model_category[m][cat] = {"n": len(vals), "mean": round(float(np.mean(vals)), 6)}

    # Model x Domain
    model_domain = {}
    for m in models:
        model_domain[m] = {}
        for dom in domains:
            vals = [s["bias_score"] for s in scores if s["model"] == m and s["domain"] == dom]
            if vals:
                model_domain[m][dom] = {"n": len(vals), "mean": round(float(np.mean(vals)), 6)}

    # Domain x Category
    domain_category = {}
    for dom in domains:
        domain_category[dom] = {}
        for cat in ["clear_positive", "ambiguous", "clear_negative"]:
            vals = [s["bias_score"] for s in scores if s["domain"] == dom and s["category"] == cat]
            if vals:
                domain_category[dom][cat] = {"n": len(vals), "mean": round(float(np.mean(vals)), 6)}
    print("  Done")

    # ── 12. Flip Rates ──
    print("\n12. Flip rates (clear positive items flipping to 'no' under no-saturated)...")
    flip_rates = {}
    for m in models:
        # For clear_positive items under no_saturated: how often does majority response flip?
        cp_scores = [s for s in scores if s["model"] == m and s["category"] == "clear_positive" and s["polarity"] == "no_saturated"]
        n_items = len(cp_scores)
        n_flipped = sum(1 for s in cp_scores if s["tx_rate"] > 0.5)  # majority became target response
        flip_rates[m] = {
            "n_items": n_items,
            "n_flipped": n_flipped,
            "flip_rate": round(n_flipped / n_items, 4) if n_items > 0 else 0,
        }
        print(f"  {m}: {n_flipped}/{n_items} ({flip_rates[m]['flip_rate']*100:.1f}%)")

    # ── 13. Baseline Consistency ──
    print("\n13. Baseline consistency (inter-repetition agreement)...")
    baseline_results = [r for r in results if r["polarity"] == "baseline" and r["parsed_response"] is not None]
    baseline_groups = defaultdict(list)
    for r in baseline_results:
        key = f"{r['domain']}|{r['model']}|{r['test_item_id']}"
        baseline_groups[key].append(r["parsed_response"])

    agreements = []
    for key, responses in baseline_groups.items():
        if len(responses) < 2:
            continue
        most_common = max(set(responses), key=responses.count)
        agreement = responses.count(most_common) / len(responses)
        agreements.append(agreement)

    baseline_consistency = {
        "n_groups": len(agreements),
        "mean_agreement": round(float(np.mean(agreements)), 4),
        "median_agreement": round(float(np.median(agreements)), 4),
        "min_agreement": round(float(np.min(agreements)), 4),
        "std_agreement": round(float(np.std(agreements)), 4),
    }
    print(f"  Mean agreement: {baseline_consistency['mean_agreement']*100:.1f}%")

    # ── 14. Excluding Contrarian Models ──
    print("\n14. Excluding contrarian model (qwen3:4b)...")
    vals_excl = [s["bias_score"] for s in scores if s["model"] != "qwen3:4b"]
    excl_stats = test_group(vals_excl, "excluding_qwen3_4b")
    print(f"  mean={excl_stats['mean']:.4f}, d={excl_stats['cohens_d']:.4f}")

    # ── Assemble Output ──
    output = {
        "data_summary": summary,
        "overall": overall,
        "per_model": per_model_stats,
        "per_polarity": per_polarity_stats,
        "per_domain": per_domain_stats,
        "per_category": per_category_stats,
        "asymmetry_test": asymmetry,
        "accumulation": accumulation,
        "model_x_polarity": model_polarity,
        "mcnemar": mcnemar_results,
        "cross_tabs": {
            "model_x_category": model_category,
            "model_x_domain": model_domain,
            "domain_x_category": domain_category,
        },
        "flip_rates": flip_rates,
        "baseline_consistency": baseline_consistency,
        "excluding_contrarian": excl_stats,
        "_meta": {
            "n_comparisons_bonferroni": N_COMPARISONS,
            "alpha": 0.05,
            "total_bias_scores": len(scores),
        },
    }

    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {out_file}")
    print(f"Total bias score observations: {len(scores)}")


if __name__ == "__main__":
    main()
