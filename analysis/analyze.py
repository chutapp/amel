"""Statistical analysis for the AMEL experiment."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_results(results_file: Path) -> list[dict]:
    """Load results from JSONL file."""
    results = []
    with open(results_file) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_bias_scores(results: list[dict]) -> dict:
    """Compute bias scores comparing treatment vs baseline.

    Bias Score = P(response=dominant_polarity | biased_context) - P(response=dominant_polarity | baseline)
    """
    # Group responses
    groups: dict[str, list] = defaultdict(list)

    for r in results:
        if r["parsed_response"] is None:
            continue

        # Key for grouping: domain, model, test_item_id
        base_key = f"{r['domain']}|{r['model']}|{r['test_item_id']}"

        if r["polarity"] == "baseline":
            groups[f"{base_key}|baseline"].append(r)
        else:
            key = f"{base_key}|{r['polarity']}|{r['context_length']}"
            groups[key].append(r)

    bias_scores = []

    for key, group in groups.items():
        parts = key.split("|")
        if parts[-1] == "baseline":
            continue

        domain, model, test_item_id, polarity, ctx_len = parts
        baseline_key = f"{domain}|{model}|{test_item_id}|baseline"
        baseline_group = groups.get(baseline_key, [])

        if not baseline_group or not group:
            continue

        # For no_saturated: measure P("no")
        # For yes_saturated: measure P("yes")
        target_response = "no" if polarity == "no_saturated" else "yes"
        if polarity == "neutral":
            target_response = "no"  # compare neutral to baseline for "no" rate

        baseline_rate = sum(
            1 for r in baseline_group if r["parsed_response"] == target_response
        ) / len(baseline_group)

        treatment_rate = sum(
            1 for r in group if r["parsed_response"] == target_response
        ) / len(group)

        bias_score = treatment_rate - baseline_rate

        # Get test item info from first result
        first = group[0]

        bias_scores.append({
            "domain": domain,
            "model": model,
            "polarity": polarity,
            "context_length": int(ctx_len),
            "test_item_id": test_item_id,
            "test_item_category": first["test_item_category"],
            "test_item_ground_truth": first["test_item_ground_truth"],
            "target_response": target_response,
            "baseline_rate": baseline_rate,
            "treatment_rate": treatment_rate,
            "bias_score": bias_score,
            "baseline_n": len(baseline_group),
            "treatment_n": len(group),
        })

    return bias_scores


def _cohens_h(p1: float, p2: float) -> float:
    """Compute Cohen's h effect size for two proportions."""
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def _ci_95(data: list[float]) -> tuple[float, float]:
    """Compute 95% confidence interval for the mean."""
    n = len(data)
    if n < 2:
        return (float(np.mean(data)), float(np.mean(data)))
    mean = np.mean(data)
    se = stats.sem(data)
    ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    return (float(ci[0]), float(ci[1]))


def _test_group(label: str, values: list[float], n_comparisons: int = 1) -> dict:
    """Run one-sample t-test with effect size and CI, Bonferroni-corrected."""
    if not values:
        return {}
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    t_stat, p_value = stats.ttest_1samp(values, 0) if len(values) > 1 else (0.0, 1.0)
    p_corrected = min(float(p_value) * n_comparisons, 1.0)  # Bonferroni
    ci = _ci_95(values)
    # Cohen's d (effect size for one-sample t-test)
    cohens_d = mean / std if std > 0 else 0.0
    return {
        "mean_bias_score": mean,
        "std_bias_score": std,
        "ci_95_lower": ci[0],
        "ci_95_upper": ci[1],
        "t_statistic": float(t_stat),
        "p_value_raw": float(p_value),
        "p_value_corrected": p_corrected,
        "cohens_d": float(cohens_d),
        "n": len(values),
        "significant_raw": float(p_value) < 0.05,
        "significant_corrected": p_corrected < 0.05,
    }


def statistical_tests(bias_scores: list[dict]) -> dict:
    """Run statistical tests on bias scores with Bonferroni correction."""
    results = {}

    # Count total number of independent tests for Bonferroni
    polarities = sorted(set(b["polarity"] for b in bias_scores))
    domains = sorted(set(b["domain"] for b in bias_scores))
    models = sorted(set(b["model"] for b in bias_scores))
    categories = ["clear_positive", "ambiguous", "clear_negative"]
    n_tests = 1 + len(polarities) + len(domains) + len(models) + len(categories)

    # 1. Overall
    all_bs = [b["bias_score"] for b in bias_scores]
    results["overall"] = _test_group("overall", all_bs, n_tests)

    # 2. By polarity
    for polarity in polarities:
        pol_bs = [b["bias_score"] for b in bias_scores if b["polarity"] == polarity]
        results[f"polarity_{polarity}"] = _test_group(f"polarity_{polarity}", pol_bs, n_tests)

    # 3. By domain
    for domain in domains:
        dom_bs = [b["bias_score"] for b in bias_scores if b["domain"] == domain]
        results[f"domain_{domain}"] = _test_group(f"domain_{domain}", dom_bs, n_tests)

    # 4. By model
    for model in models:
        mod_bs = [b["bias_score"] for b in bias_scores if b["model"] == model]
        results[f"model_{model}"] = _test_group(f"model_{model}", mod_bs, n_tests)

    # 5. By test item category
    for cat in categories:
        cat_bs = [b["bias_score"] for b in bias_scores if b["test_item_category"] == cat]
        if cat_bs:
            results[f"category_{cat}"] = _test_group(f"category_{cat}", cat_bs, n_tests)

    # 6. Correlation: bias score vs context length (Spearman for robustness)
    lengths = [b["context_length"] for b in bias_scores]
    scores = [b["bias_score"] for b in bias_scores]
    if lengths and scores:
        pearson_r, pearson_p = stats.pearsonr(lengths, scores)
        spearman_r, spearman_p = stats.spearmanr(lengths, scores)
        results["context_length_correlation"] = {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "significant": float(pearson_p) < 0.05,
        }

    # 7. Asymmetry test: is no-saturated bias different from yes-saturated?
    no_bs = [b["bias_score"] for b in bias_scores if b["polarity"] == "no_saturated"]
    yes_bs = [b["bias_score"] for b in bias_scores if b["polarity"] == "yes_saturated"]
    if no_bs and yes_bs:
        # Use absolute values to compare magnitude
        t_stat, p_value = stats.ttest_ind(
            [abs(b) for b in no_bs],
            [abs(b) for b in yes_bs],
            equal_var=False,  # Welch's t-test
        )
        results["asymmetry_test"] = {
            "mean_abs_no_saturated": float(np.mean([abs(b) for b in no_bs])),
            "mean_abs_yes_saturated": float(np.mean([abs(b) for b in yes_bs])),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": float(p_value) < 0.05,
            "direction": "no-bias stronger" if np.mean([abs(b) for b in no_bs]) > np.mean([abs(b) for b in yes_bs]) else "yes-bias stronger",
        }

    # 8. Interaction: category x context_length (is ambiguous more affected at longer contexts?)
    for cat in categories:
        cat_scores = [b for b in bias_scores if b["test_item_category"] == cat and b["polarity"] == "no_saturated"]
        if cat_scores:
            lengths_cat = [b["context_length"] for b in cat_scores]
            scores_cat = [b["bias_score"] for b in cat_scores]
            if len(set(lengths_cat)) > 1:
                r, p = stats.spearmanr(lengths_cat, scores_cat)
                results[f"accumulation_{cat}"] = {
                    "spearman_r": float(r),
                    "p_value": float(p),
                    "significant": float(p) < 0.05,
                }

    results["_meta"] = {
        "n_comparisons_bonferroni": n_tests,
        "alpha": 0.05,
        "total_bias_scores": len(bias_scores),
    }

    return results


def compute_flip_rate(results: list[dict]) -> dict:
    """Compute flip rate: how often clear-positive items flip to "no" under bias."""
    flips = defaultdict(lambda: {"flipped": 0, "total": 0})

    # Get baseline rates per item
    baseline_responses: dict[str, list] = defaultdict(list)
    for r in results:
        if r["polarity"] == "baseline" and r["parsed_response"]:
            key = f"{r['domain']}|{r['model']}|{r['test_item_id']}"
            baseline_responses[key].append(r["parsed_response"])

    # Check treatment responses
    for r in results:
        if r["polarity"] == "baseline" or not r["parsed_response"]:
            continue
        if r["test_item_category"] != "clear_positive":
            continue

        key = f"{r['domain']}|{r['model']}|{r['test_item_id']}"
        baseline = baseline_responses.get(key, [])
        if not baseline:
            continue

        # Baseline majority is "yes" for clear_positive items
        baseline_majority = max(set(baseline), key=baseline.count)
        if baseline_majority != "yes":
            continue

        group_key = f"{r['domain']}|{r['model']}|{r['polarity']}|{r['context_length']}"
        flips[group_key]["total"] += 1
        if r["parsed_response"] == "no":
            flips[group_key]["flipped"] += 1

    return {
        k: {**v, "flip_rate": v["flipped"] / v["total"] if v["total"] > 0 else 0}
        for k, v in flips.items()
    }


def plot_bias_curve(bias_scores: list[dict], output_dir: Path) -> None:
    """Plot bias score vs context length (accumulation curve)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # By polarity
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, polarity in enumerate(["no_saturated", "yes_saturated"]):
        ax = axes[idx]
        pol_data = [b for b in bias_scores if b["polarity"] == polarity]

        # Group by context length
        by_length: dict[int, list[float]] = defaultdict(list)
        for b in pol_data:
            by_length[b["context_length"]].append(b["bias_score"])

        lengths = sorted(by_length.keys())
        means = [np.mean(by_length[l]) for l in lengths]
        stds = [np.std(by_length[l]) for l in lengths]
        sems = [s / np.sqrt(len(by_length[l])) for s, l in zip(stds, lengths)]

        ax.errorbar(lengths, means, yerr=[1.96 * s for s in sems], fmt="o-", capsize=5)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Context Length (# prior turns)")
        ax.set_ylabel("Bias Score")
        ax.set_title(f"Bias Accumulation — {polarity.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "bias_accumulation_curve.png", dpi=150)
    plt.close()


def plot_model_comparison(bias_scores: list[dict], output_dir: Path) -> None:
    """Plot bias scores by model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to no_saturated for cleaner comparison
    no_sat = [b for b in bias_scores if b["polarity"] == "no_saturated"]

    by_model: dict[str, list[float]] = defaultdict(list)
    for b in no_sat:
        by_model[b["model"]].append(b["bias_score"])

    models = sorted(by_model.keys())
    means = [np.mean(by_model[m]) for m in models]
    stds = [np.std(by_model[m]) for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([m.split(":")[0] for m in models], rotation=45, ha="right")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Mean Bias Score (No-Saturated Context)")
    ax.set_title("Model Susceptibility to Context Bias")
    ax.grid(True, alpha=0.3, axis="y")

    # Color bars
    for bar, mean in zip(bars, means):
        bar.set_color("salmon" if mean > 0 else "lightblue")

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150)
    plt.close()


def plot_domain_comparison(bias_scores: list[dict], output_dir: Path) -> None:
    """Plot bias scores by domain."""
    output_dir.mkdir(parents=True, exist_ok=True)

    no_sat = [b for b in bias_scores if b["polarity"] == "no_saturated"]

    by_domain: dict[str, list[float]] = defaultdict(list)
    for b in no_sat:
        by_domain[b["domain"]].append(b["bias_score"])

    domains = sorted(by_domain.keys())
    means = [np.mean(by_domain[d]) for d in domains]
    stds = [np.std(by_domain[d]) for d in domains]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = range(len(domains))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", " ").title() for d in domains])
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Mean Bias Score (No-Saturated Context)")
    ax.set_title("Domain Sensitivity to Context Bias")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, mean in zip(bars, means):
        bar.set_color("salmon" if mean > 0 else "lightblue")

    plt.tight_layout()
    plt.savefig(output_dir / "domain_comparison.png", dpi=150)
    plt.close()


def plot_category_comparison(bias_scores: list[dict], output_dir: Path) -> None:
    """Plot bias scores by test item category."""
    output_dir.mkdir(parents=True, exist_ok=True)

    no_sat = [b for b in bias_scores if b["polarity"] == "no_saturated"]

    by_cat: dict[str, list[float]] = defaultdict(list)
    for b in no_sat:
        by_cat[b["test_item_category"]].append(b["bias_score"])

    categories = ["clear_positive", "ambiguous", "clear_negative"]
    categories = [c for c in categories if c in by_cat]
    means = [np.mean(by_cat[c]) for c in categories]
    stds = [np.std(by_cat[c]) for c in categories]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = range(len(categories))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=["green", "orange", "red"])
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ").title() for c in categories])
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Mean Bias Score (No-Saturated Context)")
    ax.set_title("Bias by Test Item Ambiguity")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "category_comparison.png", dpi=150)
    plt.close()


def plot_heatmap(bias_scores: list[dict], output_dir: Path) -> None:
    """Plot heatmap of bias scores: model x context_length."""
    output_dir.mkdir(parents=True, exist_ok=True)

    no_sat = [b for b in bias_scores if b["polarity"] == "no_saturated"]

    models = sorted(set(b["model"] for b in no_sat))
    lengths = sorted(set(b["context_length"] for b in no_sat))

    matrix = np.zeros((len(models), len(lengths)))
    for b in no_sat:
        i = models.index(b["model"])
        j = lengths.index(b["context_length"])
        # Accumulate for averaging
        if matrix[i, j] == 0:
            matrix[i, j] = b["bias_score"]
        else:
            matrix[i, j] = (matrix[i, j] + b["bias_score"]) / 2

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap="RdYlBu_r", aspect="auto", vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(lengths)))
    ax.set_xticklabels(lengths)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.split(":")[0] for m in models])
    ax.set_xlabel("Context Length")
    ax.set_ylabel("Model")
    ax.set_title("Bias Score Heatmap (No-Saturated Context)")

    # Add values
    for i in range(len(models)):
        for j in range(len(lengths)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)

    plt.colorbar(im, label="Bias Score")
    plt.tight_layout()
    plt.savefig(output_dir / "bias_heatmap.png", dpi=150)
    plt.close()


def generate_report(results_file: Path, output_dir: Path) -> None:
    """Generate full analysis report."""
    print(f"Loading results from {results_file}...")
    results = load_results(results_file)
    print(f"Loaded {len(results)} results")

    # Filter out unparseable
    parseable = [r for r in results if r["parsed_response"] is not None]
    unparseable = len(results) - len(parseable)
    print(f"Parseable: {len(parseable)}, Unparseable: {unparseable} ({unparseable/len(results)*100:.1f}%)")

    # Compute bias scores
    print("\nComputing bias scores...")
    bias_scores = compute_bias_scores(results)
    print(f"Computed {len(bias_scores)} bias score comparisons")

    # Statistical tests
    print("\nRunning statistical tests...")
    test_results = statistical_tests(bias_scores)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    meta = test_results.get("_meta", {})
    print(f"Bonferroni correction: {meta.get('n_comparisons_bonferroni', '?')} comparisons, α=0.05")
    print()

    for test_name, test_result in sorted(test_results.items()):
        if test_name.startswith("_"):
            continue
        if "mean_bias_score" in test_result:
            sig_raw = "*" if test_result.get("significant_raw") else " "
            sig_corr = "†" if test_result.get("significant_corrected") else " "
            print(
                f" {sig_raw}{sig_corr} {test_name:40s} "
                f"BS={test_result['mean_bias_score']:+.4f} "
                f"CI=[{test_result['ci_95_lower']:+.4f}, {test_result['ci_95_upper']:+.4f}] "
                f"d={test_result['cohens_d']:+.3f} "
                f"p={test_result['p_value_raw']:.4f} "
                f"(corrected: {test_result['p_value_corrected']:.4f}) "
                f"n={test_result['n']}"
            )
        elif "pearson_r" in test_result and "spearman_r" in test_result:
            sig = "*" if test_result.get("significant") else " "
            print(
                f" {sig}  {test_name:40s} "
                f"r_pearson={test_result['pearson_r']:+.4f} (p={test_result['pearson_p']:.4f}) "
                f"r_spearman={test_result['spearman_r']:+.4f} (p={test_result['spearman_p']:.4f})"
            )
        elif "pearson_r" in test_result:
            sig = "*" if test_result.get("significant") else " "
            print(
                f" {sig}  {test_name:40s} "
                f"r={test_result['pearson_r']:+.4f} (p={test_result.get('p_value', test_result.get('pearson_p', 0)):.4f})"
            )
        elif "spearman_r" in test_result:
            sig = "*" if test_result.get("significant") else " "
            print(
                f" {sig}  {test_name:40s} "
                f"r_spearman={test_result['spearman_r']:+.4f} (p={test_result['p_value']:.4f})"
            )
        elif "direction" in test_result:
            sig = "*" if test_result.get("significant") else " "
            print(
                f" {sig}  {test_name:40s} "
                f"|no|={test_result['mean_abs_no_saturated']:.4f} vs |yes|={test_result['mean_abs_yes_saturated']:.4f} "
                f"p={test_result['p_value']:.4f} ({test_result['direction']})"
            )

    print()
    print("Legend: * = p<0.05 (raw), † = p<0.05 (Bonferroni-corrected)")

    # Flip rates
    print("\n" + "=" * 60)
    print("FLIP RATES (Clear Positive items flipping to 'No')")
    print("=" * 60)
    flip_rates = compute_flip_rate(results)
    for key, data in sorted(flip_rates.items()):
        if data["total"] > 0:
            print(f"  {key:60s} {data['flipped']}/{data['total']} = {data['flip_rate']:.2%}")

    # Generate plots
    print("\nGenerating plots...")
    figures_dir = output_dir / "figures"

    if bias_scores:
        plot_bias_curve(bias_scores, figures_dir)
        plot_model_comparison(bias_scores, figures_dir)
        plot_domain_comparison(bias_scores, figures_dir)
        plot_category_comparison(bias_scores, figures_dir)
        plot_heatmap(bias_scores, figures_dir)
        print(f"Plots saved to {figures_dir}")

    # Save detailed results
    output_dir.mkdir(parents=True, exist_ok=True)

    def _serialize(obj):
        """Handle numpy types for JSON serialization."""
        if hasattr(obj, "item"):  # numpy scalars
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    stats_file = output_dir / "statistical_tests.json"
    with open(stats_file, "w") as f:
        json.dump(test_results, f, indent=2, default=_serialize)
    print(f"Statistical tests saved to {stats_file}")

    bs_file = output_dir / "bias_scores.json"
    with open(bs_file, "w") as f:
        json.dump(bias_scores, f, indent=2, default=_serialize)
    print(f"Bias scores saved to {bs_file}")


if __name__ == "__main__":
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/results.jsonl")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
    generate_report(results_path, output_path)
