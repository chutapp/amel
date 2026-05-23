"""Generate publication-quality figures for the AMEL paper."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from scipy import stats

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

# ── Colorblind-safe palette (Paul Tol bright) ─────────────────────────────
# https://personal.sron.nl/~pault/data/colourschemes.pdf
PROVIDER_COLORS = {
    "OpenAI": "#4477AA",      # blue
    "Anthropic": "#EE6677",   # rose
    "Google": "#228833",      # green
    "Local (OSS)": "#BBBBBB", # grey
}

# Polarity colors — used consistently across all polarity figures
POL_COLORS = {
    "no_saturated": "#4477AA",   # blue
    "neutral": "#BBBBBB",       # grey
    "yes_saturated": "#EE7733", # orange
}

# Category colors
CAT_COLORS = {
    "clear_positive": "#66CCEE",  # cyan
    "ambiguous": "#CCBB44",       # yellow
    "clear_negative": "#AA3377",  # purple
}

# Domain colors
DOM_COLORS = ["#4477AA", "#EE6677", "#228833"]

MODEL_META = {
    "gpt-4.1-nano":                 {"provider": "OpenAI",    "short": "GPT-4.1\nnano",    "order": 0, "tier": "small"},
    "gpt-5.2":                      {"provider": "OpenAI",    "short": "GPT-5.2",          "order": 1, "tier": "flagship"},
    "claude-haiku-4-5-20251001":    {"provider": "Anthropic", "short": "Haiku\n4.5",       "order": 2, "tier": "small"},
    "claude-sonnet-4-6":            {"provider": "Anthropic", "short": "Sonnet\n4.6",      "order": 3, "tier": "mid"},
    "claude-opus-4-6":              {"provider": "Anthropic", "short": "Opus\n4.6",        "order": 4, "tier": "flagship"},
    "gemini-2.5-flash":             {"provider": "Google",    "short": "Gemini\nFlash",    "order": 5, "tier": "small"},
    "gemini-2.5-pro":               {"provider": "Google",    "short": "Gemini\nPro",      "order": 6, "tier": "flagship"},
    "llama3.2:3b":                  {"provider": "Local (OSS)", "short": "Llama3.2\n3B",   "order": 7, "tier": "small"},
    "qwen3:4b":                     {"provider": "Local (OSS)", "short": "Qwen3\n4B",      "order": 8, "tier": "small"},
    "qwen3.5:4b":                   {"provider": "Local (OSS)", "short": "Qwen3.5\n4B",   "order": 9, "tier": "small"},
    "qwen3:30b":                    {"provider": "Local (OSS)", "short": "Qwen3\n30B",    "order": 10, "tier": "large"},
}

def get_color(model):
    meta = MODEL_META.get(model, {"provider": "Local (OSS)"})
    return PROVIDER_COLORS[meta["provider"]]

def get_short(model):
    return MODEL_META.get(model, {"short": model})["short"]


# ── Data Loading ───────────────────────────────────────────────────────────
def load_results(path):
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_bias_scores(results):
    """Compute per-item bias scores.

    For no_saturated/neutral: BS = P(no|treatment) - P(no|baseline)
    For yes_saturated: BS = P(yes|treatment) - P(yes|baseline)

    This matches the analysis script's convention: positive BS always means
    the model shifted toward the saturated polarity (conforming bias).
    """
    groups = defaultdict(list)
    for r in results:
        if r["parsed_response"] is None:
            continue
        base_key = f"{r['domain']}|{r['model']}|{r['test_item_id']}"
        if r["polarity"] == "baseline":
            groups[f"{base_key}|baseline"].append(r)
        else:
            key = f"{base_key}|{r['polarity']}|{r['context_length']}"
            groups[key].append(r)

    scores = []
    for key, group in groups.items():
        parts = key.split("|")
        if parts[-1] == "baseline":
            continue
        domain, model, item_id, polarity, ctx_len = parts
        baseline_key = f"{domain}|{model}|{item_id}|baseline"
        baseline = groups.get(baseline_key, [])
        if not baseline:
            continue

        # Match analysis script: flip target for yes_saturated
        target = "no" if polarity in ("no_saturated", "neutral") else "yes"

        bl_rate = sum(1 for r in baseline if r["parsed_response"] == target) / len(baseline)
        tx_rate = sum(1 for r in group if r["parsed_response"] == target) / len(group)
        bs = tx_rate - bl_rate

        category = group[0].get("test_item_category", "unknown")
        scores.append({
            "domain": domain, "model": model, "item_id": item_id,
            "polarity": polarity, "context_length": int(ctx_len),
            "category": category, "bias_score": bs,
            "bl_no_rate": bl_rate, "tx_no_rate": tx_rate,
            "n_baseline": len(baseline), "n_treatment": len(group),
        })
    return scores


# ── Figure 0: Hero Figure ─────────────────────────────────────────────────
def fig0_hero(scores, results, out_dir):
    """Hero figure: 3-panel overview of AMEL."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    # ── Panel A: Category interaction ──
    ax = axes[0]
    categories = ["clear_positive", "ambiguous", "clear_negative"]
    cat_labels_short = ["Clear\nPositive", "Ambiguous", "Clear\nNegative"]
    cat_colors_list = [CAT_COLORS[c] for c in categories]

    cat_scores = {c: [] for c in categories}
    for s in scores:
        if s["category"] in cat_scores:
            cat_scores[s["category"]].append(s["bias_score"])

    x = np.arange(len(categories))
    means_a = [np.mean(cat_scores[c]) for c in categories]
    cis_a = [1.96 * np.std(cat_scores[c]) / np.sqrt(len(cat_scores[c])) for c in categories]
    ds_a = [np.mean(cat_scores[c]) / np.std(cat_scores[c]) if np.std(cat_scores[c]) > 0 else 0 for c in categories]

    ax.bar(x, means_a, yerr=cis_a, width=0.55,
           color=cat_colors_list, edgecolor="white", linewidth=0.5, alpha=0.85,
           error_kw=dict(capsize=5, capthick=1.2, elinewidth=1.2, color="#374151"))
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels_short, fontsize=8)
    ax.set_ylabel("Mean Bias Score", fontsize=9)
    ax.axhline(0, color="#374151", lw=0.8)
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")
    ax.text(0.02, 0.87, "Ambiguity modulates\nthe effect", transform=ax.transAxes,
            fontsize=8, va="top", ha="left", color="#555555")

    n_comparisons = 21
    for i, (m, d) in enumerate(zip(means_a, ds_a)):
        y_off = -0.008 if m < 0 else 0.008
        va = "top" if m < 0 else "bottom"
        t, p = stats.ttest_1samp(cat_scores[categories[i]], 0)
        p_corr = min(p * n_comparisons, 1.0)
        sig = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else "ns"
        ax.text(i, m + y_off, f"d={d:.2f} {sig}", ha="center", va=va, fontsize=7, fontweight="bold")

    # ── Panel B: Polarity asymmetry ──
    ax = axes[1]
    polarities = ["no_saturated", "neutral", "yes_saturated"]
    pol_labels_short = ["No-Sat", "Neutral", "Yes-Sat"]
    pol_colors_list = [POL_COLORS[p] for p in polarities]

    pol_scores = {p: [] for p in polarities}
    for s in scores:
        if s["polarity"] in pol_scores:
            pol_scores[s["polarity"]].append(s["bias_score"])

    x = np.arange(len(polarities))
    means_b = [np.mean(pol_scores[p]) for p in polarities]
    cis_b = [1.96 * np.std(pol_scores[p]) / np.sqrt(len(pol_scores[p])) for p in polarities]
    ds_b = [np.mean(pol_scores[p]) / np.std(pol_scores[p]) if np.std(pol_scores[p]) > 0 else 0 for p in polarities]

    ax.bar(x, means_b, yerr=cis_b, width=0.5,
           color=pol_colors_list, edgecolor="white", linewidth=0.5, alpha=0.85,
           error_kw=dict(capsize=5, capthick=1.2, elinewidth=1.2, color="#374151"))
    ax.set_xticks(x)
    ax.set_xticklabels(pol_labels_short, fontsize=8)
    ax.axhline(0, color="#374151", lw=0.8)
    ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")

    ratio = abs(means_b[0]) / abs(means_b[2]) if abs(means_b[2]) > 0 else float("inf")
    ax.text(0.02, 0.87, f"Negativity asymmetry ({ratio:.1f}x)",
            transform=ax.transAxes, fontsize=8, va="top", ha="left", color="#555555")

    for i, (m, d) in enumerate(zip(means_b, ds_b)):
        y_off = -0.005 if m < 0 else 0.005
        va = "top" if m < 0 else "bottom"
        ax.text(i, m + y_off, f"d={d:.2f}", ha="center", va=va, fontsize=7, fontweight="bold")

    # ── Panel C: Accumulation flatness (no_saturated only) ──
    ax = axes[2]
    length_scores = defaultdict(list)
    for s in scores:
        if s["polarity"] == "no_saturated":
            length_scores[s["context_length"]].append(s["bias_score"])

    lengths = sorted(length_scores.keys())
    means_c = [np.mean(length_scores[l]) for l in lengths]
    cis_c = [1.96 * np.std(length_scores[l]) / np.sqrt(len(length_scores[l])) for l in lengths]

    ax.errorbar(lengths, means_c, yerr=cis_c, marker="o", markersize=6,
                color=POL_COLORS["no_saturated"], linewidth=2, capsize=4, capthick=1.2,
                markerfacecolor="white", markeredgewidth=2)
    ax.axhline(0, color="#374151", lw=0.8, ls="-")
    ax.set_xlabel("Context length (turns)", fontsize=9)
    ax.set_xticks(lengths)
    ax.text(0.02, 0.97, "(c)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")
    ax.text(0.02, 0.87, "Bias saturates immediately", transform=ax.transAxes,
            fontsize=8, va="top", ha="left", color="#555555")

    all_lens = []
    all_scores_c = []
    for s in scores:
        if s["polarity"] == "no_saturated":
            all_lens.append(s["context_length"])
            all_scores_c.append(s["bias_score"])
    if all_lens:
        r, p = stats.spearmanr(all_lens, all_scores_c)
        ax.text(0.97, 0.05, f"r = {r:.3f}\np = {p:.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="#6b7280",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e5e7eb", alpha=0.9))

    fig.tight_layout(w_pad=3)
    fig.savefig(out_dir / "fig0_hero.pdf")
    fig.savefig(out_dir / "fig0_hero.png")
    plt.close(fig)
    print("  Fig 0: Hero figure")


# ── Figure 1: Experimental Design Schematic ────────────────────────────────
def fig1_design(out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # ── Column background fills ──
    baseline_bg = FancyBboxPatch((0.3, 0.2), 4.2, 4.3, boxstyle="round,pad=0.15",
                                  facecolor="#f0f4f8", edgecolor="none", alpha=0.5)
    treatment_bg = FancyBboxPatch((5.5, 0.2), 4.2, 4.3, boxstyle="round,pad=0.15",
                                   facecolor="#fff8f0", edgecolor="none", alpha=0.5)
    ax.add_patch(baseline_bg)
    ax.add_patch(treatment_bg)

    # ── Baseline column ──
    ax.text(2.5, 4.2, "Baseline (Control)", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#374151")

    # System prompt box
    sp_bl = FancyBboxPatch((1.3, 3.1), 2.4, 0.55, boxstyle="round,pad=0.12",
                            facecolor="#dbeafe", edgecolor="#4477AA", linewidth=1.2)
    ax.add_patch(sp_bl)
    ax.text(2.5, 3.37, "System Prompt", ha="center", va="center", fontsize=9)

    # Arrow
    arrow1 = FancyArrowPatch((2.5, 3.1), (2.5, 2.7), arrowstyle="-|>",
                              mutation_scale=12, color="#888888", lw=1.2)
    ax.add_patch(arrow1)

    # Test item box
    ti_bl = FancyBboxPatch((1.5, 2.1), 2.0, 0.55, boxstyle="round,pad=0.12",
                            facecolor="#fef3c7", edgecolor="#CCBB44", linewidth=1.2)
    ax.add_patch(ti_bl)
    ax.text(2.5, 2.37, "Test Item", ha="center", va="center", fontsize=9)

    # Arrow
    arrow2 = FancyArrowPatch((2.5, 2.1), (2.5, 1.7), arrowstyle="-|>",
                              mutation_scale=12, color="#888888", lw=1.2)
    ax.add_patch(arrow2)

    # Response box
    resp_bl = FancyBboxPatch((1.5, 1.1), 2.0, 0.55, boxstyle="round,pad=0.12",
                              facecolor="#f3f4f6", edgecolor="#999999", linewidth=1.2)
    ax.add_patch(resp_bl)
    ax.text(2.5, 1.37, "Response", ha="center", va="center", fontsize=9)

    # ── Treatment column ──
    ax.text(7.5, 4.2, "Biased Context (Treatment)", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#374151")

    # System prompt box
    sp_tx = FancyBboxPatch((6.3, 3.1), 2.4, 0.55, boxstyle="round,pad=0.12",
                            facecolor="#dbeafe", edgecolor="#4477AA", linewidth=1.2)
    ax.add_patch(sp_tx)
    ax.text(7.5, 3.37, "System Prompt", ha="center", va="center", fontsize=9)

    # Arrow
    arrow3 = FancyArrowPatch((7.5, 3.1), (7.5, 2.75), arrowstyle="-|>",
                              mutation_scale=12, color="#888888", lw=1.2)
    ax.add_patch(arrow3)

    # Biased history box
    hist_tx = FancyBboxPatch((5.9, 2.0), 3.2, 0.7, boxstyle="round,pad=0.12",
                              facecolor="#fde8e0", edgecolor="#EE7733", linewidth=1.2)
    ax.add_patch(hist_tx)
    ax.text(7.5, 2.4, 'N turns of skewed history', ha="center", va="center", fontsize=8)
    ax.text(7.5, 2.15, '(90% "no" or 90% "yes")', ha="center", va="center", fontsize=7.5, color="#666666")

    # Arrow
    arrow4 = FancyArrowPatch((7.5, 2.0), (7.5, 1.7), arrowstyle="-|>",
                              mutation_scale=12, color="#888888", lw=1.2)
    ax.add_patch(arrow4)

    # Test item box
    ti_tx = FancyBboxPatch((6.0, 1.1), 3.0, 0.55, boxstyle="round,pad=0.12",
                            facecolor="#fef3c7", edgecolor="#CCBB44", linewidth=1.2)
    ax.add_patch(ti_tx)
    ax.text(7.5, 1.37, "Same Test Item", ha="center", va="center", fontsize=9)

    # Arrow
    arrow5 = FancyArrowPatch((7.5, 1.1), (7.5, 0.75), arrowstyle="-|>",
                              mutation_scale=12, color="#888888", lw=1.2)
    ax.add_patch(arrow5)

    # Response box
    resp_tx = FancyBboxPatch((6.0, 0.15), 3.0, 0.55, boxstyle="round,pad=0.12",
                              facecolor="#f3f4f6", edgecolor="#999999", linewidth=1.2)
    ax.add_patch(resp_tx)
    ax.text(7.5, 0.42, "Response (biased?)", ha="center", va="center", fontsize=9)

    # ── Comparison arrow ──
    comp_arrow = FancyArrowPatch((4.2, 1.37), (5.8, 1.37), arrowstyle="<|-|>",
                                  mutation_scale=14, color="#4477AA", lw=2)
    ax.add_patch(comp_arrow)
    ax.text(5.0, 0.85, "Bias Score =\nP(no|treatment) - P(no|baseline)",
            ha="center", va="center", fontsize=8, color="#4477AA", fontweight="bold")

    # ── Divider ──
    ax.plot([5, 5], [0.2, 4.5], color="#e0e0e0", lw=1, ls="--")

    fig.savefig(out_dir / "fig1_experimental_design.pdf")
    fig.savefig(out_dir / "fig1_experimental_design.png")
    plt.close(fig)
    print("  Fig 1: Experimental design")


# ── Figure 2: Model Comparison ─────────────────────────────────────────────
def fig2_model_comparison(scores, out_dir):
    model_scores = defaultdict(list)
    for s in scores:
        model_scores[s["model"]].append(s["bias_score"])

    # Compute stats
    model_stats = {}
    for model, vals in model_scores.items():
        arr = np.array(vals)
        mean = np.mean(arr)
        ci = 1.96 * np.std(arr) / np.sqrt(len(arr))
        d = mean / np.std(arr) if np.std(arr) > 0 else 0
        t, p = stats.ttest_1samp(arr, 0)
        model_stats[model] = {"mean": mean, "ci": ci, "d": d, "p": p, "n": len(arr)}

    # Sort by effect size
    sorted_models = sorted(model_stats.keys(),
                           key=lambda m: model_stats[m]["mean"])

    fig, ax = plt.subplots(figsize=(7, 5.5))

    y_pos = np.arange(len(sorted_models))
    means = [model_stats[m]["mean"] for m in sorted_models]
    cis = [model_stats[m]["ci"] for m in sorted_models]
    colors = [get_color(m) for m in sorted_models]

    bars = ax.barh(y_pos, means, xerr=cis, height=0.65,
                   color=colors, edgecolor="white", linewidth=0.5,
                   error_kw=dict(capsize=3, capthick=1, elinewidth=1, color="#374151"))

    ax.set_yticks(y_pos)
    ax.set_yticklabels([get_short(m) for m in sorted_models], fontsize=8)
    ax.set_xlabel("Mean Bias Score (conforming <-- | --> contrarian)")
    ax.axvline(0, color="#374151", lw=0.8, ls="-")
    ax.set_title("Model Susceptibility to AMEL", fontweight="bold", pad=12)

    # Add significance stars (Bonferroni-corrected, 21 comparisons)
    n_comparisons = 21
    for i, model in enumerate(sorted_models):
        s = model_stats[model]
        x_pos = s["mean"] + s["ci"] + 0.005 if s["mean"] >= 0 else s["mean"] - s["ci"] - 0.005
        ha = "left" if s["mean"] >= 0 else "right"
        p_corr = min(s["p"] * n_comparisons, 1.0)
        sig = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else "ns"
        label = f"d={s['d']:.2f} {sig}"
        ax.text(x_pos, i, label, va="center", ha=ha, fontsize=7, color="#6b7280")

    # Provider legend
    handles = [mpatches.Patch(color=c, label=p) for p, c in PROVIDER_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", framealpha=0.9, fontsize=8)

    fig.savefig(out_dir / "fig2_model_comparison.pdf")
    fig.savefig(out_dir / "fig2_model_comparison.png")
    plt.close(fig)
    print("  Fig 2: Model comparison")


# ── Figure 3: Ambiguity Interaction ────────────────────────────────────────
def fig3_category(scores, out_dir):
    categories = ["clear_positive", "ambiguous", "clear_negative"]
    cat_labels = ["Clear Positive\n(should be 'yes')", "Ambiguous\n(borderline)", "Clear Negative\n(should be 'no')"]
    cat_colors = [CAT_COLORS[c] for c in categories]

    cat_scores = {c: [] for c in categories}
    for s in scores:
        if s["category"] in cat_scores:
            cat_scores[s["category"]].append(s["bias_score"])

    fig, ax = plt.subplots(figsize=(6, 4.5))

    x = np.arange(len(categories))
    means = [np.mean(cat_scores[c]) for c in categories]
    cis = [1.96 * np.std(cat_scores[c]) / np.sqrt(len(cat_scores[c])) for c in categories]
    ds = [np.mean(cat_scores[c]) / np.std(cat_scores[c]) if np.std(cat_scores[c]) > 0 else 0 for c in categories]

    bars = ax.bar(x, means, yerr=cis, width=0.55,
                  color=cat_colors, edgecolor="white", linewidth=0.5, alpha=0.85,
                  error_kw=dict(capsize=5, capthick=1.2, elinewidth=1.2, color="#374151"))

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=9)
    ax.set_ylabel("Mean Bias Score")
    ax.axhline(0, color="#374151", lw=0.8)
    ax.set_title("AMEL by Test Item Ambiguity", fontweight="bold", pad=12)

    # Annotations (Bonferroni-corrected, 21 comparisons)
    n_comparisons = 21
    for i, (m, d) in enumerate(zip(means, ds)):
        y_off = -0.008 if m < 0 else 0.008
        va = "top" if m < 0 else "bottom"
        t, p = stats.ttest_1samp(cat_scores[categories[i]], 0)
        p_corr = min(p * n_comparisons, 1.0)
        sig = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else "ns"
        ax.text(i, m + y_off, f"d = {d:.2f} {sig}", ha="center", va=va, fontsize=8, fontweight="bold")

    fig.savefig(out_dir / "fig3_category_interaction.pdf")
    fig.savefig(out_dir / "fig3_category_interaction.png")
    plt.close(fig)
    print("  Fig 3: Category interaction")


# ── Figure 4: Domain Sensitivity ──────────────────────────────────────────
def fig4_domain(scores, out_dir):
    domains = ["code_review", "content_moderation", "meals"]
    domain_labels = ["Code Review", "Content\nModeration", "Meal\nEvaluation"]
    domain_colors = DOM_COLORS

    dom_scores = {d: [] for d in domains}
    for s in scores:
        if s["domain"] in dom_scores:
            dom_scores[s["domain"]].append(s["bias_score"])

    fig, ax = plt.subplots(figsize=(5.5, 4))

    x = np.arange(len(domains))
    means = [np.mean(dom_scores[d]) for d in domains]
    cis = [1.96 * np.std(dom_scores[d]) / np.sqrt(len(dom_scores[d])) for d in domains]
    ds = [np.mean(dom_scores[d]) / np.std(dom_scores[d]) if np.std(dom_scores[d]) > 0 else 0 for d in domains]

    bars = ax.bar(x, means, yerr=cis, width=0.5,
                  color=domain_colors, edgecolor="white", linewidth=0.5,
                  error_kw=dict(capsize=5, capthick=1.2, elinewidth=1.2, color="#374151"))

    ax.set_xticks(x)
    ax.set_xticklabels(domain_labels, fontsize=9)
    ax.set_ylabel("Mean Bias Score")
    ax.axhline(0, color="#374151", lw=0.8)
    ax.set_title("AMEL by Evaluation Domain", fontweight="bold", pad=12)

    # Bonferroni-corrected, 21 comparisons
    n_comparisons = 21
    for i, (m, d) in enumerate(zip(means, ds)):
        y_off = -0.005 if m < 0 else 0.005
        va = "top" if m < 0 else "bottom"
        t, p = stats.ttest_1samp(dom_scores[domains[i]], 0)
        p_corr = min(p * n_comparisons, 1.0)
        sig = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else "ns"
        ax.text(i, m + y_off, f"d = {d:.2f} {sig}", ha="center", va=va, fontsize=8, fontweight="bold")

    fig.savefig(out_dir / "fig4_domain_sensitivity.pdf")
    fig.savefig(out_dir / "fig4_domain_sensitivity.png")
    plt.close(fig)
    print("  Fig 4: Domain sensitivity")


# ── Figure 5: Accumulation Curves ─────────────────────────────────────────
def fig5_accumulation(scores, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)

    for idx, (polarity, title, color) in enumerate([
        ("no_saturated", "No-Saturated Context", POL_COLORS["no_saturated"]),
        ("yes_saturated", "Yes-Saturated Context", POL_COLORS["yes_saturated"]),
    ]):
        ax = axes[idx]
        length_scores = defaultdict(list)
        for s in scores:
            if s["polarity"] == polarity:
                length_scores[s["context_length"]].append(s["bias_score"])

        lengths = sorted(length_scores.keys())
        means = [np.mean(length_scores[l]) for l in lengths]
        cis = [1.96 * np.std(length_scores[l]) / np.sqrt(len(length_scores[l])) for l in lengths]

        ax.errorbar(lengths, means, yerr=cis, marker="o", markersize=6,
                    color=color, linewidth=2, capsize=4, capthick=1.2,
                    markerfacecolor="white", markeredgewidth=2)

        ax.axhline(0, color="#374151", lw=0.8, ls="-")
        ax.set_xlabel("Context Length (# prior turns)")
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xticks(lengths)

        # Correlation annotation
        all_lens = []
        all_scores = []
        for s in scores:
            if s["polarity"] == polarity:
                all_lens.append(s["context_length"])
                all_scores.append(s["bias_score"])
        if all_lens:
            r, p = stats.spearmanr(all_lens, all_scores)
            ax.text(0.97, 0.05, f"r = {r:.3f}\np = {p:.3f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color="#6b7280",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e5e7eb", alpha=0.9))

    axes[0].set_ylabel("Bias Score")
    fig.suptitle("Bias Does Not Accumulate with Context Length", fontweight="bold", fontsize=11, y=1.02)
    fig.tight_layout()

    fig.savefig(out_dir / "fig5_accumulation_curves.pdf")
    fig.savefig(out_dir / "fig5_accumulation_curves.png")
    plt.close(fig)
    print("  Fig 5: Accumulation curves")


# ── Figure 6: Heatmap (Model x Length) ─────────────────────────────────────
def fig6_heatmap(scores, out_dir):
    # Only no_saturated for clarity
    model_length = defaultdict(lambda: defaultdict(list))
    for s in scores:
        if s["polarity"] == "no_saturated":
            model_length[s["model"]][s["context_length"]].append(s["bias_score"])

    models = sorted(model_length.keys(), key=lambda m: MODEL_META.get(m, {"order": 99})["order"])
    lengths = sorted({s["context_length"] for s in scores if s["polarity"] == "no_saturated"})

    if not models or not lengths:
        print("  Fig 6: Skipped (no data)")
        return

    matrix = np.zeros((len(models), len(lengths)))
    for i, m in enumerate(models):
        for j, l in enumerate(lengths):
            vals = model_length[m].get(l, [])
            matrix[i, j] = np.mean(vals) if vals else 0

    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = max(abs(matrix.min()), abs(matrix.max()))
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(lengths)))
    ax.set_xticklabels(lengths)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([get_short(m).replace("\n", " ") for m in models], fontsize=8)
    ax.set_xlabel("Context Length (# prior turns)")
    ax.set_title("Bias Score Heatmap (No-Saturated Context)", fontweight="bold", pad=12)

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(lengths)):
            val = matrix[i, j]
            color = "white" if abs(val) > vmax * 0.5 else "#374151"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Bias Score")

    fig.savefig(out_dir / "fig6_heatmap.pdf")
    fig.savefig(out_dir / "fig6_heatmap.png")
    plt.close(fig)
    print("  Fig 6: Heatmap")


# ── Figure 7: Provider Scaling Ladder ──────────────────────────────────────
def fig7_scaling_ladder(scores, out_dir):
    model_stats = {}
    model_scores_map = defaultdict(list)
    for s in scores:
        model_scores_map[s["model"]].append(s["bias_score"])

    for model, vals in model_scores_map.items():
        arr = np.array(vals)
        model_stats[model] = {
            "mean": np.mean(arr),
            "d": np.mean(arr) / np.std(arr) if np.std(arr) > 0 else 0,
            "ci": 1.96 * np.std(arr) / np.sqrt(len(arr)),
        }

    # Define scaling chains
    chains = {
        "OpenAI":    [("gpt-4.1-nano", "Nano"), ("gpt-5.2", "5.2 Flagship")],
        "Anthropic": [("claude-haiku-4-5-20251001", "Haiku"), ("claude-sonnet-4-6", "Sonnet"), ("claude-opus-4-6", "Opus")],
        "Google":    [("gemini-2.5-flash", "Flash"), ("gemini-2.5-pro", "Pro")],
    }

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5), sharey=True)

    for idx, (provider, chain) in enumerate(chains.items()):
        ax = axes[idx]
        color = PROVIDER_COLORS[provider]

        x = np.arange(len(chain))
        ds = [abs(model_stats.get(m, {"d": 0})["d"]) for m, _ in chain]
        labels = [label for _, label in chain]

        bars = ax.bar(x, ds, width=0.5, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(provider, fontweight="bold", fontsize=11, color=color)

        for i, d in enumerate(ds):
            ax.text(i, d + 0.008, f"|d| = {d:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_ylim(0, 0.42)

    axes[0].set_ylabel("|Cohen's d| (effect size)")
    fig.suptitle("Scaling Reduces AMEL Within Each Provider", fontweight="bold", fontsize=12, y=1.04)
    fig.tight_layout()

    fig.savefig(out_dir / "fig7_scaling_ladder.pdf")
    fig.savefig(out_dir / "fig7_scaling_ladder.png")
    plt.close(fig)
    print("  Fig 7: Scaling ladder")


# ── Figure 8: Polarity Comparison ──────────────────────────────────────────
def fig8_polarity(scores, out_dir):
    polarities = ["no_saturated", "neutral", "yes_saturated"]
    pol_labels = ["No-Saturated\n(90% no history)", "Neutral\n(50/50 history)", "Yes-Saturated\n(90% yes history)"]
    pol_colors = [POL_COLORS[p] for p in polarities]

    pol_scores = {p: [] for p in polarities}
    for s in scores:
        if s["polarity"] in pol_scores:
            pol_scores[s["polarity"]].append(s["bias_score"])

    fig, ax = plt.subplots(figsize=(5.5, 4))

    x = np.arange(len(polarities))
    means = [np.mean(pol_scores[p]) for p in polarities]
    cis = [1.96 * np.std(pol_scores[p]) / np.sqrt(len(pol_scores[p])) for p in polarities]
    ds = [np.mean(pol_scores[p]) / np.std(pol_scores[p]) if np.std(pol_scores[p]) > 0 else 0 for p in polarities]

    bars = ax.bar(x, means, yerr=cis, width=0.5,
                  color=pol_colors, edgecolor="white", linewidth=0.5, alpha=0.85,
                  error_kw=dict(capsize=5, capthick=1.2, elinewidth=1.2, color="#374151"))

    ax.set_xticks(x)
    ax.set_xticklabels(pol_labels, fontsize=9)
    ax.set_ylabel("Mean Bias Score")
    ax.axhline(0, color="#374151", lw=0.8)
    ax.set_title("Negativity Asymmetry: \"No\" Bias Is Stronger", fontweight="bold", pad=12)

    for i, (m, d) in enumerate(zip(means, ds)):
        y_off = -0.005 if m < 0 else 0.005
        va = "top" if m < 0 else "bottom"
        ax.text(i, m + y_off, f"d = {d:.2f}", ha="center", va=va, fontsize=8, fontweight="bold")

    # Asymmetry annotation
    ax.annotate("", xy=(0, means[0] - 0.015), xytext=(2, means[2] + 0.015),
                arrowprops=dict(arrowstyle="<->", color="#4477AA", lw=1.5))
    ratio = abs(means[0]) / abs(means[2]) if abs(means[2]) > 0 else float("inf")
    mid_y = (means[0] + means[2]) / 2
    ax.text(1, mid_y - 0.02, f"{ratio:.1f}x stronger", ha="center", fontsize=8,
            color="#4477AA", fontweight="bold")

    fig.savefig(out_dir / "fig8_polarity_asymmetry.pdf")
    fig.savefig(out_dir / "fig8_polarity_asymmetry.png")
    plt.close(fig)
    print("  Fig 8: Polarity asymmetry")


# ── Figure 9: Contrast vs. Assimilation ───────────────────────────────────
def fig9_contrast_assimilation(scores, out_dir):
    """Grouped bar chart: congruent vs incongruent bias by category."""

    def classify(s):
        pol, gt = s["polarity"], s.get("ground_truth", s.get("category", ""))
        if pol == "neutral":
            return None
        # Map category to expected ground truth for older data
        if "ground_truth" not in s:
            if s["category"] == "clear_positive":
                gt = "yes"
            elif s["category"] == "clear_negative":
                gt = "no"
            else:
                gt = "yes"  # ambiguous coded as yes
        if pol == "no_saturated":
            return "congruent" if gt == "no" else "incongruent"
        if pol == "yes_saturated":
            return "congruent" if gt == "yes" else "incongruent"
        return None

    categories = ["clear_positive", "ambiguous", "clear_negative"]
    cat_labels = ["Clear\nPositive", "Ambiguous", "Clear\nNegative"]

    cong_means, incong_means = [], []
    cong_cis, incong_cis = [], []

    for cat in categories:
        cong_vals = [s["bias_score"] for s in scores if s["category"] == cat and classify(s) == "congruent"]
        incong_vals = [s["bias_score"] for s in scores if s["category"] == cat and classify(s) == "incongruent"]

        cong_means.append(np.mean(cong_vals) if cong_vals else 0)
        incong_means.append(np.mean(incong_vals) if incong_vals else 0)
        cong_cis.append(1.96 * np.std(cong_vals) / np.sqrt(len(cong_vals)) if len(cong_vals) > 1 else 0)
        incong_cis.append(1.96 * np.std(incong_vals) / np.sqrt(len(incong_vals)) if len(incong_vals) > 1 else 0)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, cong_means, width, yerr=cong_cis, label="Congruent",
                   color="#4477AA", edgecolor="white", linewidth=0.5, alpha=0.85,
                   error_kw=dict(capsize=4, capthick=1, elinewidth=1, color="#374151"))
    bars2 = ax.bar(x + width/2, incong_means, width, yerr=incong_cis, label="Incongruent",
                   color="#EE7733", edgecolor="white", linewidth=0.5, alpha=0.85,
                   error_kw=dict(capsize=4, capthick=1, elinewidth=1, color="#374151"))

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels)
    ax.set_ylabel("Mean Bias Score")
    ax.axhline(0, color="#374151", lw=0.8)
    ax.set_title("Assimilation vs. Contrast by Item Category", fontweight="bold", pad=12)
    ax.legend(framealpha=0.9, fontsize=9)

    fig.savefig(out_dir / "fig9_contrast_assimilation.pdf")
    fig.savefig(out_dir / "fig9_contrast_assimilation.png")
    plt.close(fig)
    print("  Fig 9: Contrast vs. assimilation")


# ── Figure 10: Confidence Scatter ─────────────────────────────────────────
def fig10_confidence_scatter(scores, results, out_dir):
    """Scatter plot: baseline entropy vs |bias score| with regression line."""
    from collections import defaultdict

    # Compute baseline entropy
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
        if p_yes == 0 or p_yes == 1:
            h = 0.0
        else:
            h = -p_yes * np.log2(p_yes) - (1 - p_yes) * np.log2(1 - p_yes)
        entropy_map[key] = h

    # Match
    entropies, abs_bias, cats = [], [], []
    for s in scores:
        key = f"{s['domain']}|{s['model']}|{s['item_id']}"
        if key in entropy_map:
            entropies.append(entropy_map[key])
            abs_bias.append(abs(s["bias_score"]))
            cats.append(s["category"])

    ent = np.array(entropies)
    ab = np.array(abs_bias)

    cat_colors = {
        "clear_positive": CAT_COLORS["clear_positive"],
        "ambiguous": CAT_COLORS["ambiguous"],
        "clear_negative": CAT_COLORS["clear_negative"],
    }
    cat_labels = {"clear_positive": "Clear Pos.", "ambiguous": "Ambiguous", "clear_negative": "Clear Neg."}

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for cat in ["clear_positive", "ambiguous", "clear_negative"]:
        mask = np.array([c == cat for c in cats])
        ax.scatter(ent[mask], ab[mask], c=cat_colors[cat], alpha=0.25, s=15,
                   label=cat_labels[cat], rasterized=True)

    # Regression line
    slope, intercept, r_val, p_val, se = stats.linregress(ent, ab)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, slope * x_line + intercept, color="#4477AA", lw=2, ls="--",
            label=f"$r$ = {stats.spearmanr(ent, ab)[0]:.3f}")

    ax.set_xlabel("Baseline Entropy (model uncertainty)")
    ax.set_ylabel("|Bias Score|")
    ax.set_title("Higher Uncertainty, Larger Bias Effect", fontweight="bold", pad=12)
    ax.legend(fontsize=8, framealpha=0.9, markerscale=3)
    ax.set_xlim(-0.05, 1.05)

    fig.savefig(out_dir / "fig10_confidence_scatter.pdf")
    fig.savefig(out_dir / "fig10_confidence_scatter.png")
    plt.close(fig)
    print("  Fig 10: Confidence scatter")


# ── Figure 11: Logprobs Density + Paired Dots ─────────────────────────────
def fig11_logprobs(out_dir):
    """P(Yes) distribution across conditions from logprobs experiment."""
    results_path = Path("results/logprobs_analysis.json")
    if not results_path.exists():
        print("  Fig 11: Skipped (no logprobs analysis data)")
        return

    with open(results_path) as f:
        data = json.load(f)

    conditions = data.get("conditions", {})
    per_item = data.get("per_item_shifts", {})

    if not conditions or not per_item:
        print("  Fig 11: Skipped (empty logprobs data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: Bar chart of mean P(Yes) across conditions
    ax = axes[0]
    cond_order = ["baseline", "yes_saturated@5", "yes_saturated@50", "no_saturated@5", "no_saturated@50"]
    cond_labels = ["Baseline", "Yes-sat\n@5", "Yes-sat\n@50", "No-sat\n@5", "No-sat\n@50"]
    cond_colors = ["#BBBBBB", "#EE7733", "#CC5500", "#4477AA", "#225588"]

    means = []
    stds = []
    for c in cond_order:
        if c in conditions:
            means.append(conditions[c]["p_yes_mean"])
            stds.append(conditions[c]["p_yes_std"] / np.sqrt(conditions[c]["n"]))
        else:
            means.append(0)
            stds.append(0)

    x = np.arange(len(cond_order))
    bars = ax.bar(x, means, yerr=stds, width=0.6,
                  color=cond_colors, edgecolor="white", linewidth=0.5,
                  error_kw=dict(capsize=4, capthick=1, elinewidth=1, color="#374151"))

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=8)
    ax.set_ylabel("Mean P(Yes) from First Token")
    ax.set_title("(a) P(Yes) Shifts Continuously", fontweight="bold", fontsize=10)
    ax.axhline(means[0] if means else 0, color="#BBBBBB", lw=0.8, ls="--", alpha=0.5)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.01, f"{m:.3f}", ha="center", va="bottom", fontsize=7)

    # Panel B: Per-item paired dot plot
    ax = axes[1]
    items = sorted(per_item.keys())

    # Classify items into categories and build display data
    def _item_category(item_id):
        """Infer category from item ID prefix."""
        if "cp_" in item_id or item_id.startswith("test_code_cp") or item_id.startswith("cp"):
            return "clear_positive"
        elif "cn_" in item_id or item_id.startswith("test_code_cn") or item_id.startswith("cn"):
            return "clear_negative"
        elif "amb_" in item_id or item_id.startswith("test_code_amb") or item_id.startswith("amb"):
            return "ambiguous"
        # Fall back to per_item data if available
        return per_item.get(item_id, {}).get("category", "ambiguous")

    def _short_label(item_id):
        """Shorten item ID: cp_01 -> CP1, amb_01 -> A1, cn_01 -> CN1."""
        name = item_id.replace("test_code_", "")
        # Try to extract category prefix and number
        for prefix, short in [("cp_", "CP"), ("cn_", "CN"), ("amb_", "A"),
                               ("cp", "CP"), ("cn", "CN"), ("amb", "A")]:
            if name.startswith(prefix):
                num = name[len(prefix):].lstrip("_0") or "0"
                return f"{short}{num}"
        return name

    # Sort items by category
    cat_order = {"clear_positive": 0, "ambiguous": 1, "clear_negative": 2}
    items_with_cat = [(item_id, _item_category(item_id)) for item_id in items]
    items_with_cat.sort(key=lambda x: (cat_order.get(x[1], 1), x[0]))

    sorted_items = [x[0] for x in items_with_cat]
    sorted_cats = [x[1] for x in items_with_cat]

    item_shifts_no = []
    item_shifts_yes = []
    item_labels = []

    for item_id in sorted_items:
        d = per_item[item_id]
        item_shifts_no.append(d.get("no_saturated@5_shift", 0))
        item_shifts_yes.append(d.get("yes_saturated@5_shift", 0))
        item_labels.append(_short_label(item_id))

    y = np.arange(len(sorted_items))

    # Draw light grey background bands to group by category
    prev_cat = None
    band_start = 0
    cat_label_map = {"clear_positive": "Clear Pos.", "ambiguous": "Ambiguous", "clear_negative": "Clear Neg."}
    band_color_toggle = False

    for i, cat in enumerate(sorted_cats + [None]):
        if cat != prev_cat and prev_cat is not None:
            if band_color_toggle:
                ax.axhspan(band_start - 0.5, i - 0.5, facecolor="#f5f5f5", edgecolor="none", zorder=0)
            # Draw separator line
            if i < len(sorted_cats):
                ax.axhline(i - 0.5, color="#dddddd", lw=0.8, zorder=0)
            band_start = i
            band_color_toggle = not band_color_toggle
        if prev_cat is None:
            prev_cat = cat
            continue
        prev_cat = cat

    ax.scatter(item_shifts_no, y, color="#4477AA", s=30, zorder=3, label="No-sat@5", alpha=0.8)
    ax.scatter(item_shifts_yes, y, color="#EE7733", s=30, zorder=3, label="Yes-sat@5", marker="s", alpha=0.8)

    for i in range(len(sorted_items)):
        ax.plot([item_shifts_no[i], item_shifts_yes[i]], [y[i], y[i]],
                color="#d1d5db", lw=0.8, zorder=1)

    ax.axvline(0, color="#374151", lw=0.8, ls="-")
    ax.set_yticks(y)
    ax.set_yticklabels(item_labels, fontsize=6)
    ax.set_xlabel("Shift in P(Yes) from Baseline")
    ax.set_title("(b) Per-Item P(Yes) Shifts", fontweight="bold", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Logprobs: Probability Distribution Shifts Under AMEL",
                 fontweight="bold", fontsize=11, y=1.02)
    fig.tight_layout()

    fig.savefig(out_dir / "fig11_logprobs.pdf")
    fig.savefig(out_dir / "fig11_logprobs.png")
    plt.close(fig)
    print("  Fig 11: Logprobs")


# ── Figure 12: Flipped Framing Comparison ─────────────────────────────────
def fig12_flipped(out_dir):
    """Compares asymmetry ratios between original and flipped framing."""
    results_path = Path("results/flipped_analysis.json")
    if not results_path.exists():
        print("  Fig 12: Skipped (no flipped analysis data)")
        return

    with open(results_path) as f:
        data = json.load(f)

    comparison = data.get("comparison", {})
    if not comparison:
        print("  Fig 12: Skipped (empty flipped data)")
        return

    models = sorted(comparison.keys())
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(n_models)
    width = 0.35

    orig_ratios = [comparison[m]["original_ratio"] for m in models]
    flip_ratios = [comparison[m]["flipped_ratio"] for m in models]

    bars1 = ax.bar(x - width/2, orig_ratios, width, label="Original\n(Is this production-ready?)",
                   color="#4477AA", edgecolor="white", linewidth=0.5, alpha=0.85)
    bars2 = ax.bar(x + width/2, flip_ratios, width, label="Flipped\n(Should this be rejected?)",
                   color="#EE7733", edgecolor="white", linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    model_labels = [get_short(m).replace("\n", " ") for m in models]
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.set_ylabel("|BS(no_sat)| / |BS(yes_sat)|")
    ax.axhline(1.0, color="#374151", lw=0.8, ls="--", label="No asymmetry")
    ax.set_title("Negativity Asymmetry: Original vs. Flipped Framing", fontweight="bold", pad=12)

    # Annotate which hypothesis — inside the plot area with background box
    for i, model in enumerate(models):
        hyp = comparison[model]["hypothesis_supported"]
        short_hyp = "RLHF" if "RLHF" in hyp else "Token"
        y_max = max(orig_ratios[i], flip_ratios[i])
        ax.text(i, y_max + 0.1, short_hyp, ha="center", va="bottom", fontsize=7,
                fontweight="bold", color="#374151",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#cccccc", alpha=0.9))

    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    fig.savefig(out_dir / "fig12_flipped_framing.pdf")
    fig.savefig(out_dir / "fig12_flipped_framing.png")
    plt.close(fig)
    print("  Fig 12: Flipped framing")


# ── Figure 13: Positional Placement Bar Chart ─────────────────────────────
def fig13_positional(out_dir):
    """Bar chart comparing START, END, SPREAD, CONTROL_5, FULL_50."""
    results_path = Path("results/positional_analysis.json")
    if not results_path.exists():
        print("  Fig 13: Skipped (no positional analysis data)")
        return

    with open(results_path) as f:
        data = json.load(f)

    combined = data.get("combined", {})
    if not combined:
        print("  Fig 13: Skipped (empty positional data)")
        return

    cond_order = ["CONTROL_5", "START", "END", "SPREAD", "FULL_50"]
    cond_labels = ["Control\n(5/5 biased)", "Start\n(5/50 biased)", "End\n(5/50 biased)",
                   "Spread\n(5/50 biased)", "Full\n(50/50 biased)"]
    # Reference conditions grey, experimental conditions blue (same color)
    cond_colors = ["#BBBBBB", "#4477AA", "#4477AA", "#4477AA", "#BBBBBB"]

    means = []
    cis = []
    present = []
    for c in cond_order:
        if c in combined and combined[c]["n"] > 0:
            means.append(combined[c]["mean_bs"])
            n = combined[c]["n"]
            std = combined[c]["std_bs"]
            cis.append(1.96 * std / np.sqrt(n) if n > 1 else 0)
            present.append(True)
        else:
            means.append(0)
            cis.append(0)
            present.append(False)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(cond_order))
    bars = ax.bar(x, means, yerr=cis, width=0.6,
                  color=[c if p else "#e5e7eb" for c, p in zip(cond_colors, present)],
                  edgecolor="white", linewidth=0.5, alpha=0.85,
                  error_kw=dict(capsize=5, capthick=1.2, elinewidth=1.2, color="#374151"))

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=9)
    ax.set_ylabel("Mean Bias Score")
    ax.axhline(0, color="#374151", lw=0.8)
    ax.set_title("Position of Biased Turns Does Not Matter", fontweight="bold", pad=12)

    # Annotations
    for i, (m, ci) in enumerate(zip(means, cis)):
        if present[i]:
            d = combined[cond_order[i]].get("cohens_d", 0)
            y_off = -0.008 if m < 0 else 0.008
            va = "top" if m < 0 else "bottom"
            ax.text(i, m + y_off, f"d={d:.2f}", ha="center", va=va, fontsize=8, fontweight="bold")

    # Kruskal-Wallis annotation
    kw = data.get("kruskal_wallis_combined", {})
    if kw:
        sig_text = f"KW: H={kw['H']:.1f}, p={kw['p']:.3f}"
        ax.text(0.98, 0.95, sig_text, transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#6b7280",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e5e7eb"))

    fig.savefig(out_dir / "fig13_positional.pdf")
    fig.savefig(out_dir / "fig13_positional.png")
    plt.close(fig)
    print("  Fig 13: Positional placement")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    data_file = Path("data/all_results.jsonl")
    out_dir = Path("results/paper_figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {data_file}...")
    results = load_results(data_file)
    print(f"  Loaded {len(results)} results")

    print("Computing bias scores...")
    scores = compute_bias_scores(results)
    print(f"  Computed {len(scores)} bias comparisons")

    print("\nGenerating figures...")
    fig0_hero(scores, results, out_dir)
    fig1_design(out_dir)
    fig2_model_comparison(scores, out_dir)
    fig3_category(scores, out_dir)
    fig4_domain(scores, out_dir)
    fig5_accumulation(scores, out_dir)
    fig6_heatmap(scores, out_dir)
    fig7_scaling_ladder(scores, out_dir)
    fig8_polarity(scores, out_dir)
    fig9_contrast_assimilation(scores, out_dir)
    fig10_confidence_scatter(scores, results, out_dir)
    fig11_logprobs(out_dir)
    fig12_flipped(out_dir)
    fig13_positional(out_dir)

    print(f"\nAll figures saved to {out_dir}/")
    print("Formats: .pdf (vector for submission) + .png (preview)")


if __name__ == "__main__":
    main()
