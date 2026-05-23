"""Qualitative examples.

Finds the top 5 most-biased ambiguous items and extracts raw response pairs
(baseline vs no_saturated) for GPT-4.1 Nano. Outputs LaTeX table.
"""

from collections import defaultdict

import numpy as np

from analysis.utils import load_results, compute_bias_scores, save_json


def main():
    print("Loading data...")
    results = load_results()
    scores = compute_bias_scores(results)

    # Focus on ambiguous items, no_saturated, GPT-4.1 Nano
    target_model = "gpt-4.1-nano"
    target_polarity = "no_saturated"

    # Find most-biased ambiguous items
    ambig_scores = [
        s for s in scores
        if s["category"] == "ambiguous"
        and s["model"] == target_model
        and s["polarity"] == target_polarity
    ]

    # Sort by absolute bias score
    ambig_scores.sort(key=lambda s: abs(s["bias_score"]), reverse=True)
    top5 = ambig_scores[:5]

    print(f"Top 5 most-biased ambiguous items ({target_model}, {target_polarity}):")
    for s in top5:
        print(f"  {s['item_id']}: BS={s['bias_score']:.3f} (bl={s['bl_rate']:.2f}, tx={s['tx_rate']:.2f})")

    # Extract raw responses
    examples = []
    for s in top5:
        # Get baseline responses
        bl_responses = [
            r for r in results
            if r["model"] == target_model
            and r["test_item_id"] == s["item_id"]
            and r["domain"] == s["domain"]
            and r["polarity"] == "baseline"
            and r["parsed_response"] is not None
        ]
        # Get treatment responses (pick context_length=10 for readability)
        tx_responses = [
            r for r in results
            if r["model"] == target_model
            and r["test_item_id"] == s["item_id"]
            and r["domain"] == s["domain"]
            and r["polarity"] == target_polarity
            and r["context_length"] == 10
            and r["parsed_response"] is not None
        ]

        # Pick representative examples (first yes and first no if both exist)
        bl_yes = next((r for r in bl_responses if r["parsed_response"] == "yes"), None)
        bl_no = next((r for r in bl_responses if r["parsed_response"] == "no"), None)
        tx_example = tx_responses[0] if tx_responses else None

        bl_yes_count = sum(1 for r in bl_responses if r["parsed_response"] == "yes")
        bl_no_count = sum(1 for r in bl_responses if r["parsed_response"] == "no")
        tx_yes_count = sum(1 for r in tx_responses if r["parsed_response"] == "yes")
        tx_no_count = sum(1 for r in tx_responses if r["parsed_response"] == "no")

        examples.append({
            "item_id": s["item_id"],
            "domain": s["domain"],
            "item_text": tx_responses[0]["test_item_text"] if tx_responses else bl_responses[0]["test_item_text"] if bl_responses else "",
            "bias_score": s["bias_score"],
            "baseline_yes": bl_yes_count,
            "baseline_no": bl_no_count,
            "treatment_yes": tx_yes_count,
            "treatment_no": tx_no_count,
            "baseline_example": (bl_yes or bl_no or {}).get("raw_response", "")[:200],
            "treatment_example": (tx_example or {}).get("raw_response", "")[:200],
        })

    # Generate LaTeX table
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{p{4cm}ccccp{5cm}}",
        r"\toprule",
        r"\textbf{Item} & \textbf{Domain} & \textbf{BL (y/n)} & \textbf{TX (y/n)} & $BS$ & \textbf{Treatment Response (excerpt)} \\",
        r"\midrule",
    ]

    for ex in examples:
        item_text = _escape_latex(ex["item_text"][:60] + "..." if len(ex["item_text"]) > 60 else ex["item_text"])
        domain = ex["domain"].replace("_", r"\_")
        bl = f"{ex['baseline_yes']}/{ex['baseline_no']}"
        tx = f"{ex['treatment_yes']}/{ex['treatment_no']}"
        bs = f"{ex['bias_score']:.2f}"
        excerpt = _escape_latex(ex["treatment_example"][:100] + "..." if len(ex["treatment_example"]) > 100 else ex["treatment_example"])
        latex_lines.append(f"{item_text} & {domain} & {bl} & {tx} & {bs} & {excerpt} \\\\")

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Top 5 most-biased ambiguous items (GPT-4.1 Nano, no-saturated context, $N=10$). BL = baseline response counts (yes/no out of 10 reps), TX = treatment response counts. $BS$ = bias score.}",
        r"\label{tab:qualitative}",
        r"\end{table*}",
    ])

    latex_content = "\n".join(latex_lines)

    # Save
    from pathlib import Path
    out_path = Path("results/qualitative_examples.tex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(latex_content)
    print(f"Saved: {out_path}")

    save_json(examples, "results/qualitative_examples.json")


def _escape_latex(text):
    """Escape special LaTeX characters."""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    main()
