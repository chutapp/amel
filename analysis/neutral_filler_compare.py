"""Compare the new non-evaluative-neutral arm against the main
experiment's evaluative-neutral arm on the same (model, item) pairs.

Question: does conversation history of any kind (non-evaluative
factual Q&A) pull responses toward "no", or is the negative shift in
the main experiment's neutral arm specifically driven by evaluative
prior turns?

Output: results/neutral_filler_compare.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results"


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def per_cell_rate(rows: list[dict], target: str) -> dict[tuple, float]:
    """Group by (model, item, polarity, ctxlen) and return modal target rate."""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for r in rows:
        key = (r["model"], r["test_item_id"], r["polarity"], r["context_length"])
        if r["parsed_response"] in ("yes", "no"):
            groups[key].append(r["parsed_response"])
    return {k: sum(1 for x in v if x == target) / len(v) for k, v in groups.items() if v}


def main() -> None:
    main_rows = load_jsonl(REPO / "data" / "all_results.jsonl")
    nef_rows = load_jsonl(REPO / "data" / "neutral_filler" / "results.jsonl")

    models = ("gpt-4.1-nano", "llama3.2:3b")
    domain = "code_review"

    # Filter main rows to (domain=code_review, model in models, polarity in baseline + neutral)
    main_relevant = [r for r in main_rows
                     if r["domain"] == domain
                     and r["model"] in models
                     and r["polarity"] in ("baseline", "neutral")]

    base_rate = per_cell_rate([r for r in main_relevant if r["polarity"] == "baseline"], "no")
    eval_neutral_rate = per_cell_rate([r for r in main_relevant if r["polarity"] == "neutral" and r["context_length"] == 50], "no")
    nef_rate = per_cell_rate(nef_rows, "no")

    out = {
        "domain": domain,
        "models": list(models),
        "context_length": 50,
        "n_items": 21,
        "per_model": {},
    }

    for model in models:
        # Build per-item paired triples
        # Baseline key has ctxlen=0; collapse to (model, item)
        base_per_item = {k[1]: v for k, v in base_rate.items() if k[0] == model}
        eval_per_item = {k[1]: v for k, v in eval_neutral_rate.items() if k[0] == model}
        nef_per_item = {k[1]: v for k, v in nef_rate.items() if k[0] == model}

        common = sorted(set(base_per_item) & set(eval_per_item) & set(nef_per_item))
        if not common:
            out["per_model"][model] = {"error": "no overlapping items"}
            continue

        eval_bs = np.array([eval_per_item[i] - base_per_item[i] for i in common])
        nef_bs = np.array([nef_per_item[i] - base_per_item[i] for i in common])

        # Per-item paired test
        t_one_e, p_one_e = stats.ttest_1samp(eval_bs, 0.0)
        t_one_n, p_one_n = stats.ttest_1samp(nef_bs, 0.0)
        diff = eval_bs - nef_bs
        t_diff, p_diff = stats.ttest_rel(eval_bs, nef_bs)

        out["per_model"][model] = {
            "n_items": len(common),
            "evaluative_neutral": {
                "mean_BS": round(float(eval_bs.mean()), 4),
                "std": round(float(eval_bs.std(ddof=1)), 4),
                "cohens_d": round(float(eval_bs.mean() / eval_bs.std(ddof=1)) if eval_bs.std(ddof=1) > 0 else 0.0, 4),
                "t_vs_zero": round(float(t_one_e), 3),
                "p_vs_zero": float(p_one_e),
            },
            "non_evaluative_neutral": {
                "mean_BS": round(float(nef_bs.mean()), 4),
                "std": round(float(nef_bs.std(ddof=1)), 4),
                "cohens_d": round(float(nef_bs.mean() / nef_bs.std(ddof=1)) if nef_bs.std(ddof=1) > 0 else 0.0, 4),
                "t_vs_zero": round(float(t_one_n), 3),
                "p_vs_zero": float(p_one_n),
            },
            "eval_minus_nef": {
                "mean_diff": round(float(diff.mean()), 4),
                "t_paired": round(float(t_diff), 3),
                "p_paired": float(p_diff),
            },
        }

    out_path = RESULTS_DIR / "neutral_filler_compare.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
