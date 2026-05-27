"""Compare the §5.1 logprobs experiment across the two replicated
models (GPT-4.1 Nano original; Llama 3.2 3B new). Tests whether the
"continuous P(Yes) shift, not a threshold flip" finding generalises
beyond a single model.

Output: results/logprobs_llama_compare.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent

CONDITIONS = [
    ("baseline", 0),
    ("no_saturated", 5),
    ("yes_saturated", 5),
    ("no_saturated", 50),
    ("yes_saturated", 50),
]


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in open(path):
        if line.strip():
            out.append(json.loads(line))
    return out


def per_cond_pyes(rows: list[dict]) -> dict[str, list[float]]:
    """Group P(Yes) values by condition. Filter matches analysis/logprobs_analysis.py
    (require both p_yes and p_no not None) so the Nano numbers here reproduce the
    paper's existing §5.1 statistics exactly."""
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("p_yes") is None or r.get("p_no") is None:
            continue
        label = "baseline" if r["polarity"] == "baseline" else f"{r['polarity']}@{r['context_length']}"
        out[label].append(float(r["p_yes"]))
    return out


def summary(vals: list[float]) -> dict:
    a = np.array(vals)
    return {
        "n": int(len(a)),
        "p_yes_mean": round(float(a.mean()), 4),
        "p_yes_median": round(float(np.median(a)), 4),
        "p_yes_std": round(float(a.std(ddof=1) if len(a) > 1 else 0.0), 4),
    }


def main():
    nano = load(REPO / "data" / "logprobs" / "results.jsonl")
    llama = load(REPO / "data" / "logprobs_llama" / "results.jsonl")
    print(f"Nano rows:  {len(nano)}")
    print(f"Llama rows: {len(llama)}")

    out: dict = {"models": {}, "tests": {}}
    for name, rows in (("gpt-4.1-nano", nano), ("llama3.2:3b", llama)):
        bins = per_cond_pyes(rows)
        out["models"][name] = {label: summary(vals) for label, vals in bins.items()}

        # statistical tests vs baseline
        tests = {}
        bl = bins.get("baseline", [])
        for pol, ctx in CONDITIONS:
            if pol == "baseline":
                continue
            label = f"{pol}@{ctx}"
            arr = bins.get(label, [])
            if not bl or not arr:
                continue
            u, p = stats.mannwhitneyu(bl, arr, alternative="two-sided")
            tests[f"baseline_vs_{label}"] = {
                "U": round(float(u), 2),
                "p": float(p),
                "mean_diff": round(float(np.mean(arr) - np.mean(bl)), 4),
                "direction": "P(Yes) drops" if np.mean(arr) < np.mean(bl) else "P(Yes) rises",
            }
        # context-length saturation
        for pol in ("no_saturated", "yes_saturated"):
            a5 = bins.get(f"{pol}@5", [])
            a50 = bins.get(f"{pol}@50", [])
            if a5 and a50:
                u, p = stats.mannwhitneyu(a5, a50, alternative="two-sided")
                tests[f"{pol}_5_vs_50"] = {
                    "U": round(float(u), 2),
                    "p": float(p),
                    "mean_diff": round(float(np.mean(a50) - np.mean(a5)), 4),
                }
        out["tests"][name] = tests

    # Cross-model consistency: does the qualitative pattern match?
    out["cross_model"] = {
        "nano_no_sat_5_pyes_shift": out["models"].get("gpt-4.1-nano", {}).get("no_saturated@5", {}).get("p_yes_mean", None),
        "llama_no_sat_5_pyes_shift": out["models"].get("llama3.2:3b", {}).get("no_saturated@5", {}).get("p_yes_mean", None),
        "nano_yes_sat_5_pyes": out["models"].get("gpt-4.1-nano", {}).get("yes_saturated@5", {}).get("p_yes_mean", None),
        "llama_yes_sat_5_pyes": out["models"].get("llama3.2:3b", {}).get("yes_saturated@5", {}).get("p_yes_mean", None),
        "interpretation": (
            "Both models should show the same qualitative pattern: P(Yes) lower under "
            "no_saturated than baseline, higher under yes_saturated than baseline, with the "
            "5-turn and 50-turn versions of each polarity statistically indistinguishable "
            "(saturation). Quantitative magnitudes may differ across models."
        ),
    }

    out_path = REPO / "results" / "logprobs_llama_compare.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
