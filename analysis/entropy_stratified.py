"""Empirical-entropy stratification (B2).

Bypasses the author-coded clear/ambiguous/clear-negative labels and instead
groups every bias-score observation by the EMPIRICAL baseline behavior of
that (model, item) pair: i.e. how uncertain the model is on the item in
baseline, measured as binary entropy of P(yes|baseline) across 10 reps.

Quartiles:
  Q1: lowest baseline entropy (model is confident)
  Q4: highest baseline entropy (model is genuinely uncertain)

If "ambiguity absorbs the most bias" is a real claim about model-internal
uncertainty (rather than about author-coded labels), then Q4 should show
the largest |BS|. The R4 critique is that the author-coded "ambiguous"
items in code review are operationally clear-negative for the models
(baseline P(no) >= 0.83 for 6/7 items), so the author labels and the
empirical uncertainty diverge.

Outputs:
    results/entropy_stratified.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from math import log2

import numpy as np
from scipy.stats import t as student_t

from analysis.utils import load_results, compute_bias_scores


def binary_entropy(p: float) -> float:
    if p in (0.0, 1.0):
        return 0.0
    return -(p * log2(p) + (1 - p) * log2(1 - p))


def main() -> None:
    rows = load_results()
    parsed = [r for r in rows if r.get("parsed_response") in ("yes", "no")]

    # Step 1: compute baseline P(yes) per (model, item) across the 10 baseline reps
    baselines: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for r in parsed:
        if r["polarity"] != "baseline":
            continue
        k = (r["model"], r["domain"], r["test_item_id"])
        baselines[k].append(r["parsed_response"])

    base_pyes: dict[tuple[str, str, str], float] = {}
    for k, labels in baselines.items():
        if len(labels) < 5:
            continue
        base_pyes[k] = sum(1 for x in labels if x == "yes") / len(labels)

    # Step 2: bias score per (model, item, polarity, context_length) — use existing util
    bs_records = compute_bias_scores(rows)

    # Step 3: join with empirical entropy
    enriched = []
    for bsr in bs_records:
        k = (bsr["model"], bsr["domain"], bsr["item_id"])
        if k not in base_pyes:
            continue
        p = base_pyes[k]
        enriched.append({
            "bs": bsr["bias_score"],
            "p_yes_baseline": p,
            "entropy": binary_entropy(p),
            "author_category": bsr["category"],
            "polarity": bsr["polarity"],
            "domain": bsr["domain"],
        })

    if not enriched:
        raise RuntimeError("No enriched records — check baseline coverage")

    entropies = np.array([e["entropy"] for e in enriched])
    bss = np.array([e["bs"] for e in enriched])

    # The entropy distribution is bimodal: most items have entropy=0 (the
    # model is confident at baseline). Stratify into three meaningful bins
    # rather than degenerate quartiles.
    NONZERO_MEDIAN = float(np.median(entropies[entropies > 0])) if (entropies > 0).any() else 0.5

    def stratify(ent: float) -> str:
        if ent == 0.0:
            return "B1_confident_entropy_0"
        if ent <= NONZERO_MEDIAN:
            return "B2_uncertain_low"
        return "B3_uncertain_high"

    by_q: dict[str, list[float]] = defaultdict(list)
    for e in enriched:
        by_q[stratify(e["entropy"])].append(e["bs"])

    def summary(values: list[float]) -> dict:
        a = np.array(values)
        n = len(a)
        mean = float(a.mean())
        std = float(a.std(ddof=1)) if n > 1 else 0.0
        d = mean / std if std > 0 else 0.0
        if n > 1 and std > 0:
            t_stat = mean / (std / np.sqrt(n))
            p = float(2 * (1 - student_t.cdf(abs(t_stat), df=n - 1)))
        else:
            t_stat, p = 0.0, 1.0
        ci_half = 1.96 * std / np.sqrt(max(n, 1))
        return {
            "n": n,
            "mean": mean,
            "std": std,
            "ci_lower": mean - ci_half,
            "ci_upper": mean + ci_half,
            "cohens_d": float(d),
            "t": float(t_stat),
            "p": float(p),
        }

    by_quartile = {qname: summary(by_q[qname]) for qname in sorted(by_q)}

    # Cross-tab: author category × empirical entropy quartile
    crosstab: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for e in enriched:
        crosstab[e["author_category"]][stratify(e["entropy"])] += 1

    # Domain-stratified entropy distribution for the "ambiguous" items
    # (R4's specific concern: code-review ambiguous items have high baseline P(no))
    amb_entropy_by_domain: dict[str, list[float]] = defaultdict(list)
    for r in enriched:
        if r["author_category"] == "ambiguous":
            amb_entropy_by_domain[r.get("domain", "?")].append(r["entropy"])

    out = {
        "bin_definition": {
            "B1_confident_entropy_0": "Baseline P(yes) is exactly 0 or 1 (deterministic baseline)",
            "B2_uncertain_low": f"0 < entropy <= {NONZERO_MEDIAN:.3f} (median of nonzero entropy)",
            "B3_uncertain_high": f"entropy > {NONZERO_MEDIAN:.3f}",
        },
        "by_bin": by_quartile,
        "crosstab_author_category_x_quartile": {
            cat: dict(d) for cat, d in crosstab.items()
        },
        "author_label_vs_empirical": {
            "n_records": len(enriched),
            "interpretation": "If author labels match empirical entropy, ambiguous items should concentrate in Q3-Q4. Cross-tab shows the actual distribution.",
        },
    }

    with open("results/entropy_stratified.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"Bimodal split: B1=confident (entropy=0), B2/B3 split at nonzero median entropy = {NONZERO_MEDIAN:.3f}")
    print()
    print("Bias by empirical entropy bin:")
    for qname, v in by_quartile.items():
        print(f"  {qname:25s}  n={v['n']:>5d}  mean={v['mean']:+.4f}  d={v['cohens_d']:+.3f}  p={v['p']:.3g}")
    print()
    print("Author label × empirical quartile crosstab:")
    for cat, d in crosstab.items():
        print(f"  {cat:18s} " + "  ".join(f"{q}:{c}" for q, c in sorted(d.items())))
    print()
    print("Saved: results/entropy_stratified.json")


if __name__ == "__main__":
    main()
