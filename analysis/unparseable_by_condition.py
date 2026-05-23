"""Unparseable rate by (model, condition) — MAR/MNAR sensitivity for B1.

Tests whether the 7.8% unparseable rate is concentrated in specific conditions
in a way that could systematically bias the BS = P(treat) - P(base) contrast.

Outputs:
    results/unparseable_by_condition.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import chi2_contingency

from analysis.utils import load_results


def main() -> None:
    rows = load_results()
    # Counters: by polarity, by (model, polarity), by (domain, polarity)
    by_pol: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "unparseable": 0})
    by_model_pol: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"total": 0, "unparseable": 0})
    by_domain_pol: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"total": 0, "unparseable": 0})

    for r in rows:
        pol = r["polarity"]
        model = r["model"]
        domain = r["domain"]
        unp = r.get("parsed_response") is None
        by_pol[pol]["total"] += 1
        by_pol[pol]["unparseable"] += int(unp)
        by_model_pol[(model, pol)]["total"] += 1
        by_model_pol[(model, pol)]["unparseable"] += int(unp)
        by_domain_pol[(domain, pol)]["total"] += 1
        by_domain_pol[(domain, pol)]["unparseable"] += int(unp)

    # Chi-squared: is unparseable rate independent of polarity?
    pols = ["baseline", "no_saturated", "yes_saturated", "neutral"]
    parseable = [by_pol[p]["total"] - by_pol[p]["unparseable"] for p in pols]
    unparseable = [by_pol[p]["unparseable"] for p in pols]
    chi2, p, dof, _ = chi2_contingency([parseable, unparseable])

    out = {
        "by_polarity": {
            p: {
                "total": by_pol[p]["total"],
                "unparseable": by_pol[p]["unparseable"],
                "rate": by_pol[p]["unparseable"] / max(by_pol[p]["total"], 1),
            }
            for p in pols
        },
        "by_model_polarity": {
            f"{m}|{p}": {
                "total": v["total"],
                "unparseable": v["unparseable"],
                "rate": v["unparseable"] / max(v["total"], 1),
            }
            for (m, p), v in sorted(by_model_pol.items())
        },
        "by_domain_polarity": {
            f"{d}|{p}": {
                "total": v["total"],
                "unparseable": v["unparseable"],
                "rate": v["unparseable"] / max(v["total"], 1),
            }
            for (d, p), v in sorted(by_domain_pol.items())
        },
        "polarity_independence_test": {
            "chi2": float(chi2),
            "dof": int(dof),
            "p_value": float(p),
            "significant": bool(p < 0.05),
            "interpretation": "Unparseable rate differs across polarities" if p < 0.05 else "Unparseable rate is independent of polarity",
        },
    }

    Path("results/unparseable_by_condition.json").write_text(json.dumps(out, indent=2))

    print("Unparseable rate by polarity:")
    for p in pols:
        v = out["by_polarity"][p]
        print(f"  {p:>15s}  n={v['total']:>6d}  unparse={v['unparseable']:>5d}  rate={v['rate']:.4f}")
    print()
    print(f"Chi-squared independence test: chi2={chi2:.2f}, dof={dof}, p={p:.3g}")
    print("Saved: results/unparseable_by_condition.json")


if __name__ == "__main__":
    main()
