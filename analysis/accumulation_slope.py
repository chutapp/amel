"""Continuous-slope test for accumulation (B4).

R4 critique: the paper's "no accumulation" claim relies on Spearman rank
correlation over only 4 distinct context-length values {5, 10, 20, 50},
which is severely underpowered. Run linear and log-linear OLS regressions
of BS on context_length to test the same null with more appropriate
statistics.

Outputs:
    results/accumulation_slope.json
"""
from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np
from scipy import stats

from analysis.utils import compute_bias_scores, load_results


def fit_slope(xs: list[float], ys: list[float]) -> dict:
    if len(xs) < 3:
        return {"n": len(xs), "slope": None, "intercept": None, "p": None, "se": None, "r2": None}
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    res = stats.linregress(x, y)
    return {
        "n": len(xs),
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "p_value": float(res.pvalue),
        "stderr": float(res.stderr),
        "r_squared": float(res.rvalue ** 2),
    }


def main() -> None:
    rows = load_results()
    bs = compute_bias_scores(rows)

    by_polarity_xs: dict[str, list[float]] = defaultdict(list)
    by_polarity_ys: dict[str, list[float]] = defaultdict(list)
    by_polarity_log_xs: dict[str, list[float]] = defaultdict(list)

    overall_xs: list[float] = []
    overall_ys: list[float] = []
    overall_log_xs: list[float] = []

    by_model_xs: dict[str, list[float]] = defaultdict(list)
    by_model_ys: dict[str, list[float]] = defaultdict(list)

    for r in bs:
        x = float(r["context_length"])
        y = float(r["bias_score"])
        pol = r["polarity"]
        by_polarity_xs[pol].append(x)
        by_polarity_ys[pol].append(y)
        by_polarity_log_xs[pol].append(math.log(x))
        overall_xs.append(x)
        overall_ys.append(y)
        overall_log_xs.append(math.log(x))
        by_model_xs[r["model"]].append(x)
        by_model_ys[r["model"]].append(y)

    out: dict[str, dict] = {
        "overall": {
            "linear": fit_slope(overall_xs, overall_ys),
            "log_linear": fit_slope(overall_log_xs, overall_ys),
        },
        "by_polarity": {
            pol: {
                "linear": fit_slope(by_polarity_xs[pol], by_polarity_ys[pol]),
                "log_linear": fit_slope(by_polarity_log_xs[pol], by_polarity_ys[pol]),
            }
            for pol in sorted(by_polarity_xs)
        },
        "by_model": {
            m: {"linear": fit_slope(by_model_xs[m], by_model_ys[m])} for m in sorted(by_model_xs)
        },
        "interpretation": {
            "null": "Slope = 0 means bias does not accumulate with context length.",
            "scope": "Linear and log-linear over four distinct x-values {5, 10, 20, 50}. Limited because of the discrete x grid, but better-specified than rank correlation given the small number of unique x values.",
        },
    }

    with open("results/accumulation_slope.json", "w") as f:
        json.dump(out, f, indent=2)

    print("Overall OLS slope of BS on context_length:")
    for name, fit in out["overall"].items():
        print(f"  {name:10s}  slope={fit['slope']:+.6f}  p={fit['p_value']:.3g}  R^2={fit['r_squared']:.4f}")
    print()
    print("By polarity:")
    for pol, fits in out["by_polarity"].items():
        f = fits["linear"]
        print(f"  {pol:15s} linear  slope={f['slope']:+.6f}  p={f['p_value']:.3g}")
    print()
    print("By model (linear slope):")
    for m, fits in out["by_model"].items():
        f = fits["linear"]
        print(f"  {m:35s}  slope={f['slope']:+.6f}  p={f['p_value']:.3g}")
    print()
    print("Saved: results/accumulation_slope.json")


if __name__ == "__main__":
    main()
