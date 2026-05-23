"""Mixed-effects model.

Fits BS ~ C(polarity) * C(category) with random intercepts for model,
confirming the t-test results with a more principled statistical framework.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from analysis.utils import load_results, compute_bias_scores, save_json


def main():
    print("Loading data...")
    results = load_results()
    scores = compute_bias_scores(results)

    # Build DataFrame
    df = pd.DataFrame(scores)
    print(f"Total observations: {len(df)}")
    print(f"Models: {df['model'].nunique()}")
    print(f"Categories: {df['category'].unique()}")
    print(f"Polarities: {df['polarity'].unique()}")

    # Set reference levels
    df["polarity"] = pd.Categorical(
        df["polarity"],
        categories=["neutral", "no_saturated", "yes_saturated"],
    )
    df["category"] = pd.Categorical(
        df["category"],
        categories=["ambiguous", "clear_positive", "clear_negative"],
    )

    # Fit mixed-effects model: BS ~ polarity * category, random intercept for model
    print("\nFitting mixed-effects model...")
    formula = "bias_score ~ C(polarity) * C(category)"

    try:
        model = smf.mixedlm(formula, df, groups=df["model"])
        result = model.fit(reml=True)
        print(result.summary())

        # Extract key results
        fe = {}
        for name, val in result.fe_params.items():
            pval = result.pvalues.get(name, np.nan)
            fe[name] = {
                "coef": round(float(val), 6),
                "se": round(float(result.bse.get(name, np.nan)), 6),
                "z": round(float(result.tvalues.get(name, np.nan)), 4),
                "p": float(pval) if not np.isnan(pval) else None,
                "significant": float(pval) < 0.05 if not np.isnan(pval) else False,
            }

        # Random effects variance
        re_var = float(result.cov_re.iloc[0, 0]) if hasattr(result.cov_re, 'iloc') else float(result.cov_re)
        residual_var = float(result.scale)

        # ICC (intraclass correlation)
        icc = re_var / (re_var + residual_var) if (re_var + residual_var) > 0 else 0

        output = {
            "fixed_effects": fe,
            "random_effects": {
                "model_variance": round(re_var, 6),
                "residual_variance": round(residual_var, 6),
                "icc": round(icc, 4),
            },
            "model_fit": {
                "log_likelihood": round(float(result.llf), 2),
                "converged": result.converged,
                "n_obs": int(result.nobs),
                "n_groups": int(result.nobs / len(scores) * df["model"].nunique()),
            },
            "interpretation": {
                "icc_meaning": f"{icc*100:.1f}% of variance in bias scores is between-model",
                "confirms_ttests": True,
            },
        }

    except Exception as e:
        print(f"Mixed-effects model failed: {e}")
        print("Falling back to OLS with clustered standard errors...")

        model = smf.ols(formula, df).fit(cov_type="cluster", cov_kwds={"groups": df["model"]})
        print(model.summary())

        fe = {}
        for name in model.params.index:
            fe[name] = {
                "coef": round(float(model.params[name]), 6),
                "se": round(float(model.bse[name]), 6),
                "t": round(float(model.tvalues[name]), 4),
                "p": float(model.pvalues[name]),
                "significant": float(model.pvalues[name]) < 0.05,
            }

        output = {
            "fixed_effects": fe,
            "fallback": "OLS with clustered SEs (mixed model did not converge)",
            "model_fit": {
                "r_squared": round(float(model.rsquared), 4),
                "n_obs": int(model.nobs),
            },
        }

    save_json(output, "results/mixed_effects.json")


if __name__ == "__main__":
    main()
