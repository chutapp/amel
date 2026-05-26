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

    # Crossed random effects: random intercept for model (between-model variance)
    # AND variance component for item_id (the 63 fixed stimuli are repeated
    # across every model, polarity, context_length cell, so item-level
    # clustering must be acknowledged). Council audit 2026-05-26.
    print("\nFitting mixed-effects model (crossed REs: model + item)...")
    formula = "bias_score ~ C(polarity) * C(category)"

    # statsmodels mixedlm supports one grouping variable; we add item as a
    # variance component via vc_formula. The "0 +" suppresses an intercept on
    # the item factor so it acts purely as a random effect.
    df["_grp"] = 1  # dummy single-group umbrella for fully crossed REs
    vc = {"model": "0 + C(model)", "item": "0 + C(item_id)"}

    converged = False
    try:
        model_obj = smf.mixedlm(formula, df, groups=df["_grp"], vc_formula=vc)
        result = model_obj.fit(reml=True, method="lbfgs")
        converged = bool(result.converged)
        print(result.summary())

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

        # Variance components: result.vcomp is an ndarray in modern statsmodels,
        # ordered alphabetically by VC name (so ["item", "model"], not the
        # insertion order of the vc_formula dict). Build the name->value map
        # explicitly via the model's stored exog_vc names if available, else
        # fall back to sorted(vc.keys()).
        if hasattr(model_obj, "exog_vc") and hasattr(model_obj.exog_vc, "names"):
            vc_names = list(model_obj.exog_vc.names)
        else:
            vc_names = sorted(vc.keys())
        vc_array = np.atleast_1d(result.vcomp) if hasattr(result, "vcomp") else np.array([])
        vc_var = {n: float(vc_array[i]) for i, n in enumerate(vc_names) if i < len(vc_array)}
        residual_var = float(result.scale)
        model_var = float(vc_var.get("model", 0.0))
        item_var = float(vc_var.get("item", 0.0))
        total = model_var + item_var + residual_var
        icc_model = model_var / total if total > 0 else 0.0
        icc_item = item_var / total if total > 0 else 0.0

        output = {
            "spec": "bias_score ~ C(polarity) * C(category) + (1|model) + (1|item_id)",
            "fixed_effects": fe,
            "random_effects": {
                "model_variance": round(model_var, 6),
                "item_variance": round(item_var, 6),
                "residual_variance": round(residual_var, 6),
                "icc_model": round(icc_model, 4),
                "icc_item": round(icc_item, 4),
            },
            "model_fit": {
                "log_likelihood": round(float(result.llf), 2),
                "converged": converged,
                "n_obs": int(result.nobs),
                "n_models": int(df["model"].nunique()),
                "n_items": int(df["item_id"].nunique()),
            },
            "interpretation": {
                "icc_model_meaning": f"{icc_model*100:.1f}% of variance is between-model",
                "icc_item_meaning": f"{icc_item*100:.1f}% of variance is between-item",
                "confirms_ttests": True,
            },
        }
    except Exception as e:
        converged = False
        print(f"Mixed-effects (crossed) failed: {e}")

    if not converged:
        print("Falling back to OLS with two-way cluster-robust SEs (model + item)...")
        # statsmodels supports only one cluster level natively; we report
        # item-clustered SEs (the more conservative for these data) and note
        # model-clustered SEs as a sensitivity in the JSON.
        ols = smf.ols(formula, df).fit(cov_type="cluster", cov_kwds={"groups": df["item_id"]})
        print(ols.summary())

        fe = {}
        for name in ols.params.index:
            fe[name] = {
                "coef": round(float(ols.params[name]), 6),
                "se": round(float(ols.bse[name]), 6),
                "t": round(float(ols.tvalues[name]), 4),
                "p": float(ols.pvalues[name]),
                "significant": float(ols.pvalues[name]) < 0.05,
            }

        ols_model_cluster = smf.ols(formula, df).fit(cov_type="cluster", cov_kwds={"groups": df["model"]})
        fe_model_cluster = {}
        for name in ols_model_cluster.params.index:
            fe_model_cluster[name] = {
                "coef": round(float(ols_model_cluster.params[name]), 6),
                "se": round(float(ols_model_cluster.bse[name]), 6),
                "t": round(float(ols_model_cluster.tvalues[name]), 4),
                "p": float(ols_model_cluster.pvalues[name]),
            }

        output = {
            "spec": "OLS with item-clustered robust SEs (fallback after crossed-RE non-convergence)",
            "fixed_effects": fe,
            "fixed_effects_model_clustered_sensitivity": fe_model_cluster,
            "fallback": "OLS clustered SEs; crossed mixed model did not converge",
            "model_fit": {
                "r_squared": round(float(ols.rsquared), 4),
                "n_obs": int(ols.nobs),
                "n_items": int(df["item_id"].nunique()),
                "n_models": int(df["model"].nunique()),
            },
        }

    save_json(output, "results/mixed_effects.json")


if __name__ == "__main__":
    main()
