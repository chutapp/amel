"""Shared utilities for AMEL analysis scripts."""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# Default data file; override with AMEL_DATA_FILE env var for v2 / adjudicated runs.
DEFAULT_DATA_FILE = Path(os.environ.get("AMEL_DATA_FILE", "data/all_results.jsonl"))

# Bonferroni correction factor (21 test items)
N_COMPARISONS = 21


def load_results(path=None):
    """Load experiment results from JSONL file."""
    path = path or DEFAULT_DATA_FILE
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_bias_scores(results):
    """Compute bias scores matching paper convention.

    For no_saturated/neutral: BS = P(no|treatment) - P(no|baseline)
    For yes_saturated: BS = P(yes|treatment) - P(yes|baseline)

    Positive BS = model shifted toward saturated polarity (conforming).
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

        target = "no" if polarity in ("no_saturated", "neutral") else "yes"
        bl_rate = sum(1 for r in baseline if r["parsed_response"] == target) / len(baseline)
        tx_rate = sum(1 for r in group if r["parsed_response"] == target) / len(group)
        bs = tx_rate - bl_rate

        category = group[0].get("test_item_category", "unknown")
        ground_truth = group[0].get("test_item_ground_truth", "unknown")
        scores.append({
            "domain": domain, "model": model, "item_id": item_id,
            "polarity": polarity, "context_length": int(ctx_len),
            "category": category, "ground_truth": ground_truth,
            "bias_score": bs,
            "bl_rate": bl_rate, "tx_rate": tx_rate,
            "n_baseline": len(baseline), "n_treatment": len(group),
        })
    return scores


def save_json(data, path):
    """Save data to JSON with numpy serialization support."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _serialize(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_serialize)
    print(f"Saved: {path}")
