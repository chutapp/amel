"""Inter-rater reliability for the AMEL 63-item categorization.

Reads the four pseudonymized annotator files under data/annotators/anon_*.json
(produced by the external annotation task), computes:

  - Krippendorff's alpha (overall + per domain), nominal level
  - Fleiss' kappa (overall + per domain)
  - Pairwise Cohen's kappa for every annotator pair
  - Percentage agreement (overall + per domain)
  - Majority-vote adjudicated labels per item (with ties flagged)
  - Comparison of majority-vote label vs. the author's original label

Outputs:
  results/iir_scores.json
  data/annotators/aggregate_labels.json

The "single-author categorization" objection in the v1 paper is the
specific concern this script addresses.

Run:  python -m analysis.iir
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import krippendorff
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa

REPO = Path(__file__).resolve().parent.parent
ANNOTATORS_DIR = REPO / "data" / "annotators"
RESULTS_DIR = REPO / "results"
ITEM_ID_MAPPING = REPO / "data" / "annotator_id_mapping.json"

CATEGORIES = ("clear_positive", "ambiguous", "clear_negative")
CAT_TO_INT = {c: i for i, c in enumerate(CATEGORIES)}


def load_annotators() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(ANNOTATORS_DIR.glob("anon_*.json")):
        rec = json.loads(path.read_text())
        out[path.stem] = rec
    return out


def join_labels(records: dict[str, dict]) -> tuple[list[str], dict[str, str], dict[str, dict[str, str]]]:
    """Return (sorted_item_ids, item_id -> domain, annotator_id -> {item_id -> label})."""
    item_meta: dict[str, str] = {}
    per_annot: dict[str, dict[str, str]] = {}
    for ann_id, rec in records.items():
        labels = {}
        for r in rec["ratings"]:
            labels[r["item_id"]] = r.get("label")
            item_meta[r["item_id"]] = r["domain"]
        per_annot[ann_id] = labels
    ids = sorted(item_meta)
    return ids, item_meta, per_annot


def to_matrix(item_ids: list[str], per_annot: dict[str, dict[str, str]]) -> np.ndarray:
    """Shape (n_annotators, n_items). Entries: 0/1/2 or np.nan."""
    anns = sorted(per_annot)
    mat = np.full((len(anns), len(item_ids)), np.nan)
    for ai, ann in enumerate(anns):
        for ii, iid in enumerate(item_ids):
            lab = per_annot[ann].get(iid)
            if lab in CAT_TO_INT:
                mat[ai, ii] = CAT_TO_INT[lab]
    return mat


def fleiss_input(item_ids: list[str], per_annot: dict[str, dict[str, str]]) -> np.ndarray:
    """Build the (n_items, n_categories) count matrix that fleiss_kappa expects."""
    arr = np.zeros((len(item_ids), len(CATEGORIES)), dtype=int)
    for ii, iid in enumerate(item_ids):
        for ann in per_annot:
            lab = per_annot[ann].get(iid)
            if lab in CAT_TO_INT:
                arr[ii, CAT_TO_INT[lab]] += 1
    return arr


def overall_agreement(item_ids: list[str], per_annot: dict[str, dict[str, str]]) -> float:
    """Fraction of items where ALL annotators agree."""
    total = 0
    full_agree = 0
    for iid in item_ids:
        labels = [per_annot[ann].get(iid) for ann in per_annot]
        labels = [l for l in labels if l is not None]
        if not labels:
            continue
        total += 1
        if len(set(labels)) == 1:
            full_agree += 1
    return full_agree / total if total else 0.0


def majority_vote(item_ids: list[str], per_annot: dict[str, dict[str, str]]) -> dict[str, dict]:
    """Return per-item: {label, agreement (max-rate), votes, tie}."""
    out = {}
    for iid in item_ids:
        votes = [per_annot[ann].get(iid) for ann in per_annot]
        votes = [v for v in votes if v is not None]
        c = Counter(votes)
        if not c:
            out[iid] = {"label": None, "agreement": 0.0, "votes": {}, "tie": False}
            continue
        top_count = max(c.values())
        winners = [k for k, v in c.items() if v == top_count]
        tie = len(winners) > 1
        label = winners[0] if not tie else None  # None marks "no consensus"
        out[iid] = {
            "label": label,
            "agreement": top_count / sum(c.values()),
            "votes": dict(c),
            "tie": tie,
        }
    return out


def author_labels(item_ids: list[str]) -> dict[str, str]:
    """Recover the author's original category for each pseudonymized item_id."""
    # The annotation form anonymises item_id (item_001 ..). The mapping back to the
    # original test_item_id is in data/annotator_id_mapping.json. The original
    # test_item_id encodes the category via its suffix (_cp, _amb, _cn).
    if not ITEM_ID_MAPPING.exists():
        return {}
    mapping = json.loads(ITEM_ID_MAPPING.read_text())
    out = {}
    for anon, orig in mapping.items():
        suffix_map = {
            "cp": "clear_positive",
            "amb": "ambiguous",
            "cn": "clear_negative",
        }
        # original ids look like  test_<domain>_<suffix>_NN
        parts = orig.split("_")
        if len(parts) >= 3 and parts[-2] in suffix_map:
            out[anon] = suffix_map[parts[-2]]
    return out


def summarize_subset(item_ids: list[str], per_annot: dict[str, dict[str, str]], label: str) -> dict:
    if not item_ids:
        return {"n_items": 0}

    mat = to_matrix(item_ids, per_annot)
    alpha = krippendorff.alpha(reliability_data=mat, level_of_measurement="nominal")

    fleiss_arr = fleiss_input(item_ids, per_annot)
    fleiss = fleiss_kappa(fleiss_arr)

    agree = overall_agreement(item_ids, per_annot)

    out = {
        "label": label,
        "n_items": len(item_ids),
        "n_annotators": mat.shape[0],
        "krippendorff_alpha": round(float(alpha), 4),
        "fleiss_kappa": round(float(fleiss), 4),
        "full_agreement_rate": round(agree, 4),
    }
    return out


def pairwise_cohen(item_ids: list[str], per_annot: dict[str, dict[str, str]]) -> dict[str, float]:
    out = {}
    anns = sorted(per_annot)
    for a, b in combinations(anns, 2):
        y_a, y_b = [], []
        for iid in item_ids:
            la = per_annot[a].get(iid)
            lb = per_annot[b].get(iid)
            if la and lb:
                y_a.append(la)
                y_b.append(lb)
        if y_a:
            out[f"{a}_vs_{b}"] = round(float(cohen_kappa_score(y_a, y_b)), 4)
    return out


def main() -> None:
    records = load_annotators()
    print(f"Loaded {len(records)} annotators: {sorted(records)}")
    item_ids, item_domain, per_annot = join_labels(records)
    print(f"Items: {len(item_ids)}  |  domains: {sorted(set(item_domain.values()))}")

    # Overall
    overall = summarize_subset(item_ids, per_annot, "overall")
    pairs = pairwise_cohen(item_ids, per_annot)
    print(f"\nOverall: alpha={overall['krippendorff_alpha']}  "
          f"fleiss_k={overall['fleiss_kappa']}  full-agreement={overall['full_agreement_rate']:.2%}")
    print("Pairwise Cohen's kappa:")
    for k, v in pairs.items():
        print(f"  {k}: {v}")

    # Per domain
    per_domain = {}
    for dom in sorted(set(item_domain.values())):
        ids = [i for i in item_ids if item_domain[i] == dom]
        per_domain[dom] = summarize_subset(ids, per_annot, dom)
        per_domain[dom]["pairwise_cohen_kappa"] = pairwise_cohen(ids, per_annot)
        print(f"  {dom}: alpha={per_domain[dom]['krippendorff_alpha']}  "
              f"fleiss_k={per_domain[dom]['fleiss_kappa']}  "
              f"full-agreement={per_domain[dom]['full_agreement_rate']:.2%}")

    # Majority-vote adjudication
    mv = majority_vote(item_ids, per_annot)
    ties = [iid for iid, v in mv.items() if v["tie"]]
    print(f"\nMajority-vote: {sum(1 for v in mv.values() if v['label'])} resolved, "
          f"{len(ties)} ties (no consensus): {ties}")

    # Compare to author labels
    auth = author_labels(item_ids)
    match = sum(1 for iid in item_ids if auth.get(iid) and mv[iid]["label"] == auth[iid])
    n_compare = sum(1 for iid in item_ids if auth.get(iid))
    print(f"\nAuthor labels available for {n_compare}/{len(item_ids)} items.")
    print(f"Author label == majority vote: {match}/{n_compare} ({100*match/max(n_compare,1):.0f}%)")
    mismatches = {iid: {"author": auth.get(iid), "majority": mv[iid]["label"],
                        "votes": mv[iid]["votes"]}
                  for iid in item_ids
                  if auth.get(iid) and mv[iid]["label"] and mv[iid]["label"] != auth[iid]}
    print(f"Author/majority mismatches: {len(mismatches)} items")
    for iid, info in list(mismatches.items())[:10]:
        print(f"  {iid}: author={info['author']} | majority={info['majority']} | votes={info['votes']}")

    # Save outputs
    iir = {
        "n_annotators": len(records),
        "annotator_ids": sorted(records),
        "n_items": len(item_ids),
        "categories": list(CATEGORIES),
        "overall": overall,
        "overall_pairwise_cohen_kappa": pairs,
        "per_domain": per_domain,
        "interpretation": {
            "krippendorff_alpha_bands": {
                "≥0.80": "strong agreement",
                "0.67–0.80": "acceptable (NLP/HCI norm)",
                "<0.67": "labels problematic"
            },
            "note": "Ambiguous items by design split annotator votes; per-domain α gives the cleanest picture."
        }
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "iir_scores.json").write_text(json.dumps(iir, indent=2))
    print(f"\nSaved: {RESULTS_DIR / 'iir_scores.json'}")

    aggregate = {
        "n_annotators": len(records),
        "annotator_ids": sorted(records),
        "n_items": len(item_ids),
        "adjudication_rule": "majority of N annotators; ties (no consensus) labelled null",
        "per_item": {
            iid: {
                **mv[iid],
                "domain": item_domain[iid],
                "author_label": auth.get(iid),
            }
            for iid in item_ids
        },
    }
    (ANNOTATORS_DIR / "aggregate_labels.json").write_text(json.dumps(aggregate, indent=2))
    print(f"Saved: {ANNOTATORS_DIR / 'aggregate_labels.json'}")


if __name__ == "__main__":
    main()
