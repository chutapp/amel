"""Shared types + helpers for stratified validated-corpus sampling.

For each validated corpus (Civil Comments, CodeReviewer, SciFact) we sample
21 items into three bins driven by inter-annotator agreement:

    B_clear_yes   (n=7): high mean label, high annotator agreement
    B_ambiguous   (n=7): mid-range mean label, low annotator agreement
    B_clear_no    (n=7): low mean label, high annotator agreement

Each sample carries provenance metadata so the file is reproducible by
re-running the same script with the same seed.

Output schema (one JSON file per corpus):

    {
        "corpus":       "civil_comments",
        "version":      "<hf dataset version or commit>",
        "sampled_at":   "2026-05-24T...",
        "seed":         20260524,
        "bins": {
            "clear_yes":  {"rule": "...", "items": [...]},
            "ambiguous":  {"rule": "...", "items": [...]},
            "clear_no":   {"rule": "...", "items": [...]}
        }
    }

Each item:

    {
        "item_id":          "<corpus>_<bin>_<ix>",
        "source_id":        "<dataset row id>",
        "text":             "<the prompt content>",
        "mean_label":       0.0..1.0,
        "agreement":        0.0..1.0,    # fraction of annotators picking the majority class
        "n_annotators":     <int>,
        "metadata":         {...}        # corpus-specific extras
    }
"""
from __future__ import annotations

import datetime as _dt
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "validated_samples"
SEED = 20260524
BIN_SIZE = 7


@dataclass(frozen=True)
class Item:
    item_id: str
    source_id: str
    text: str
    mean_label: float
    agreement: float
    n_annotators: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BinSpec:
    name: str
    rule: str
    items: list[Item]


def stratify_random(pool: list[Item], n: int, seed: int) -> list[Item]:
    rng = random.Random(seed)
    if len(pool) < n:
        raise ValueError(f"pool has only {len(pool)} items, need {n}")
    return rng.sample(pool, n)


def save_sample(
    corpus: str,
    version: str,
    bins: Iterable[BinSpec],
    output_path: Path | None = None,
) -> Path:
    out = output_path or (OUT_DIR / f"{corpus}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "corpus": corpus,
        "version": version,
        "sampled_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "bin_size": BIN_SIZE,
        "bins": {
            b.name: {
                "rule": b.rule,
                "items": [asdict(it) for it in b.items],
            }
            for b in bins
        },
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return out


def print_bin_summary(bins: Iterable[BinSpec]) -> None:
    for b in bins:
        means = [it.mean_label for it in b.items]
        agreements = [it.agreement for it in b.items]
        if means:
            print(
                f"  {b.name:14s} n={len(b.items):2d}  "
                f"mean_label={sum(means) / len(means):.3f}  "
                f"agreement={sum(agreements) / len(agreements):.3f}"
            )
