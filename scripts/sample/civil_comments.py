"""Sample 21 items from Civil Comments (Borkan et al., WWW 2019).

Dataset on HuggingFace as `google/civil_comments`. Each row has a
toxicity score in [0, 1] computed as the fraction of annotators (at
least 10 per row) who labelled the comment toxic. We use this both as
the mean label and as the basis for the annotator-agreement metric:
agreement = max(toxicity, 1 - toxicity), the fraction of annotators
agreeing on the majority verdict.

We treat the binary judgment as ``Is this comment INAPPROPRIATE?'':

    clear_yes  (inappropriate)   = toxicity >= 0.80  & agreement >= 0.85
    ambiguous                    = 0.40 <= toxicity <= 0.60 & agreement <= 0.60
    clear_no   (appropriate)     = toxicity <= 0.05  & agreement >= 0.95

The framing matches the v1 content_moderation domain prompt so the
prompts in src/domains/content_mod.py can be regenerated with the
same question template.

Run:  python -m scripts.sample.civil_comments
Output: data/validated_samples/civil_comments.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets import load_dataset  # noqa: E402

from scripts.sample.common import (  # noqa: E402
    BIN_SIZE,
    BinSpec,
    Item,
    SEED,
    print_bin_summary,
    save_sample,
    stratify_random,
)


CORPUS = "civil_comments"
DATASET_ID = "google/civil_comments"
SPLIT = "train"  # has ~1.8M rows with toxicity scores
N_ANNOTATORS_DEFAULT = 10  # google/civil_comments documents ≥10 per row


def main() -> None:
    print(f"Loading {DATASET_ID}:{SPLIT}...")
    ds = load_dataset(DATASET_ID, split=SPLIT)
    print(f"  {len(ds):,} rows")

    # Keep only fields we need to keep memory small.
    pool_clear_yes: list[Item] = []
    pool_ambiguous: list[Item] = []
    pool_clear_no: list[Item] = []

    for ix, row in enumerate(ds):
        tox = float(row["toxicity"])
        agreement = max(tox, 1.0 - tox)
        text = row["text"].strip()
        # Skip very long or empty
        if not text or len(text) > 600:
            continue
        item = Item(
            item_id=f"cc_{ix}",
            source_id=str(ix),
            text=text,
            mean_label=tox,
            agreement=agreement,
            n_annotators=N_ANNOTATORS_DEFAULT,
            metadata={
                "severe_toxicity": float(row.get("severe_toxicity", 0.0)),
                "identity_attack": float(row.get("identity_attack", 0.0)),
                "insult": float(row.get("insult", 0.0)),
                "threat": float(row.get("threat", 0.0)),
            },
        )
        if tox >= 0.80 and agreement >= 0.85:
            pool_clear_yes.append(item)
        elif 0.40 <= tox <= 0.60 and agreement <= 0.60:
            pool_ambiguous.append(item)
        elif tox <= 0.05 and agreement >= 0.95:
            pool_clear_no.append(item)

        # We do not need millions of items per bin; cap pools to keep memory
        # bounded and the script fast.
        if (
            len(pool_clear_yes) >= 2000
            and len(pool_ambiguous) >= 2000
            and len(pool_clear_no) >= 2000
        ):
            print(f"  reached pool caps after {ix:,} rows; stopping scan")
            break

    print(
        f"  pool sizes: clear_yes={len(pool_clear_yes)} "
        f"ambiguous={len(pool_ambiguous)} clear_no={len(pool_clear_no)}"
    )

    bins = [
        BinSpec(
            name="clear_yes",
            rule="toxicity >= 0.80 AND agreement >= 0.85 (inappropriate)",
            items=stratify_random(pool_clear_yes, BIN_SIZE, seed=SEED + 1),
        ),
        BinSpec(
            name="ambiguous",
            rule="0.40 <= toxicity <= 0.60 AND agreement <= 0.60",
            items=stratify_random(pool_ambiguous, BIN_SIZE, seed=SEED + 2),
        ),
        BinSpec(
            name="clear_no",
            rule="toxicity <= 0.05 AND agreement >= 0.95 (appropriate)",
            items=stratify_random(pool_clear_no, BIN_SIZE, seed=SEED + 3),
        ),
    ]
    print_bin_summary(bins)
    out = save_sample(CORPUS, version=f"hf:{DATASET_ID}@{SPLIT}", bins=bins)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
