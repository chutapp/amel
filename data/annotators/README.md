# AMEL annotator dataset

External inter-rater validation for the test items used in the AMEL
study (Temkit, 2026). Five annotators independently rated each of the
63 test items into one of three categories (`clear_positive`,
`ambiguous`, `clear_negative`) using the codebook in this folder.

This directory is the publicly released annotation dataset and is
designed to be reusable by other researchers.

## Files

| File | What it is |
|---|---|
| `anon_A.json` ... `anon_E.json` | Per-annotator pseudonymised rating files. Each row records the rating, an optional free-text comment, and per-item seconds-on-screen. |
| `aggregate_labels.json` | Majority-vote label per item across the five annotators, with the per-category vote counts and a `tie` flag (set when no clear majority emerges, i.e. a 2-2-1 split). Items with `tie: true` are excluded from the primary per-category analysis in the paper. |
| `codebook.md` | The same codebook the annotators received before rating. Two-three worked examples per category per domain plus an FAQ section. |
| `../../results/iir_scores.json` | Krippendorff's α (nominal level), Fleiss' κ, full-agreement rate, and all pairwise Cohen's κ values, overall and per domain. |

## Methodology summary

- **Recruitment.** Five annotators recruited via Upwork in May 2026
  with prior NLP / CS / data-science experience (filter: ≥80 % job-success
  score). All five signed digital consent allowing publication and
  public dataset release.
- **Demographics.** Four annotators in Pakistan, one in Serbia;
  three data scientists / ML engineers, two academic researchers. All
  reported "fluent (working proficiency)" English (full per-annotator
  demographics fields are kept inside the individual `anon_*.json` files).
- **Compensation.** $50 per annotator (~$40 / hour for a ~75-minute
  task), well above platform minimums.
- **Order.** Items presented in a per-annotator deterministic
  shuffled order (seeded by annotator name) to control for ordering
  effects within rater.
- **Quality control.** Each rating carries `seconds_spent`. The
  per-item timing distribution was checked for AI-shortcut signals
  (impossibly fast cells; uniform pacing). Comments were scanned for
  LLM-style vocabulary tells; none of the four annotators showed
  evidence of AI-assisted rating.
- **No AI tools.** Annotators explicitly consented to rating without
  ChatGPT / Claude / other LLM assistance. The form recorded this
  consent.

## Inter-rater agreement (results/iir_scores.json)

| Domain | Krippendorff's α | Fleiss' κ | Full 5-way agreement |
|---|---|---|---|
| Overall | 0.53 | 0.53 | 41 % |
| Code review | 0.28 | 0.27 | 19 % |
| Content moderation | 0.62 | 0.61 | 48 % |
| Meals (nutrition) | **0.70** | **0.69** | 57 % |

Interpretation: meals crosses the conventional NLP / HCI threshold
(α ≥ 0.67); content moderation is moderate; code review is the
lowest-agreement domain and quantitatively confirms that
"production-ready" is a genuinely contested judgment, even among
experienced annotators. Per-item votes are released so other
researchers can apply alternative adjudication rules and recompute α
themselves.

## Adjudication rule (paper §3.2)

The primary per-category analysis in the paper uses the **majority
vote of the five annotators** as the item's category. Items with no
clear majority (a 2-2-1 split — the only kind of tie that can occur
with five raters on three categories) are marked `tie: true` and are
not assigned a category for that analysis; they are included in all
analyses that do not depend on per-category labels (overall effect
size, polarity asymmetry, accumulation, contrast / assimilation,
empirical-entropy stratification).

## Reusability

These ratings can be reused for any study of:

- Inter-annotator reliability on binary subjective judgments
- LLM-judge bias (the items themselves are public research stimuli)
- Crowd-annotation methodology

Cite the paper for attribution:

```bibtex
@article{temkit2026amel,
  title  = {{AMEL}: Accumulated Message Effects on {LLM} Judgments},
  author = {Temkit, Sid-Ali},
  year   = {2026},
  journal = {arXiv preprint arXiv:2605.22714}
}
```

The dataset itself is released under CC-BY-4.0 (see `../LICENSE-CC-BY-4.0`).

## Provenance

The annotation form (`annotator_package/annotate.html`), the codebook,
and the deterministic item-shuffle seed are kept in version control
in the public AMEL repository so the data-collection setup is
reproducible:

  https://github.com/chutapp/amel
