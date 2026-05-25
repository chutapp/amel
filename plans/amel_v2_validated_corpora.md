# AMEL v2 plan — replace hand-built items with validated corpora

**Goal:** kill the "single-author categorization" reviewer concern by
replacing all 63 hand-authored test items with samples from three
peer-reviewed, multi-annotated, binary-task datasets. Re-run the full
experiment pipeline. Publish as arXiv v2.

**Decision (2026-05-23, with professor input):** use validated corpora
for all three domains, not just content moderation. The three-bin
split (clear / ambiguous / clear) is sampled from inter-annotator
agreement instead of author judgment.

---

## Dataset choices per domain

| Current domain | Replacement dataset | Source | Why |
|---|---|---|---|
| content_moderation | **Civil Comments** | Borkan et al., WWW 2019. HuggingFace `civil_comments`. | Toxicity score per comment + multi-annotator labels (10+ raters per item). De-facto gold standard for online-moderation eval; used in Perspective API training and dozens of LLM-bias papers. |
| code_review | **CodeReviewer dataset** | Li et al., 2022, arXiv:2203.09095. | 1.18M real GitHub code-change pairs with merge/reject decisions. Binary, validated by reality (got merged or not). Inter-reviewer disagreement available for ambiguous-bin sampling. |
| meals (nutrition) | **SciFact** | Wadden et al., EMNLP 2020. arXiv:2004.14974. | 1.4K scientific claims with binary supports/refutes labels + multi-annotator agreement. Binary, peer-reviewed, well-used in NLP fact-checking benchmarks. Replaces nutrition entirely. |

Rationale for SciFact over keeping nutrition:
- No widely-validated binary "healthy/not healthy" corpus exists
- SciFact preserves the "domain breadth" justification of the paper
  (technical / social / scientific) while bringing it into a
  validated regime
- SciFact's claim structure ("X causes Y") naturally extends the
  binary-judgment task: model says yes (supports) or no (refutes)

---

## Three-bin sampling rules

For each dataset, replace the author label with annotator-agreement strata:

- **Clear-positive** (n=7): items where ≥80% of annotators agree the
  answer is "yes" AND the mean label is ≥ 0.8
- **Ambiguous** (n=7): items where annotator agreement is in the
  40-60% band (genuine disagreement) AND mean label is 0.4-0.6
- **Clear-negative** (n=7): items where ≥80% of annotators agree on
  "no" AND the mean label is ≤ 0.2

Sample stratified-random within each bin. Use a fixed seed so the
sample is reproducible. Record the sampling script + the resulting
21 items per domain in `data/validated_samples/`.

For ambiguous items, replace the existing "ground-truth = yes" coding
convention with the actual mean annotator score (or report both
directions in analysis). This also resolves the §3.2 reviewer concern
about ambiguous-item coding.

---

## Execution steps

1. **Pull datasets**
   - Civil Comments via HuggingFace `datasets.load_dataset('civil_comments')`
   - CodeReviewer via the official repo (microsoft/CodeBERT)
   - SciFact via HuggingFace `datasets.load_dataset('allenai/scifact')`

2. **Sample 21 items per domain into the 3 bins** with a written
   sampling script (`scripts/sample_validated_items.py`). Seed = 20260523.

3. **Replace domain item lists**
   - `src/domains/content_mod.py` → Civil Comments samples
   - `src/domains/code_review.py` → CodeReviewer samples
   - `src/domains/meals.py` → rename to `src/domains/scifact.py`, swap
     items + adjust question template to "Does evidence support this
     claim?"
   - Update `src/config.py` and `run_*.py` to register the new domain.

4. **Re-run the main experiment** across all 11 models for all 3
   replaced domains. Estimated cost: comparable to the original
   experiment (~$300-800 in API + a few days wall-clock; Gemini Pro
   and Opus dominate).

5. **Re-run the side experiments**
   - Logprobs (GPT-4.1 Nano, code review only): re-run on new CodeReviewer items
   - Flipped framing (Nano + Llama 3.2 3B): re-run on new CodeReviewer items
   - Positional (Nano + Llama, code review only): re-run on new items
   - Mitigation (3 domains): re-run on all 3 new sets
   - Temperature (Nano, code review only): re-run on new items

6. **Re-run analyses**
   - `paper_statistics.py`
   - `contrast_assimilation.py`
   - `continuous_confidence.py`
   - `mixed_effects.py`
   - `response_time.py`
   - `qualitative_examples.py`
   - `entropy_stratified.py`
   - `unparseable_by_condition.py`
   - `accumulation_slope.py`
   - `asymmetry_baseline_corr.py`

7. **Regenerate 14 figures** via `generate_paper_figures.py`.

8. **Update paper**
   - §3.2 Evaluation Domains: describe the three validated sources
     and the agreement-based 3-bin sampling
   - References.bib: add Borkan et al. 2019, Li et al. 2022,
     Wadden et al. 2020
   - Limitations: remove the "single-author categorization" caveat;
     replace with "we sample from existing validated corpora, so
     conclusions inherit the framing of those corpora"
   - Future work: remove the inter-rater item
   - Update all numbers throughout

9. **Submit arXiv v2** with note: "v2 (substantial revision): all three
   evaluation domains replaced with samples from peer-reviewed
   multi-annotated corpora (Civil Comments, CodeReviewer, SciFact);
   three-bin item split is now data-driven rather than author-coded;
   analysis pipeline and per-domain numbers updated accordingly."

---

## Cost / time estimate

| Step | Wall-clock | API cost |
|---|---|---|
| Sample + swap items | 2-4 hours | $0 |
| Re-run main experiment | 2-3 days | $300-600 |
| Re-run side experiments | 1 day | $100-200 |
| Re-run analyses + figures | 1 hour | $0 |
| Update paper | 4-8 hours | $0 |
| **Total** | **4-5 days** | **$400-800** |

---

## Open questions to resolve before execution

1. Do you want to swap **all 3** domains in one v2, or do it
   incrementally (Civil Comments first, then v3 adds CodeReviewer +
   SciFact)? Incremental is safer; one-shot is more efficient.
2. SciFact replaces "meals" entirely. Do you want a brief paragraph
   in the paper acknowledging the domain shift (nutritional → fact-
   checking) and what it means for generalizability?
3. Budget approval for the $400-800 API spend?

---

## Added 2026-05-23 (professor critique #2): API vs OSS comparison

**Critique:** closed APIs (GPT, Claude, Gemini) include unknown infrastructure
layers on top of the raw model — hidden system prompts, safety filters,
output post-processing, possible routing across model variants. Local
Ollama models do not. So a direct "GPT-5.2 vs Llama 3.2 3B" comparison
conflates the model with the platform.

The critique applies to causal / mechanistic claims, not to the
descriptive claim AMEL actually makes ("LLMs in production exhibit
history bias"). But the paper currently presents API and OSS models in
one panel as if they were directly comparable, which invites the
critique.

### Changes to make in v2

1. **Reorganize Figure 2 and Table 2.** Show API models and OSS models
   as separate sub-panels (or separate rows in the table). Caption
   notes the platform-layer difference.
2. **Update §4.2 wording.** Drop any sentence that compares a closed
   API model to a local OSS model on equal footing. Within-family
   scaling claims (Haiku → Sonnet → Opus, Nano → GPT-5.2) are safe
   because the platform is constant.
3. **Add a Limitations bullet:** "Closed APIs include unobservable
   infrastructure layers (system prompts, safety filters, output
   post-processing, possible model-version routing) that we cannot
   inspect. Comparisons across the API / OSS boundary therefore
   conflate model effects with platform effects. Within-provider
   scaling comparisons are unaffected."
4. **Add to Future Work:** replicate the OSS-model runs through a
   hosted inference API (Together, Fireworks, Replicate) so the
   HTTP-layer treatment matches the closed APIs. This would isolate
   the model effect from the platform effect.

### Cost of the change

- Items 1-3 are paper edits only. ~3-4 hours.
- Item 4 (rerun OSS through hosted APIs) is optional: ~$50-100 +
  half a day if you want a fully apples-to-apples version.

---

## Added 2026-05-23 (professor critique #3): add DeepSeek and Mistral

**Critique:** the model panel is missing two important families
(DeepSeek, Mistral) that are widely deployed in production.

### Recommended additions (4 models, count 11 → 15)

| Model | Access | Tier | Cost estimate | Why |
|---|---|---|---|---|
| **DeepSeek-V3** | `api.deepseek.com` (OpenAI-compatible) | Flagship (671B MoE) | ~$10-20 for full run | Cheapest top-tier API; closes the "missing Chinese frontier model" gap |
| **Mistral Large 3** | `api.mistral.ai` | Flagship | ~$40-80 | Largest closed Mistral; representative European provider |
| **Mistral Small 3** | `api.mistral.ai` | Small (24B) | ~$10-20 | Within-family scaling pair vs Large 3 |
| **Mixtral 8x7B** | local via Ollama | Open-weight | $0 (compute time only) | Bonus: same family available as both API and OSS — directly addresses the API/OSS-conflate critique in §critique #2 above |

### Models intentionally skipped

- **DeepSeek-R1** (reasoning model): emits long chain-of-thought
  output that would not fit cleanly into a binary parser; skip.
- **DeepSeek-V2-Lite**: older, less interesting.
- **Mistral Medium 3**: redundant with Small + Large for scaling
  evidence; skip unless we want a 3-point ladder.
- **Codestral / Pixtral**: code-specific / multimodal; not relevant
  to AMEL's binary-judgment design.

### Changes to the experiment pipeline

1. Add 4 runners: `run_deepseek.py`, `run_mistral.py` (handles
   Large 3 + Small 3 via the same script). Mixtral 8x7B fits the
   existing local-Ollama runner.
2. Re-run the main experiment for the 4 new models on the 3
   validated-corpus domains.
3. Re-run the relevant side experiments. Logprobs / flipped /
   positional currently use only GPT-4.1 Nano + Llama 3.2 3B —
   skip those (don't expand model panel for side experiments
   unless we want to grow the paper).
4. Update Figure 2 / Table 2 / Table 1 / §3.3 (Models Tested)
   with the four new entries.

### Cost / time addition

- ~$60-120 in API for the 3 closed-API runs (DeepSeek + Mistral L3 + Mistral S3)
- ~0 for Mixtral 8x7B (local) but needs a machine that can fit
  the model in memory (~80 GB GPU for fp16, or quantized via Ollama)
- +1-2 days wall-clock added to the existing redo budget

### Open question for this addition

5. Do you have a machine that can run Mixtral 8x7B locally via
   Ollama? If not, drop it and we add only DeepSeek-V3 + Mistral
   Large 3 + Mistral Small 3 (3 models, no API/OSS-conflate bonus).

---

## Confirmations (2026-05-24)

- Anthropic / OpenAI / DeepSeek / Mistral API keys provided + stored in
  Keychain (`*.amel.api-key`)
- Gemini key (`google.gemini.api-key`) still valid from v1
- **Mistral Small 3 → local Ollama only** (`mistral-small:24b`,
  already pulled); no API duplicate
- **Mixtral 8x7B → local Ollama**, `ollama pull mixtral:8x7b` in
  flight at start
- API calls run **in parallel** (async httpx already in v1 runner
  pattern; new runners follow the same pattern)
- Final model panel (13 models):
  - OpenAI: GPT-4.1 Nano, GPT-5.2
  - Anthropic: Haiku 4.5, Sonnet 4.6, Opus 4.6
  - Google: Gemini 2.5 Flash, Gemini 2.5 Pro
  - DeepSeek: DeepSeek-V3 (new)
  - Mistral: Mistral Large 3 API (new)
  - Local (Ollama): Mistral Small 3 (new), Mixtral 8x7B (new),
    Llama 3.2 3B (v1), Qwen3 30B (v1)

## Council update 2026-05-24 — corpora swap REJECTED

Three independent council agents (data-science, llm-eval, peer-review) all
concluded that the validated-corpora swap (Civil Comments + CodeReviewer
+ SciFact) is the wrong remedy for the professor's "single-author
categorization" concern. Reasons:

- **CodeReviewer has no multi-annotator agreement.** The "3-bin from
  agreement" trick structurally cannot work; any ambiguous bin
  reintroduces author-coded selection with extra steps.
- **Civil Comments items at toxicity≈0 or 1 produce deterministic
  model responses** — they land in §4.9's $B_1$ bin where AMEL
  barely operates. Annotator disagreement ≠ model uncertainty.
- **SciFact has the wrong task shape** (claim verification with
  evidence). Doesn't fit coherently as prior turns in a binary-
  judgment conversation.
- **None of the three datasets are the convention** for LLM-judge bias
  studies. The actual community conventions are LLMBar, JudgeBench,
  MT-Bench, BiGGen-Bench, CodeJudgeBench.
- **The paper already partly defuses the concern.** §4.9 empirical-
  entropy stratification ($d = -0.34$ on $B_3$ vs $-0.15$ on $B_1$)
  is grounded in *model behaviour*, not author labels.

### Recommended path instead (cheaper + stronger)

1. **Inter-rater validation on the existing 63 items** by 2 external
   annotators + Krippendorff's α. Directly answers "single-author
   categorization". Cost: ~$200-300, 2-3 days. Need to recruit 2
   annotators with appropriate domain knowledge for each domain
   (code review needs a software engineer, content mod can be any
   adult, nutrition can be any adult).

2. **Add LLMBar (Zeng et al. 2023, arXiv:2310.07641) as a robustness
   appendix.** LLMBar was designed specifically to expose LLM-as-judge
   biases; sampling 21 items from it for a robustness check carries
   real weight in this community. Cost: ~$100 in API + 1 day.

3. **Promote §4.9 to a first-class result.** Move the empirical-
   entropy finding ($d = -0.34$ on $B_3$, $d = -0.15$ on $B_1$) into
   the headline of the abstract and Section 4. Frame the author-
   coded "ambiguous" category as a secondary stratifier that the
   entropy result subsumes. This is mostly a paper-edit cost (a
   few hours).

**New cost / time estimate:**

| Step | Wall-clock | Cost |
|---|---|---|
| Recruit + run 2 annotators on 63 items | 2-3 days | $200-300 |
| Add LLMBar robustness appendix | 1 day | $100 |
| Promote §4.9 in paper | 4-6 hours | $0 |
| Paper edits + tarball + push + v2 | 1 day | $0 |
| **Total** | **4-5 days** | **$300-400** |

Compare to original plan: $400-800 + 5-7 days + risk of headline
shifting + reviewer attack on "you relabeled samples from datasets
built for other tasks".

### Still applies from earlier critique rounds

- **API/OSS reframe (critique #2)** — separate Figure 2 / Table 2
  panels, restrict scaling claims to within-provider, add
  Limitations bullet. Still valid; cheap to do.
- **DeepSeek + Mistral additions (critique #3)** — still valid
  if you want broader provider coverage. Independent of the corpora
  question. Worth doing only if you want the wider panel; not
  required to answer the single-author concern.

## Status

- **Plan saved:** 2026-05-23.
- **Council review 2026-05-24:** corpora swap rejected; switched to
  IRR + LLMBar + §4.9 promotion path.
- **Open questions** before execution:
  1. Recruit 2 annotators (who?)
  2. LLMBar API budget approved (~$100)?
  3. Still want DeepSeek + Mistral additions? (separate from the
     corpora question)
  4. Still want API/OSS reframe? (paper edit only, cheap)
