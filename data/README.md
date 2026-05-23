# AMEL dataset

Experimental data for "AMEL: Accumulated Message Effects on LLM Judgments."

## License

This dataset is released under the **Creative Commons Attribution 4.0
International License (CC-BY-4.0)**. See `LICENSE-CC-BY-4.0` in this directory
or `../NOTICE` at the repository root.

## Contents

- `all_results.jsonl` — Main experiment dataset, **75,898 deduplicated rows**
  spanning 11 models, 3 evaluation domains, 4 polarity conditions, and 4
  context lengths (5, 10, 20, 50 turns), plus baselines. Each row is one API
  call.
- `mitigation/` — Sequential batch mitigation experiment (3,780 rows).
- `temperature/` — Temperature spot-check (840 rows at T ∈ {0.3, 0.7}; the
  T=1.0 cells come from the main experiment).
- `logprobs/` — First-token logprobs experiment, GPT-4.1 Nano, code review
  (1,050 rows).
- `flipped/` — Flipped-framing experiment, GPT-4.1 Nano + Llama 3.2 3B
  (1,260 rows).
- `positional/` — Positional-placement experiment, GPT-4.1 Nano +
  Llama 3.2 3B (1,260 rows).
- Per-model subdirectories (`openai/`, `claude-haiku-4-5/`, etc.) hold the
  raw per-provider response logs that are concatenated into
  `all_results.jsonl`.

## Schema (one JSON object per line)

```json
{
  "domain": "code_review",
  "model": "claude-sonnet-4-6",
  "polarity": "no_saturated",
  "context_length": 10,
  "test_item_id": "test_code_amb_03",
  "test_item_category": "ambiguous",
  "test_item_ground_truth": "yes",
  "repetition": 3,
  "raw_response": "<full model output>",
  "parsed_response": "no",
  "response_time_ms": 2146.46,
  "seed": 3941144273,
  "timestamp": "2026-03-16T12:41:40.789808+00:00"
}
```

Field reference:

| Field | Type | Description |
|---|---|---|
| `domain` | string | One of `code_review`, `content_moderation`, `meals`. |
| `model` | string | Exact API model identifier (see Table 1 in the paper). |
| `polarity` | string | `baseline`, `no_saturated`, `yes_saturated`, or `neutral`. |
| `context_length` | int | Number of turns in the conversation history. `0` for baseline. |
| `test_item_id` | string | Stable item identifier; see `src/domains/*.py`. |
| `test_item_category` | string | `clear_positive`, `ambiguous`, or `clear_negative`. |
| `test_item_ground_truth` | string | `yes` or `no` — see paper Section 3.2 for the ambiguous-item coding convention. |
| `repetition` | int | 0–9 (10 reps per condition). |
| `raw_response` | string | Full model output text. |
| `parsed_response` | string | `yes`, `no`, or `unparseable`. Parser logic in `src/parser.py`. |
| `response_time_ms` | float | API round-trip time, including network. |
| `seed` | int | Per-condition deterministic seed; see paper Section 3.1. |
| `timestamp` | string | ISO 8601 UTC. |

## Deduplication note (Qwen3 30B)

During the local-model batch on 2026-03-16, the run experienced a disk-full
crash mid-way through `qwen3:30b`. The resume script was inadvertently
launched twice concurrently against the same output file. Because Python
randomizes `PYTHONHASHSEED` per process, the two interpreters generated
different per-condition seeds and produced two independent samples for the
overlapping conditions: **2,186 duplicate-condition rows** for `qwen3:30b`
(distinguishable by their `seed` field).

The shipped `all_results.jsonl` keeps the first occurrence per
`(domain, polarity, context_length, test_item_id, repetition)` key for
`qwen3:30b` only; all other models pass through unchanged. The dedup script
is `../scripts/dedupe_qwen30b.py`; re-running it on the released dataset is
a no-op. The pre-dedup raw file is not shipped (it adds 126 MB and contains
only duplicate rows beyond what is already published).

For `qwen3:30b` specifically, the headline statistic moves from
$\bar{BS} = +0.029$ ($d = +0.08$, $n.s.$ after Bonferroni) to
$\bar{BS} = +0.019$ ($d = +0.05$, $n.s.$ before or after correction);
direction unchanged, magnitude slightly smaller. Headline cross-model
statistics in the paper are unaffected to the displayed precision. See the
paper's Appendix on dataset deduplication for details.

## Citation

```bibtex
@article{temkit2026amel,
  title  = {{AMEL}: Accumulated Message Effects on {LLM} Judgments},
  author = {Temkit, Sid-Ali},
  year   = {2026}
}
```
