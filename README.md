# AMEL: Accumulated Message Effects on LLM Judgments

This repository contains the code, data, and analysis for our study on how conversation history systematically distorts sequential binary judgments in LLM evaluation pipelines. We test 12 models from 5 providers across 90K+ API calls (84,088 in the main deduplicated experiment plus ~6.4K from the mitigation, temperature, and mechanistic experiments).

## Key Findings

- **AMEL is cross-provider** (d = -0.17, p < 10^-54, N = 84,088 main-experiment API calls after dedup; 10/12 models significant after Bonferroni; bootstrap 95% CI on d: [-0.19, -0.15])
- **Uncertainty predicts susceptibility**: items where the model is genuinely uncertain at baseline (nonzero binary entropy) absorb roughly twice the bias of confident-baseline items (d = -0.34 vs d = -0.15)
- **Two regimes**: assimilation for congruent items (model conforms when context matches item ground truth), resistance/anchoring for incongruent items (model shifts away from context when item contradicts it); difference d = 0.56
- **Negativity asymmetry**: paired per-item ratio 1.52x (t = 13.03, p < 10^-37, n = 2,733 pairs); marginal-means ratio is ~2.2x but mixes item composition
- **No accumulation**: 5 turns of biased history produce the same effect as 50 (Spearman |r| < 0.01; linear-slope OLS p = 0.80)
- **Scaling reduces but doesn't eliminate**: Haiku d=-0.22 > Sonnet d=-0.18 > Opus d=-0.17
- **Temperature doesn't help**: lower temperature trends toward stronger bias, not weaker
- **Balanced ordering mitigates drift**: interleaving expected-yes/no items prevents positional drift in sequential evaluation
- **External 5-annotator IRR validation**: Krippendorff α = 0.53 overall; per-domain α = 0.28 (code), 0.62 (content), 0.70 (meals); full per-annotator ratings + codebook released under data/annotators/ (CC-BY-4.0)
- **Consistency rate**: 84.7% of (model, item, polarity, context-length) cells have the same modal answer under treatment as baseline; lowest in code review (70.7%)

### Characterization Experiments (Section 5)

- **Logprobs** (1 model, 1 domain, 1,050 calls): the probability distribution shifts continuously, not just binary flips
- **Flipped framing** (2 models, 1,260 calls): negativity asymmetry has both token-level and semantic components; per-model attribution is directional but not significant at this sample size
- **Positional placement** (2 models, 1,260 calls): START ≈ END ≈ SPREAD, position of biased turns is irrelevant (KW H=0.19, p=0.91)
- **Baseline correlation**: items with higher baseline P(no) show stronger negativity asymmetry (Spearman r=0.08, n.s.; Pearson r=0.22, p<0.001)

## Repository Structure

```
.
├── paper/                      # Manuscript (LaTeX)
│   ├── main.tex                # Full paper source
│   ├── main.pdf                # Compiled manuscript
│   └── references.bib          # Bibliography (34 references)
│
├── src/                        # Experiment framework
│   ├── config.py               # Experimental parameters
│   ├── conversation.py         # Context construction (polarity + positional)
│   ├── parser.py               # Original yes/no parser (kept for archival reference)
│   ├── parser_v2.py            # Symmetric yes/no parser (canonical; used for the published numbers)
│   ├── runner.py               # Async experiment runner (Ollama)
│   └── domains/                # Evaluation domain definitions
│       ├── base.py             # Abstract domain interface
│       ├── code_review.py      # "Is this code production-ready?"
│       ├── code_review_flipped.py # "Should this code be rejected?" (flipped framing)
│       ├── content_mod.py      # "Is this comment appropriate?"
│       └── meals.py            # "Is this a healthy choice?"
│
├── run_experiment.py           # Main CLI (local models via Ollama)
├── run_openai.py               # OpenAI GPT-4.1 Nano runner
├── run_openai_5_2.py           # OpenAI GPT-5.2 runner
├── run_claude.py               # Anthropic Claude runner
├── run_gemini.py               # Google Gemini runner
├── run_deepseek.py             # DeepSeek runner (added in paper v2)
├── run_mitigation.py           # Sequential batch mitigation experiment
├── run_temperature.py          # Temperature sensitivity experiment
├── run_logprobs.py             # Logprobs mechanistic experiment (Phase 1)
├── run_flipped.py              # Flipped framing experiment (Phase 2)
├── run_positional.py           # Positional placement experiment (Phase 3)
│
├── analysis/                   # Statistical analysis
│   ├── utils.py                # Shared utilities (load_results, compute_bias_scores)
│   ├── analyze.py              # Core analysis functions
│   ├── paper_statistics.py     # Comprehensive stats for paper
│   ├── contrast_assimilation.py # Congruent vs incongruent bias analysis
│   ├── continuous_confidence.py # Baseline entropy vs bias susceptibility
│   ├── mixed_effects.py        # Mixed-effects model (BS ~ polarity * category | model)
│   ├── response_time.py        # Response latency analysis
│   ├── qualitative_examples.py # Top biased items with raw responses
│   ├── mitigation_analysis.py  # Sequential batch experiment analysis
│   ├── temperature_analysis.py # Temperature sensitivity analysis
│   ├── asymmetry_baseline_corr.py # Baseline P(no) vs asymmetry correlation
│   ├── logprobs_analysis.py    # First-token probability analysis
│   ├── flipped_analysis.py     # Original vs flipped framing comparison
│   └── positional_analysis.py  # START/END/SPREAD placement analysis
│
├── generate_paper_figures.py   # Publication figure generation (14 figures)
│
├── data/
│   ├── all_results.jsonl       # Main experiment dataset (84,088 deduplicated responses; see scripts/dedupe_qwen30b.py for the dedup procedure)
│   ├── annotators/             # External 5-annotator IRR ratings + codebook (CC-BY-4.0; see data/annotators/README.md)
│   ├── mitigation/             # Sequential batch experiment (3,780 responses)
│   ├── temperature/            # Temperature spot-check (840 responses)
│   ├── logprobs/               # Logprobs experiment (1,050 responses)
│   ├── flipped/                # Flipped framing experiment (1,260 responses)
│   ├── positional/             # Positional placement experiment (1,260 responses)
│   ├── raw/                    # Local models (Llama, Qwen) via Ollama
│   ├── openai/                 # GPT-4.1 Nano results
│   ├── openai-gpt52/           # GPT-5.2 results
│   ├── claude-haiku-4-5/       # Claude Haiku 4.5 results
│   ├── claude-sonnet-4-6/      # Claude Sonnet 4.6 results
│   ├── claude-opus-4-6/        # Claude Opus 4.6 results
│   ├── gemini-flash/           # Gemini 2.5 Flash results
│   ├── gemini-pro/             # Gemini 2.5 Pro results
│   └── deepseek-v3/            # DeepSeek V4 Flash results (added in paper v2)
│
└── results/
    ├── paper_figures/          # Figures 0-13 (PDF + PNG, 14 total)
    ├── paper_statistics.json   # Main experiment statistics
    ├── asymmetry_baseline_corr.json
    ├── logprobs_analysis.json
    ├── flipped_analysis.json
    ├── positional_analysis.json
    ├── contrast_assimilation.json
    ├── continuous_confidence.json
    ├── mixed_effects.json
    ├── response_time.json
    ├── qualitative_examples.json
    ├── mitigation_analysis.json
    └── temperature_analysis.json
```

## Experimental Design

We use a between-subjects design with four conditions per test item:

| Condition | Context | Description |
|-----------|---------|-------------|
| **Baseline** | None | Test item presented after system prompt only |
| **No-saturated** | 90% "no" | N turns of predominantly negative evaluations |
| **Yes-saturated** | 90% "yes" | N turns of predominantly positive evaluations |
| **Neutral** | 50/50 | N turns of balanced evaluations |

Each condition is repeated 10 times at temperature T=1.0 across context lengths N = {5, 10, 20, 50}.

## Models Tested

| Provider | Model | Cohen's d |
|----------|-------|-----------|
| OpenAI | GPT-4.1 Nano | -0.34 |
| OpenAI | GPT-5.2 | -0.17 |
| Anthropic | Claude Haiku 4.5 | -0.22 |
| Anthropic | Claude Sonnet 4.6 | -0.18 |
| Anthropic | Claude Opus 4.6 | -0.17 |
| Google | Gemini 2.5 Flash | -0.18 |
| Google | Gemini 2.5 Pro | -0.27 |
| Local | Llama 3.2 3B | -0.32 |
| Local | Qwen3 4B | +0.19 (contrarian) |
| Local | Qwen3.5 4B | -0.08 (n.s.) |
| Local | Qwen3 30B | +0.10 (n.s.) |

Negative d = model shifts toward the saturated polarity (conforming); positive d = model shifts away (contrarian). See paper Table 2 for full per-model statistics.

## Reproducing Results

### Prerequisites

```bash
pip install -r requirements.txt
```

For local models, install [Ollama](https://ollama.ai) and pull the required models.

### Running Experiments

```bash
# For bit-exact seed reproduction across processes, fix the Python hash seed.
export PYTHONHASHSEED=0

# Local models (Ollama)
python run_experiment.py run

# API models (set environment variables first)
export OPENAI_API_KEY="..."
python run_openai.py
python run_openai_5_2.py

export ANTHROPIC_API_KEY="..."
python run_claude.py

export GEMINI_API_KEY="..."
python run_gemini.py

export DEEPSEEK_API_KEY="..."
python run_deepseek.py

# Mitigation experiment (sequential batch)
python run_mitigation.py

# Temperature sensitivity
python run_temperature.py

# Mechanistic experiments
python run_logprobs.py      # Logprobs (OpenAI only)
python run_flipped.py       # Flipped framing (OpenAI + Ollama)
python run_positional.py    # Positional placement (OpenAI + Ollama)
```

### Analysis

```bash
# Generate comprehensive statistics
python -m analysis.paper_statistics

# Run all supplementary analyses
python -m analysis.contrast_assimilation
python -m analysis.continuous_confidence
python -m analysis.mixed_effects
python -m analysis.response_time
python -m analysis.qualitative_examples
python -m analysis.mitigation_analysis
python -m analysis.temperature_analysis

# Mechanistic analyses
python -m analysis.asymmetry_baseline_corr
python -m analysis.logprobs_analysis
python -m analysis.flipped_analysis
python -m analysis.positional_analysis

# Generate paper figures (0-13, 14 total)
python generate_paper_figures.py
```

### Building the Paper

```bash
cd paper
tectonic main.tex
# or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Data Format

Each line in `data/all_results.jsonl` is a JSON object:

```json
{
  "domain": "code_review",
  "model": "claude-sonnet-4-6",
  "polarity": "no_saturated",
  "context_length": 10,
  "test_item_id": "test_code_amb_03",
  "test_item_text": "<test item content>",
  "test_item_category": "ambiguous",
  "test_item_ground_truth": "yes",
  "repetition": 3,
  "raw_response": "<full model output>",
  "parsed_response": "no",
  "response_time_ms": 2146.46,
  "num_context_turns": 10,
  "num_messages": 21,
  "seed": 3941144273,
  "timestamp": "2026-03-16T12:41:40.789808+00:00"
}
```

See `data/README.md` for a per-field reference.

## Citation

```bibtex
@article{temkit2026amel,
  title={AMEL: Accumulated Message Effects on LLM Judgments},
  author={Temkit, Sid-Ali},
  year={2026}
}
```

## License

This research is released under the MIT License. The dataset is released under CC-BY 4.0.
