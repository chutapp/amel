"""Experiment configuration."""

from dataclasses import dataclass, field

OLLAMA_BASE_URL = "http://localhost:11434"

# Models to test (local via Ollama) — matches the 4 local models reported in the paper.
MODELS = [
    "qwen3:30b",
    "qwen3:4b",
    "qwen3.5:4b",
    "llama3.2:3b",
]

# Context lengths to test (number of prior turns)
CONTEXT_LENGTHS = [5, 10, 20, 50]

# Context polarities
POLARITIES = ["no_saturated", "yes_saturated", "neutral"]

# Number of repetitions per condition
REPETITIONS = 10

# Ollama concurrency (how many parallel requests)
CONCURRENCY = 4

# Temperature for all runs
TEMPERATURE = 1.0

# Max tokens for response
MAX_TOKENS = 100


@dataclass
class ExperimentCondition:
    """A single experimental condition to run."""

    domain: str
    model: str
    polarity: str  # "no_saturated", "yes_saturated", "neutral", "baseline"
    context_length: int  # 0 for baseline
    test_item_id: str
    test_item_text: str
    test_item_category: str  # "clear_positive", "ambiguous", "clear_negative"
    repetition: int
    context_items: list[dict] = field(default_factory=list)
    system_prompt: str = ""
