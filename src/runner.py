"""Async experiment runner using Ollama API."""

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from src.config import (
    CONCURRENCY,
    CONTEXT_LENGTHS,
    MAX_TOKENS,
    MODELS,
    OLLAMA_BASE_URL,
    POLARITIES,
    REPETITIONS,
    TEMPERATURE,
    ExperimentCondition,
)
from src.conversation import build_messages
from src.domains import ALL_DOMAINS
from src.domains.base import BaseDomain
from src.parser import parse_yes_no

# Concurrency limits per model size to avoid OOM / thrashing
MODEL_CONCURRENCY = {
    "llama3.3:70b": 1,      # 42GB — run one at a time
    "qwen3:30b": 2,         # 18GB — careful
    "qwen3-coder:30b": 2,
    "qwen3:4b": 4,
    "qwen3.5:4b": 4,
    "llama3.2:3b": 4,
}


@dataclass
class ExperimentResult:
    """Result of a single experimental run."""

    domain: str
    model: str
    polarity: str
    context_length: int
    test_item_id: str
    test_item_text: str
    test_item_category: str
    test_item_ground_truth: str
    repetition: int
    raw_response: str
    parsed_response: str | None
    response_time_ms: float
    num_context_turns: int
    num_messages: int          # total messages sent (context + test)
    seed: int
    timestamp: str             # ISO timestamp of when the call was made


async def call_ollama(
    client: httpx.AsyncClient,
    model: str,
    messages: list[dict],
    semaphore: asyncio.Semaphore,
) -> tuple[str, float, int]:
    """Call Ollama chat API and return (response_text, time_ms, num_messages)."""
    async with semaphore:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,  # Disable thinking mode (qwen3)
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": MAX_TOKENS,
            },
        }

        start = time.perf_counter()
        try:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=600.0,  # 10 min timeout for large models + long contexts
            )
            resp.raise_for_status()
            elapsed_ms = (time.perf_counter() - start) * 1000
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            return content, elapsed_ms, len(messages)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return f"ERROR: {e}", elapsed_ms, len(messages)


def generate_conditions(
    domains: list[str] | None = None,
    models: list[str] | None = None,
    context_lengths: list[int] | None = None,
    polarities: list[str] | None = None,
    repetitions: int | None = None,
) -> list[tuple[BaseDomain, str, str, int, "Item", int]]:
    """Generate all experiment conditions.

    Returns list of (domain, model, polarity, context_length, test_item, repetition).
    """
    _domains = {k: v for k, v in ALL_DOMAINS.items() if not domains or k in domains}
    _models = models or MODELS
    _context_lengths = context_lengths or CONTEXT_LENGTHS
    _polarities = polarities or POLARITIES
    _repetitions = repetitions or REPETITIONS

    conditions = []

    for domain_name, domain in _domains.items():
        test_items = domain.get_test_items()
        for model in _models:
            for test_item in test_items:
                # Baseline condition (fresh conversation)
                for rep in range(_repetitions):
                    conditions.append((domain, model, "baseline", 0, test_item, rep))

                # Treatment conditions
                for polarity in _polarities:
                    for ctx_len in _context_lengths:
                        for rep in range(_repetitions):
                            conditions.append(
                                (domain, model, polarity, ctx_len, test_item, rep)
                            )

    return conditions


async def run_experiment(
    output_dir: Path,
    domains: list[str] | None = None,
    models: list[str] | None = None,
    context_lengths: list[int] | None = None,
    polarities: list[str] | None = None,
    repetitions: int | None = None,
    concurrency: int | None = None,
    resume: bool = True,
) -> Path:
    """Run the full experiment.

    Args:
        output_dir: Directory to write results.
        domains: List of domain names to test (None = all).
        models: List of model names (None = all from config).
        context_lengths: List of context lengths (None = all from config).
        polarities: List of polarities (None = all from config).
        repetitions: Number of repetitions (None = from config).
        concurrency: Max concurrent API calls override (None = auto per model).
        resume: If True, skip conditions already in the output file.

    Returns:
        Path to the results JSONL file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.jsonl"

    # Save experiment metadata
    metadata = {
        "experiment": "context_bias",
        "start_time": datetime.now(timezone.utc).isoformat(),
        "config": {
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "context_lengths": context_lengths or CONTEXT_LENGTHS,
            "polarities": polarities or POLARITIES,
            "repetitions": repetitions or REPETITIONS,
            "models": models or MODELS,
            "domains": domains or list(ALL_DOMAINS.keys()),
        },
    }
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # Load existing results for resume
    existing_keys: set[str] = set()
    if resume and results_file.exists():
        with open(results_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    key = _result_key(r)
                    existing_keys.add(key)
        print(f"Resuming: found {len(existing_keys)} existing results")

    conditions = generate_conditions(
        domains=domains,
        models=models,
        context_lengths=context_lengths,
        polarities=polarities,
        repetitions=repetitions,
    )

    # Filter out already-completed conditions
    remaining = []
    for cond in conditions:
        domain, model, polarity, ctx_len, test_item, rep = cond
        key = f"{domain.name}|{model}|{polarity}|{ctx_len}|{test_item.id}|{rep}"
        if key not in existing_keys:
            remaining.append(cond)

    total = len(conditions)
    done = len(existing_keys)
    remaining_count = len(remaining)

    print(f"Total conditions: {total}")
    print(f"Already done: {done}")
    print(f"Remaining: {remaining_count}")

    if remaining_count == 0:
        print("All conditions already completed!")
        return results_file

    # Group by model to avoid switching models too often (Ollama loads one at a time)
    by_model: dict[str, list] = {}
    for cond in remaining:
        model = cond[1]
        by_model.setdefault(model, []).append(cond)

    # Sort models: fastest first (small models first)
    model_order = ["llama3.2:3b", "qwen3:4b", "qwen3.5:4b", "qwen3:30b", "llama3.3:70b"]
    sorted_models = sorted(
        by_model.keys(),
        key=lambda m: model_order.index(m) if m in model_order else 99,
    )

    completed = done
    experiment_start = time.perf_counter()

    async with httpx.AsyncClient() as client:
        for model in sorted_models:
            model_conditions = by_model[model]
            model_start = time.perf_counter()

            # Use model-specific concurrency unless overridden
            _concurrency = concurrency or MODEL_CONCURRENCY.get(model, CONCURRENCY)
            semaphore = asyncio.Semaphore(_concurrency)

            print(f"\n{'='*60}")
            print(f"Model: {model} ({len(model_conditions)} conditions, concurrency={_concurrency})")
            print(f"{'='*60}")

            # Pre-load the model
            print(f"Pre-loading {model}...")
            await _preload_model(client, model)
            print(f"Model loaded.")

            # Shuffle conditions within model to avoid systematic ordering effects
            import random
            random.Random(42).shuffle(model_conditions)

            # Process in batches
            batch_size = _concurrency * 2
            for batch_start in range(0, len(model_conditions), batch_size):
                batch = model_conditions[batch_start : batch_start + batch_size]

                tasks = []
                for domain, _, polarity, ctx_len, test_item, rep in batch:
                    seed = hash(f"{domain.name}|{polarity}|{ctx_len}|{test_item.id}|{rep}") & 0xFFFFFFFF
                    messages = build_messages(domain, polarity, ctx_len, test_item, seed)

                    tasks.append(
                        _run_single(
                            client=client,
                            semaphore=semaphore,
                            model=model,
                            domain=domain,
                            polarity=polarity,
                            ctx_len=ctx_len,
                            test_item=test_item,
                            rep=rep,
                            messages=messages,
                            seed=seed,
                        )
                    )

                results = await asyncio.gather(*tasks)

                # Write results atomically (append)
                with open(results_file, "a") as f:
                    for result in results:
                        f.write(json.dumps(asdict(result)) + "\n")

                completed += len(results)
                pct = (completed / total) * 100

                # ETA calculation
                elapsed = time.perf_counter() - experiment_start
                if completed > done:
                    rate = (completed - done) / elapsed
                    remaining_time = (total - completed) / rate if rate > 0 else 0
                    eta_h = remaining_time / 3600
                    print(
                        f"  Progress: {completed}/{total} ({pct:.1f}%) | "
                        f"ETA: {eta_h:.1f}h | "
                        f"Rate: {rate:.1f} calls/s"
                    )
                else:
                    print(f"  Progress: {completed}/{total} ({pct:.1f}%)")

            model_elapsed = time.perf_counter() - model_start
            print(f"Model {model} complete in {model_elapsed/60:.1f} minutes")

    total_elapsed = time.perf_counter() - experiment_start
    print(f"\nExperiment complete in {total_elapsed/3600:.1f} hours!")
    print(f"Results: {results_file}")

    # Update metadata with end time
    metadata["end_time"] = datetime.now(timezone.utc).isoformat()
    metadata["total_duration_seconds"] = total_elapsed
    metadata["total_conditions_run"] = completed - done
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    return results_file


async def _preload_model(client: httpx.AsyncClient, model: str) -> None:
    """Send a dummy request to load the model into memory."""
    try:
        await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
                "think": False,
                "options": {"num_predict": 5},
            },
            timeout=300.0,
        )
    except Exception as e:
        print(f"  Warning: preload failed: {e}")


async def _run_single(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    model: str,
    domain: BaseDomain,
    polarity: str,
    ctx_len: int,
    test_item,
    rep: int,
    messages: list[dict],
    seed: int,
) -> ExperimentResult:
    """Run a single condition and return the result."""
    raw_response, response_time, num_messages = await call_ollama(
        client, model, messages, semaphore
    )
    parsed = parse_yes_no(raw_response)

    return ExperimentResult(
        domain=domain.name,
        model=model,
        polarity=polarity,
        context_length=ctx_len,
        test_item_id=test_item.id,
        test_item_text=test_item.text[:500],  # generous truncation for paper review
        test_item_category=test_item.category,
        test_item_ground_truth=test_item.ground_truth,
        repetition=rep,
        raw_response=raw_response[:2000],  # keep full response for manual review
        parsed_response=parsed,
        response_time_ms=response_time,
        num_context_turns=ctx_len,
        num_messages=num_messages,
        seed=seed,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _result_key(result: dict) -> str:
    """Generate a unique key for a result to support resume."""
    return (
        f"{result['domain']}|{result['model']}|{result['polarity']}|"
        f"{result['context_length']}|{result['test_item_id']}|{result['repetition']}"
    )
