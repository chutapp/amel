"""Run context bias experiment against Google Gemini API models."""

import asyncio
import json
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))

from src.config import CONTEXT_LENGTHS, MAX_TOKENS, POLARITIES, REPETITIONS, TEMPERATURE
from src.conversation import build_messages
from src.domains import ALL_DOMAINS
from src.parser import parse_yes_no
from src.runner import ExperimentResult, _result_key

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Model configs: (model_id, concurrency, thinking_config)
MODEL_CONFIGS = {
    "gemini-2.5-flash": {
        "concurrency": 15,
        "thinking": {"thinkingBudget": 0},  # disable thinking for fair comparison
    },
    "gemini-2.5-pro": {
        "concurrency": 2,  # Pro has strict rate limits
        "thinking": {"thinkingBudget": 256},  # minimum thinking (can't disable)
    },
}


def convert_messages_to_gemini(messages: list[dict]) -> tuple[dict | None, list[dict]]:
    """Convert OpenAI-format messages to Gemini format.

    Returns (system_instruction, contents).
    """
    system_instruction = None
    contents = []

    for msg in messages:
        if msg["role"] == "system":
            system_instruction = {"parts": [{"text": msg["content"]}]}
        elif msg["role"] == "user":
            contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
        elif msg["role"] == "assistant":
            contents.append({"role": "model", "parts": [{"text": msg["content"]}]})

    return system_instruction, contents


async def call_gemini(
    client: httpx.AsyncClient,
    model: str,
    messages: list[dict],
    semaphore: asyncio.Semaphore,
    thinking_config: dict,
) -> tuple[str, float, int]:
    """Call Gemini generateContent API."""
    async with semaphore:
        system_instruction, contents = convert_messages_to_gemini(messages)

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": TEMPERATURE,
                "maxOutputTokens": MAX_TOKENS + 300,  # extra headroom for thinking models
                "thinkingConfig": thinking_config,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        url = f"{GEMINI_BASE_URL}/models/{model}:generateContent?key={GEMINI_API_KEY}"

        start = time.perf_counter()
        try:
            resp = await client.post(url, json=payload, timeout=120.0)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("retry-after", "10"))
                print(f"  Rate limited on {model}, waiting {retry_after}s...")
                await asyncio.sleep(retry_after)
                return await call_gemini(client, model, messages, semaphore, thinking_config)

            resp.raise_for_status()
            data = resp.json()

            candidates = data.get("candidates", [])
            if not candidates:
                return "ERROR: no candidates", elapsed_ms, len(messages)

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])

            # Extract non-thought text
            text = ""
            for part in parts:
                if "text" in part and not part.get("thought"):
                    text = part["text"]
                    break

            if not text:
                return "ERROR: empty response", elapsed_ms, len(messages)

            return text, elapsed_ms, len(messages)

        except httpx.HTTPStatusError as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return f"ERROR: {e.response.status_code} {e.response.text[:500]}", elapsed_ms, len(messages)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return f"ERROR: {e}", elapsed_ms, len(messages)


async def run_single(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    model: str,
    domain,
    polarity: str,
    ctx_len: int,
    test_item,
    rep: int,
    messages: list[dict],
    seed: int,
    thinking_config: dict,
) -> ExperimentResult:
    raw_response, response_time, num_messages = await call_gemini(
        client, model, messages, semaphore, thinking_config
    )
    parsed = parse_yes_no(raw_response)

    return ExperimentResult(
        domain=domain.name,
        model=model,
        polarity=polarity,
        context_length=ctx_len,
        test_item_id=test_item.id,
        test_item_text=test_item.text[:500],
        test_item_category=test_item.category,
        test_item_ground_truth=test_item.ground_truth,
        repetition=rep,
        raw_response=raw_response[:2000],
        parsed_response=parsed,
        response_time_ms=response_time,
        num_context_turns=ctx_len,
        num_messages=num_messages,
        seed=seed,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


async def run_model(model: str, config: dict):
    """Run experiment for a single Gemini model."""
    output_dir = Path(f"data/gemini-{model.split('-')[-1]}")  # e.g. data/gemini-flash, data/gemini-pro
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.jsonl"
    log_file = output_dir / "experiment.log"

    metadata = {
        "experiment": f"context_bias_{model}",
        "start_time": datetime.now(timezone.utc).isoformat(),
        "config": {
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "context_lengths": CONTEXT_LENGTHS,
            "polarities": POLARITIES,
            "repetitions": REPETITIONS,
            "models": [model],
            "domains": list(ALL_DOMAINS.keys()),
            "thinking_config": config["thinking"],
        },
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Resume
    existing_keys: set[str] = set()
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing_keys.add(_result_key(r))
        print(f"[{model}] Resuming: {len(existing_keys)} existing results")

    # Generate conditions
    conditions = []
    for domain_name, domain in ALL_DOMAINS.items():
        test_items = domain.get_test_items()
        for test_item in test_items:
            for rep in range(REPETITIONS):
                conditions.append((domain, model, "baseline", 0, test_item, rep))
            for polarity in POLARITIES:
                for ctx_len in CONTEXT_LENGTHS:
                    for rep in range(REPETITIONS):
                        conditions.append((domain, model, polarity, ctx_len, test_item, rep))

    remaining = []
    for cond in conditions:
        domain, m, polarity, ctx_len, test_item, rep = cond
        key = f"{domain.name}|{m}|{polarity}|{ctx_len}|{test_item.id}|{rep}"
        if key not in existing_keys:
            remaining.append(cond)

    total = len(conditions)
    done = len(existing_keys)
    print(f"[{model}] Total: {total}, Done: {done}, Remaining: {len(remaining)}")

    if not remaining:
        print(f"[{model}] All done!")
        return

    random.Random(42).shuffle(remaining)

    concurrency = config["concurrency"]
    thinking_config = config["thinking"]
    semaphore = asyncio.Semaphore(concurrency)
    completed = done
    experiment_start = time.perf_counter()

    async with httpx.AsyncClient() as client:
        batch_size = concurrency * 2
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start : batch_start + batch_size]

            tasks = []
            for domain, m, polarity, ctx_len, test_item, rep in batch:
                seed = hash(f"{domain.name}|{polarity}|{ctx_len}|{test_item.id}|{rep}") & 0xFFFFFFFF
                messages = build_messages(domain, polarity, ctx_len, test_item, seed)
                tasks.append(
                    run_single(client, semaphore, m, domain, polarity, ctx_len, test_item, rep, messages, seed, thinking_config)
                )

            results = await asyncio.gather(*tasks)

            with open(results_file, "a") as f:
                for result in results:
                    f.write(json.dumps(asdict(result)) + "\n")

            completed += len(results)
            pct = (completed / total) * 100
            elapsed = time.perf_counter() - experiment_start
            if completed > done:
                rate = (completed - done) / elapsed
                remaining_time = (total - completed) / rate if rate > 0 else 0
                eta_min = remaining_time / 60
                log_line = f"  [{model}] Progress: {completed}/{total} ({pct:.1f}%) | ETA: {eta_min:.1f}m | Rate: {rate:.1f} calls/s"
                print(log_line)
                with open(log_file, "a") as f:
                    f.write(log_line + "\n")

    total_elapsed = time.perf_counter() - experiment_start
    print(f"\n[{model}] Complete in {total_elapsed/60:.1f} minutes!")

    metadata["end_time"] = datetime.now(timezone.utc).isoformat()
    metadata["total_duration_seconds"] = total_elapsed
    metadata["total_conditions_run"] = completed - done
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


async def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else list(MODEL_CONFIGS.keys())
    print(f"Running Gemini experiment for: {models}")

    # Run models sequentially to avoid rate limit conflicts
    for model in models:
        if model not in MODEL_CONFIGS:
            print(f"Unknown model: {model}")
            continue
        await run_model(model, MODEL_CONFIGS[model])


if __name__ == "__main__":
    asyncio.run(main())
