"""Positional placement experiment (Phase 3).

Tests whether bias depends on WHERE biased turns appear in the conversation.
5 biased turns placed at different positions within a 50-turn conversation.

Placements:
- START: positions 0-4 biased, rest neutral
- END: positions 45-49 biased, rest neutral
- SPREAD: positions 0, 12, 24, 36, 49 biased, rest neutral

GPT-4.1 Nano + llama3.2:3b, code_review, no_saturated only, 21 items, 10 reps.
630 calls per model → 1,260 total.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))

from src.config import MAX_TOKENS, OLLAMA_BASE_URL
from src.conversation import build_messages_positional
from src.domains import ALL_DOMAINS
from src.parser import parse_yes_no

DOMAIN_NAME = "code_review"
POLARITY = "no_saturated"
TOTAL_TURNS = 50
REPETITIONS = 10

PLACEMENTS = {
    "START": list(range(5)),              # positions 0-4
    "END": list(range(45, 50)),           # positions 45-49
    "SPREAD": [0, 12, 24, 36, 49],       # evenly distributed
}

OPENAI_MODEL = "gpt-4.1-nano"
OLLAMA_MODEL = "llama3.2:3b"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_CONCURRENCY = 15
OLLAMA_CONCURRENCY = 4


async def call_openai(client, messages, semaphore):
    """Call OpenAI API."""
    async with semaphore:
        start = time.perf_counter()
        try:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": OPENAI_MODEL,
                    "messages": messages,
                    "temperature": 1.0,
                    "max_tokens": MAX_TOKENS,
                },
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            elapsed_ms = (time.perf_counter() - start) * 1000
            content = resp.json()["choices"][0]["message"]["content"]
            return content, elapsed_ms
        except httpx.HTTPStatusError as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            if e.response.status_code == 429:
                retry_after = float(e.response.headers.get("retry-after", "5"))
                print(f"  Rate limited, waiting {retry_after}s...")
                await asyncio.sleep(retry_after)
                return await call_openai(client, messages, semaphore)
            return f"ERROR: {e.response.status_code}", elapsed_ms
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return f"ERROR: {e}", elapsed_ms


async def call_ollama(client, messages, semaphore):
    """Call Ollama API."""
    async with semaphore:
        start = time.perf_counter()
        try:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": 1.0,
                        "num_predict": MAX_TOKENS,
                    },
                },
                timeout=600.0,
            )
            resp.raise_for_status()
            elapsed_ms = (time.perf_counter() - start) * 1000
            content = resp.json().get("message", {}).get("content", "")
            return content, elapsed_ms
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return f"ERROR: {e}", elapsed_ms


async def preload_ollama(client):
    """Pre-load model into Ollama memory."""
    try:
        await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
                "think": False,
                "options": {"num_predict": 5},
            },
            timeout=300.0,
        )
    except Exception as e:
        print(f"  Warning: preload failed: {e}")


async def run_model(model_name, call_fn, concurrency, output_dir):
    """Run all placement conditions for one model."""
    results_file = output_dir / "results.jsonl"
    domain = ALL_DOMAINS[DOMAIN_NAME]
    test_items = domain.get_test_items()

    # Load existing for resume
    existing_keys = set()
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("model") == model_name:
                        key = f"{r['placement']}|{r['test_item_id']}|{r['repetition']}"
                        existing_keys.add(key)
        print(f"  Resuming {model_name}: {len(existing_keys)} existing results")

    conditions = []
    for placement_name, positions in PLACEMENTS.items():
        for item in test_items:
            for rep in range(REPETITIONS):
                key = f"{placement_name}|{item.id}|{rep}"
                if key not in existing_keys:
                    seed = hash(f"{DOMAIN_NAME}|{POLARITY}|{placement_name}|{item.id}|{rep}") & 0xFFFFFFFF
                    conditions.append((placement_name, positions, item, rep, seed))

    print(f"  {model_name}: {len(conditions)} remaining conditions")
    if not conditions:
        return

    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    start_time = time.perf_counter()

    async with httpx.AsyncClient() as client:
        if model_name == OLLAMA_MODEL:
            print(f"  Pre-loading {model_name}...")
            await preload_ollama(client)

        batch_size = concurrency * 2
        for batch_start in range(0, len(conditions), batch_size):
            batch = conditions[batch_start:batch_start + batch_size]

            async def run_one(placement_name, positions, item, rep, seed):
                messages = build_messages_positional(
                    domain, POLARITY, TOTAL_TURNS, positions, item, seed
                )
                content, elapsed_ms = await call_fn(client, messages, semaphore)
                parsed = parse_yes_no(content)
                return {
                    "domain": DOMAIN_NAME,
                    "model": model_name,
                    "placement": placement_name,
                    "polarity": POLARITY,
                    "total_turns": TOTAL_TURNS,
                    "n_biased": len(positions),
                    "biased_positions": positions,
                    "test_item_id": item.id,
                    "test_item_text": item.text[:500],
                    "test_item_category": item.category,
                    "test_item_ground_truth": item.ground_truth,
                    "repetition": rep,
                    "raw_response": content[:2000] if isinstance(content, str) else "",
                    "parsed_response": parsed,
                    "response_time_ms": elapsed_ms,
                    "seed": seed,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            tasks = [run_one(*c) for c in batch]
            results = await asyncio.gather(*tasks)

            with open(results_file, "a") as f:
                for result in results:
                    f.write(json.dumps(result) + "\n")

            completed += len(results)
            elapsed = time.perf_counter() - start_time
            print(f"    {model_name}: {completed}/{len(conditions)} | {completed/elapsed:.1f} calls/s")


async def main():
    output_dir = Path("data/positional")
    output_dir.mkdir(parents=True, exist_ok=True)

    # OpenAI
    if OPENAI_API_KEY:
        print(f"\n{'='*60}")
        print(f"Running positional experiment: {OPENAI_MODEL}")
        print(f"{'='*60}")
        await run_model(OPENAI_MODEL, call_openai, OPENAI_CONCURRENCY, output_dir)
    else:
        print("WARNING: OPENAI_API_KEY not set, skipping OpenAI")

    # Ollama
    print(f"\n{'='*60}")
    print(f"Running positional experiment: {OLLAMA_MODEL}")
    print(f"{'='*60}")
    await run_model(OLLAMA_MODEL, call_ollama, OLLAMA_CONCURRENCY, output_dir)

    print(f"\nPositional experiment complete! Results: {output_dir / 'results.jsonl'}")


if __name__ == "__main__":
    asyncio.run(main())
