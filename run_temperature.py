"""Temperature spot-check experiment.

Tests whether temperature affects bias magnitude.
claude-sonnet-4-6 only, code_review domain only, T=0.3 and T=0.7.
no_saturated + baseline, context_length=10, all 21 items, 10 reps.
~840 API calls.
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))

from src.seed_guard import require_hashseed
require_hashseed()

from src.config import MAX_TOKENS
from src.conversation import build_messages
from src.domains import ALL_DOMAINS
from src.parser import parse_yes_no
from src.runner import ExperimentResult

MODEL = "gpt-4.1-nano"
DOMAIN = "code_review"
TEMPERATURES = [0.3, 0.7]
POLARITIES = ["no_saturated", "baseline"]
CONTEXT_LENGTH = 10
REPETITIONS = 10

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
CONCURRENCY = 15


async def call_openai(client, messages, temperature, semaphore):
    """Call OpenAI API with specified temperature."""
    async with semaphore:
        start = time.perf_counter()
        try:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": MODEL, "messages": messages,
                    "temperature": temperature, "max_tokens": MAX_TOKENS,
                },
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            elapsed_ms = (time.perf_counter() - start) * 1000
            return content, elapsed_ms
        except httpx.HTTPStatusError as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            if e.response.status_code == 429:
                retry_after = float(e.response.headers.get("retry-after", "5"))
                print(f"  Rate limited, waiting {retry_after}s...")
                await asyncio.sleep(retry_after)
                return await call_openai(client, messages, temperature, semaphore)
            return f"ERROR: {e.response.status_code}", elapsed_ms
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return f"ERROR: {e}", elapsed_ms


async def main():
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    output_dir = Path("data/temperature")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.jsonl"

    domain = ALL_DOMAINS[DOMAIN]
    test_items = domain.get_test_items()

    # Load existing for resume
    existing_keys = set()
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    key = f"{r.get('temperature', 1.0)}|{r['polarity']}|{r['test_item_id']}|{r['repetition']}"
                    existing_keys.add(key)
        print(f"Resuming: {len(existing_keys)} existing results")

    conditions = []
    for temp in TEMPERATURES:
        for polarity in POLARITIES:
            ctx_len = CONTEXT_LENGTH if polarity != "baseline" else 0
            for item in test_items:
                for rep in range(REPETITIONS):
                    key = f"{temp}|{polarity}|{item.id}|{rep}"
                    if key not in existing_keys:
                        seed = hash(f"{DOMAIN}|{polarity}|{ctx_len}|{item.id}|{rep}") & 0xFFFFFFFF
                        conditions.append((temp, polarity, ctx_len, item, rep, seed))

    print(f"Remaining conditions: {len(conditions)}")
    if not conditions:
        print("All conditions complete!")
        return

    semaphore = asyncio.Semaphore(CONCURRENCY)
    completed = 0
    start_time = time.perf_counter()

    async with httpx.AsyncClient() as client:
        batch_size = CONCURRENCY * 2
        for batch_start in range(0, len(conditions), batch_size):
            batch = conditions[batch_start:batch_start + batch_size]

            async def run_one(temp, polarity, ctx_len, item, rep, seed):
                messages = build_messages(domain, polarity, ctx_len, item, seed)
                content, elapsed_ms = await call_openai(client, messages, temp, semaphore)
                parsed = parse_yes_no(content)
                result = {
                    "domain": DOMAIN,
                    "model": MODEL,
                    "temperature": temp,
                    "polarity": polarity,
                    "context_length": ctx_len,
                    "test_item_id": item.id,
                    "test_item_text": item.text[:500],
                    "test_item_category": item.category,
                    "test_item_ground_truth": item.ground_truth,
                    "repetition": rep,
                    "raw_response": content[:2000],
                    "parsed_response": parsed,
                    "response_time_ms": elapsed_ms,
                    "num_context_turns": ctx_len,
                    "seed": seed,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                return result

            tasks = [run_one(*c) for c in batch]
            results = await asyncio.gather(*tasks)

            with open(results_file, "a") as f:
                for result in results:
                    f.write(json.dumps(result) + "\n")

            completed += len(results)
            elapsed = time.perf_counter() - start_time
            print(f"  Progress: {completed}/{len(conditions)} | {completed/elapsed:.1f} calls/s")

    print(f"\nTemperature experiment complete! Results: {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
