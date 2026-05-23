"""Logprobs experiment (Phase 1).

Tests whether the probability distribution shifts continuously under
AMEL, not just binary flips.

GPT-4.1 Nano, code_review domain, 21 items.
5 conditions: baseline, no_sat@5, yes_sat@5, no_sat@50, yes_sat@50.
10 reps each → 1,050 calls.
OpenAI API with logprobs=True, top_logprobs=5.
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

from src.config import MAX_TOKENS
from src.conversation import build_messages
from src.domains import ALL_DOMAINS
from src.parser import parse_yes_no

MODEL = "gpt-4.1-nano"
DOMAIN = "code_review"
CONDITIONS = [
    ("baseline", 0),
    ("no_saturated", 5),
    ("yes_saturated", 5),
    ("no_saturated", 50),
    ("yes_saturated", 50),
]
REPETITIONS = 10
CONCURRENCY = 15

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


async def call_openai_logprobs(client, messages, semaphore):
    """Call OpenAI API with logprobs enabled."""
    async with semaphore:
        start = time.perf_counter()
        try:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": MODEL,
                    "messages": messages,
                    "temperature": 1.0,
                    "max_tokens": MAX_TOKENS,
                    "logprobs": True,
                    "top_logprobs": 5,
                },
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            elapsed_ms = (time.perf_counter() - start) * 1000
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            logprobs_data = data["choices"][0].get("logprobs", {})
            return content, elapsed_ms, logprobs_data
        except httpx.HTTPStatusError as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            if e.response.status_code == 429:
                retry_after = float(e.response.headers.get("retry-after", "5"))
                print(f"  Rate limited, waiting {retry_after}s...")
                await asyncio.sleep(retry_after)
                return await call_openai_logprobs(client, messages, semaphore)
            return f"ERROR: {e.response.status_code}", elapsed_ms, {}
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return f"ERROR: {e}", elapsed_ms, {}


def extract_first_token_probs(logprobs_data):
    """Extract P(Yes) and P(No) from first token logprobs."""
    content_logprobs = logprobs_data.get("content", [])
    if not content_logprobs:
        return None, None

    first_token = content_logprobs[0]
    top_logprobs = first_token.get("top_logprobs", [])

    import math
    p_yes = 0.0
    p_no = 0.0

    for entry in top_logprobs:
        token = entry["token"].strip().lower()
        prob = math.exp(entry["logprob"])
        if token in ("yes", "yeah", "absolutely", "definitely"):
            p_yes += prob
        elif token in ("no", "nope", "not"):
            p_no += prob

    return p_yes, p_no


async def main():
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    output_dir = Path("data/logprobs")
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
                    key = f"{r['polarity']}|{r['context_length']}|{r['test_item_id']}|{r['repetition']}"
                    existing_keys.add(key)
        print(f"Resuming: {len(existing_keys)} existing results")

    conditions = []
    for polarity, ctx_len in CONDITIONS:
        for item in test_items:
            for rep in range(REPETITIONS):
                key = f"{polarity}|{ctx_len}|{item.id}|{rep}"
                if key not in existing_keys:
                    seed = hash(f"{DOMAIN}|{polarity}|{ctx_len}|{item.id}|{rep}") & 0xFFFFFFFF
                    conditions.append((polarity, ctx_len, item, rep, seed))

    print(f"Total conditions: {len(CONDITIONS) * len(test_items) * REPETITIONS}")
    print(f"Remaining: {len(conditions)}")

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

            async def run_one(polarity, ctx_len, item, rep, seed):
                messages = build_messages(domain, polarity, ctx_len, item, seed)
                content, elapsed_ms, logprobs_data = await call_openai_logprobs(
                    client, messages, semaphore
                )
                parsed = parse_yes_no(content)
                p_yes, p_no = extract_first_token_probs(logprobs_data)

                return {
                    "domain": DOMAIN,
                    "model": MODEL,
                    "polarity": polarity,
                    "context_length": ctx_len,
                    "test_item_id": item.id,
                    "test_item_text": item.text[:500],
                    "test_item_category": item.category,
                    "test_item_ground_truth": item.ground_truth,
                    "repetition": rep,
                    "raw_response": content[:2000] if isinstance(content, str) else "",
                    "parsed_response": parsed,
                    "p_yes": p_yes,
                    "p_no": p_no,
                    "first_token_logprobs": (
                        logprobs_data.get("content", [{}])[0].get("top_logprobs", [])
                        if logprobs_data.get("content") else []
                    ),
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
            print(f"  Progress: {completed}/{len(conditions)} | {completed/elapsed:.1f} calls/s")

    print(f"\nLogprobs experiment complete! Results: {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
