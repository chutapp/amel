"""Non-evaluative neutral-filler control experiment.

Tests whether the main experiment's neutral-arm shift is driven by
the *polarity* of the prior conversation or by the mere
*presence* of prior conversation. The current neutral arm fills
prior turns with balanced yes/no judgments on other items
(evaluative filler). Here I replace those with off-topic factual
Q&A (non-evaluative filler) and re-measure the bias score.

Design:
    domain      = code_review (the domain with the largest
                  evaluative-neutral effect in the main experiment)
    items       = the 21 code_review test items
    context_N   = 50 turns
    polarity    = non_evaluative_neutral (new condition)
    reps        = 10
    models      = GPT-4.1 Nano (API), Llama 3.2 3B (local Ollama)

Filler is drawn deterministically from src.non_evaluative_filler
with a per-item seed; sampling is without replacement when possible
(100 pairs available; we use 50 per condition).

Output: data/neutral_filler/results.jsonl with the same schema as
data/all_results.jsonl plus polarity="non_evaluative_neutral".
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))

from src.config import MAX_TOKENS, REPETITIONS, TEMPERATURE
from src.domains import ALL_DOMAINS
from src.non_evaluative_filler import FILLER_PAIRS
from src.parser_v2 import parse_yes_no

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = "https://api.openai.com/v1"
OLLAMA_BASE_URL = "http://localhost:11434"

OUT_FILE = Path("data/neutral_filler/results.jsonl")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

CONTEXT_N = 50
DOMAIN_NAME = "code_review"


def build_filler_messages(system_prompt: str, test_question: str, seed: int) -> list[dict]:
    """50-turn conversation: system, then 50 (user, assistant) factual
    Q&A pairs sampled deterministically from FILLER_PAIRS, then the
    test question."""
    rng = random.Random(seed)
    pairs = rng.sample(FILLER_PAIRS, k=CONTEXT_N)
    messages = [{"role": "system", "content": system_prompt}]
    for q, a in pairs:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": test_question})
    return messages


async def call_openai(client, semaphore, messages):
    async with semaphore:
        payload = {
            "model": "gpt-4.1-nano",
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        start = time.perf_counter()
        try:
            r = await client.post(f"{OPENAI_BASE_URL}/chat/completions",
                                  json=payload, headers=headers, timeout=120.0)
            r.raise_for_status()
            elapsed = (time.perf_counter() - start) * 1000
            return r.json()["choices"][0]["message"]["content"], elapsed
        except Exception as e:
            return f"ERROR: {e}", (time.perf_counter() - start) * 1000


async def call_ollama(client, semaphore, messages):
    async with semaphore:
        payload = {
            "model": "llama3.2:3b",
            "messages": messages,
            "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
            "stream": False,
        }
        start = time.perf_counter()
        try:
            r = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=600.0)
            r.raise_for_status()
            elapsed = (time.perf_counter() - start) * 1000
            return r.json()["message"]["content"], elapsed
        except Exception as e:
            return f"ERROR: {e}", (time.perf_counter() - start) * 1000


async def run_one(client, semaphore, model_label, caller, item, rep, seed):
    d = ALL_DOMAINS[DOMAIN_NAME]
    test_q = d.format_question(item.text)
    messages = build_filler_messages(d.system_prompt, test_q, seed)
    text, elapsed = await caller(client, semaphore, messages)
    parsed = parse_yes_no(text)
    return {
        "domain": DOMAIN_NAME,
        "model": model_label,
        "polarity": "non_evaluative_neutral",
        "context_length": CONTEXT_N,
        "test_item_id": item.id,
        "test_item_text": item.text,
        "test_item_category": item.category,
        "test_item_ground_truth": item.ground_truth,
        "repetition": rep,
        "raw_response": text,
        "parsed_response": parsed,
        "response_time_ms": round(elapsed, 2),
        "seed": seed,
    }


async def run_model(client, model_label, caller, items, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    for item in items:
        for rep in range(REPETITIONS):
            seed = abs(hash(f"{model_label}|{item.id}|nef|{rep}")) & 0xFFFFFFFF
            tasks.append(run_one(client, semaphore, model_label, caller, item, rep, seed))
    n = len(tasks)
    print(f"  {model_label}: {n} calls (concurrency={concurrency})")
    out = []
    done = 0
    for coro in asyncio.as_completed(tasks):
        r = await coro
        out.append(r)
        done += 1
        if done % 25 == 0:
            print(f"    {done}/{n}")
    return out


async def main():
    items = ALL_DOMAINS[DOMAIN_NAME].get_test_items()
    print(f"Domain: {DOMAIN_NAME}, {len(items)} test items, "
          f"{CONTEXT_N}-turn non-evaluative filler, "
          f"{REPETITIONS} reps per (model, item).")

    async with httpx.AsyncClient() as client:
        results: list[dict] = []
        if OPENAI_API_KEY:
            results += await run_model(client, "gpt-4.1-nano", call_openai, items, concurrency=10)
        else:
            print("OPENAI_API_KEY not set; skipping Nano.")
        results += await run_model(client, "llama3.2:3b", call_ollama, items, concurrency=2)

    with open(OUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(results)} rows to {OUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
