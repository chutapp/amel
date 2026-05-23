"""Mitigation experiment — sequential batch evaluation.

Tests whether bias emerges naturally when items are evaluated sequentially
in a single conversation (model's own answers form the context), and
whether balanced ordering mitigates it.

Three conditions:
  (a) fresh: each item in its own conversation (already in baseline data)
  (b) sequential_fixed: all 21 items in one conversation, fixed order
  (c) sequential_balanced: all 21 items in one conversation, interleaved
      expected-yes / expected-no order

Models: gpt-4.1-nano, claude-sonnet-4-6, llama3.2:3b
Domains: all three
Repetitions: 10
~3,780 API calls
"""

import asyncio
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))

from src.config import MAX_TOKENS, TEMPERATURE
from src.domains import ALL_DOMAINS
from src.parser import parse_yes_no

MODELS = {
    "gpt-4.1-nano": "openai",
    "llama3.2:3b": "ollama",
    "qwen3.5:4b": "ollama",
}

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OLLAMA_BASE_URL = "http://localhost:11434"

CONDITIONS = ["sequential_fixed", "sequential_balanced"]
REPETITIONS = 10


@dataclass
class MitigationResult:
    domain: str
    model: str
    condition: str  # sequential_fixed or sequential_balanced
    test_item_id: str
    test_item_text: str
    test_item_category: str
    test_item_ground_truth: str
    position: int  # position in the sequence (0-indexed)
    repetition: int
    raw_response: str
    parsed_response: str | None
    response_time_ms: float
    prior_yes_count: int  # how many "yes" before this item
    prior_no_count: int  # how many "no" before this item
    seed: int
    timestamp: str


async def call_api(client, model, provider, messages):
    """Call the appropriate API based on provider."""
    start = time.perf_counter()

    if provider == "openai":
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            json={"model": model, "messages": messages, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            timeout=60.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    elif provider == "anthropic":
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        non_system = [m for m in messages if m["role"] != "system"]
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            json={
                "model": model, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
                "system": system_msg, "messages": non_system,
            },
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        content = resp.json()["content"][0]["text"]
    elif provider == "ollama":
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": model, "messages": messages, "stream": False, "think": False,
                  "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS}},
            timeout=600.0,
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
    else:
        raise ValueError(f"Unknown provider: {provider}")

    elapsed_ms = (time.perf_counter() - start) * 1000
    return content, elapsed_ms


def order_items(test_items, condition, seed):
    """Order test items based on condition."""
    rng = random.Random(seed)
    items = list(test_items)

    if condition == "sequential_fixed":
        # Fixed order: clear_positive, ambiguous, clear_negative
        items.sort(key=lambda x: {"clear_positive": 0, "ambiguous": 1, "clear_negative": 2}[x.category])
    elif condition == "sequential_balanced":
        # Interleave expected-yes and expected-no items
        yes_items = [i for i in items if i.ground_truth == "yes"]
        no_items = [i for i in items if i.ground_truth == "no"]
        rng.shuffle(yes_items)
        rng.shuffle(no_items)
        items = []
        while yes_items or no_items:
            if yes_items:
                items.append(yes_items.pop(0))
            if no_items:
                items.append(no_items.pop(0))

    return items


async def run_sequential_batch(client, model, provider, domain, condition, rep, seed):
    """Run all 21 test items sequentially in a single conversation."""
    test_items = domain.get_test_items()
    ordered = order_items(test_items, condition, seed)

    messages = [{"role": "system", "content": domain.system_prompt}]
    results = []
    prior_yes = 0
    prior_no = 0

    for pos, item in enumerate(ordered):
        # Add the test question
        messages.append({"role": "user", "content": domain.format_question(item.text)})

        try:
            content, elapsed_ms = await call_api(client, model, provider, messages)
        except Exception as e:
            content = f"ERROR: {e}"
            elapsed_ms = 0

        parsed = parse_yes_no(content)

        result = MitigationResult(
            domain=domain.name,
            model=model,
            condition=condition,
            test_item_id=item.id,
            test_item_text=item.text[:500],
            test_item_category=item.category,
            test_item_ground_truth=item.ground_truth,
            position=pos,
            repetition=rep,
            raw_response=content[:2000],
            parsed_response=parsed,
            response_time_ms=elapsed_ms,
            prior_yes_count=prior_yes,
            prior_no_count=prior_no,
            seed=seed,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        results.append(result)

        # Add assistant response to conversation (model's own answer)
        messages.append({"role": "assistant", "content": content[:500]})

        # Track running counts
        if parsed == "yes":
            prior_yes += 1
        elif parsed == "no":
            prior_no += 1

    return results


async def main():
    output_dir = Path("data/mitigation")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.jsonl"

    # Load existing for resume
    existing_keys = set()
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    key = f"{r['domain']}|{r['model']}|{r['condition']}|{r['test_item_id']}|{r['repetition']}"
                    existing_keys.add(key)
        print(f"Resuming: {len(existing_keys)} existing results")

    # Generate all conditions
    all_batches = []
    for model, provider in MODELS.items():
        for domain_name, domain in ALL_DOMAINS.items():
            for condition in CONDITIONS:
                for rep in range(REPETITIONS):
                    # Check if any item from this batch already exists
                    test_items = domain.get_test_items()
                    first_key = f"{domain_name}|{model}|{condition}|{test_items[0].id}|{rep}"
                    if first_key not in existing_keys:
                        seed = hash(f"{domain_name}|{model}|{condition}|{rep}") & 0xFFFFFFFF
                        all_batches.append((model, provider, domain, condition, rep, seed))

    print(f"Total sequential batches to run: {len(all_batches)}")
    print(f"Total API calls: ~{len(all_batches) * 21}")

    if not all_batches:
        print("All conditions complete!")
        return

    completed = 0
    start_time = time.perf_counter()

    async with httpx.AsyncClient() as client:
        for model, provider, domain, condition, rep, seed in all_batches:
            print(f"  {model} | {domain.name} | {condition} | rep {rep}...")
            batch_results = await run_sequential_batch(
                client, model, provider, domain, condition, rep, seed
            )

            with open(results_file, "a") as f:
                for result in batch_results:
                    f.write(json.dumps(asdict(result)) + "\n")

            completed += 1
            elapsed = time.perf_counter() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = (len(all_batches) - completed) / rate if rate > 0 else 0
            print(f"    Batch {completed}/{len(all_batches)} | ETA: {remaining/60:.1f}m")

    print(f"\nMitigation experiment complete! Results: {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
