"""Run context-bias experiment against DeepSeek-V3.

DeepSeek's chat-completions API is OpenAI-compatible. We use the same
8,190 conditions per model as every other API in this study (POLARITIES x
CONTEXT_LENGTHS x REPETITIONS x test_items, plus baseline reps), no
system-prompt difference, T=1.0, 100-token cap, parsed via the v2
symmetric yes/no parser used everywhere else.

Key is read from macOS Keychain via the `secret` CLI:
    secret get deepseek.amel.api-key
This avoids putting the key in env, files, or shell history.

Output: data/deepseek-v3/results.jsonl  (resume-safe; same _result_key
schema as the other runners)
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import subprocess
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

BASE_URL = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"  # DeepSeek-V3 (non-reasoning); consistent with other no-thinking APIs in this study
CONCURRENCY = 10  # conservative; ramp up if no rate limits hit


def _load_api_key() -> str:
    if (k := os.environ.get("DEEPSEEK_API_KEY")):
        return k
    try:
        return subprocess.check_output(
            ["secret", "get", "deepseek.amel.api-key"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ERROR: could not load DeepSeek API key: {e}", file=sys.stderr)
        sys.exit(1)


API_KEY = _load_api_key()


async def call_deepseek(
    client: httpx.AsyncClient,
    model: str,
    messages: list[dict],
    semaphore: asyncio.Semaphore,
) -> tuple[str, float, int]:
    async with semaphore:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
        start = time.perf_counter()
        try:
            resp = await client.post(
                f"{BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                timeout=120.0,
            )
            resp.raise_for_status()
            elapsed_ms = (time.perf_counter() - start) * 1000
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return content, elapsed_ms, len(messages)
        except httpx.HTTPStatusError as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            if e.response.status_code == 429:
                retry_after = float(e.response.headers.get("retry-after", "10"))
                print(f"  Rate limited, waiting {retry_after}s...")
                await asyncio.sleep(retry_after)
                return await call_deepseek(client, model, messages, semaphore)
            return (
                f"ERROR: {e.response.status_code} {e.response.text[:500]}",
                elapsed_ms,
                len(messages),
            )
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
) -> ExperimentResult:
    raw_response, response_time, num_messages = await call_deepseek(
        client, model, messages, semaphore
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


async def main() -> None:
    output_dir = Path("data/deepseek-v3")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.jsonl"
    log_file = output_dir / "experiment.log"

    metadata = {
        "experiment": "context_bias_deepseek_v3",
        "start_time": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": MODEL,
            "base_url": BASE_URL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "context_lengths": CONTEXT_LENGTHS,
            "polarities": POLARITIES,
            "repetitions": REPETITIONS,
            "domains": list(ALL_DOMAINS.keys()),
        },
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    existing_keys: set[str] = set()
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing_keys.add(_result_key(r))
        print(f"Resuming: found {len(existing_keys)} existing results")

    conditions = []
    for _, domain in ALL_DOMAINS.items():
        for test_item in domain.get_test_items():
            for rep in range(REPETITIONS):
                conditions.append((domain, MODEL, "baseline", 0, test_item, rep))
            for polarity in POLARITIES:
                for ctx_len in CONTEXT_LENGTHS:
                    for rep in range(REPETITIONS):
                        conditions.append((domain, MODEL, polarity, ctx_len, test_item, rep))

    remaining = []
    for cond in conditions:
        domain, model, polarity, ctx_len, test_item, rep = cond
        key = f"{domain.name}|{model}|{polarity}|{ctx_len}|{test_item.id}|{rep}"
        if key not in existing_keys:
            remaining.append(cond)

    total = len(conditions)
    done = len(existing_keys)
    print(f"Total conditions: {total:,}")
    print(f"Already done: {done:,}")
    print(f"Remaining: {len(remaining):,}")
    if not remaining:
        print("All done!")
        return

    random.Random(42).shuffle(remaining)

    semaphore = asyncio.Semaphore(CONCURRENCY)
    completed = done
    experiment_start = time.perf_counter()

    async with httpx.AsyncClient() as client:
        batch_size = CONCURRENCY * 2
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start : batch_start + batch_size]
            tasks = []
            for domain, model, polarity, ctx_len, test_item, rep in batch:
                seed = hash(f"{domain.name}|{polarity}|{ctx_len}|{test_item.id}|{rep}") & 0xFFFFFFFF
                messages = build_messages(domain, polarity, ctx_len, test_item, seed)
                tasks.append(
                    run_single(
                        client, semaphore, model, domain, polarity, ctx_len, test_item, rep, messages, seed
                    )
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
                line = (
                    f"  Progress: {completed:,}/{total:,} ({pct:.1f}%)  "
                    f"ETA: {remaining_time/60:.1f}m  Rate: {rate:.1f} calls/s"
                )
                print(line)
                with open(log_file, "a") as f:
                    f.write(line + "\n")

    total_elapsed = time.perf_counter() - experiment_start
    print(f"\nDeepSeek-V3 run complete in {total_elapsed/60:.1f} minutes")
    print(f"Results: {results_file}")

    metadata["end_time"] = datetime.now(timezone.utc).isoformat()
    metadata["total_duration_seconds"] = total_elapsed
    metadata["total_conditions_run"] = completed - done
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
