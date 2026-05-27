"""Logprobs experiment, replicated on Llama 3.2 3B (local).

Re-runs the §5.1 logprobs experiment on a second model to test
whether the "continuous probability shift" finding generalises
beyond GPT-4.1 Nano. Ollama exposes first-token logprobs natively,
so no API cost.

Design mirrors run_logprobs.py exactly:
    model      = llama3.2:3b (via local Ollama)
    domain     = code_review
    items      = 21
    conditions = baseline, no_sat@5, yes_sat@5, no_sat@50, yes_sat@50
    reps       = 10
    total      = 1,050 calls
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))

from src.config import MAX_TOKENS
from src.conversation import build_messages
from src.domains import ALL_DOMAINS
from src.parser_v2 import parse_yes_no

MODEL = "llama3.2:3b"
DOMAIN = "code_review"
CONDITIONS = [
    ("baseline", 0),
    ("no_saturated", 5),
    ("yes_saturated", 5),
    ("no_saturated", 50),
    ("yes_saturated", 50),
]
REPETITIONS = 10
CONCURRENCY = 2  # local model; serial-ish
OLLAMA_BASE_URL = "http://localhost:11434"


async def call_ollama_logprobs(client, messages, semaphore):
    async with semaphore:
        start = time.perf_counter()
        try:
            r = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 1.0, "num_predict": MAX_TOKENS},
                    "logprobs": True,
                    "top_logprobs": 5,
                },
                timeout=600.0,
            )
            r.raise_for_status()
            elapsed = (time.perf_counter() - start) * 1000
            data = r.json()
            return data["message"]["content"], elapsed, data.get("logprobs", [])
        except Exception as e:
            return f"ERROR: {e}", (time.perf_counter() - start) * 1000, []


def extract_first_token_probs(logprobs_data):
    """Aggregate P(Yes-like) / P(No-like) over the first generated token."""
    if not logprobs_data:
        return None, None
    first = logprobs_data[0]
    top = first.get("top_logprobs", []) or []
    # Ensure the chosen token is in the pool too
    if "token" in first and not any(t.get("token") == first["token"] for t in top):
        top = top + [{"token": first["token"], "logprob": first["logprob"]}]
    p_yes = 0.0
    p_no = 0.0
    for entry in top:
        token = (entry.get("token") or "").strip().lower()
        prob = math.exp(entry["logprob"])
        if token in ("yes", "yeah", "absolutely", "definitely"):
            p_yes += prob
        elif token in ("no", "nope", "not"):
            p_no += prob
    return p_yes, p_no


async def main():
    out_dir = Path("data/logprobs_llama")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.jsonl"

    domain = ALL_DOMAINS[DOMAIN]
    items = domain.get_test_items()

    existing = set()
    if out_file.exists():
        with open(out_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing.add(f"{r['polarity']}|{r['context_length']}|{r['test_item_id']}|{r['repetition']}")
        print(f"Resuming: {len(existing)} existing rows")

    todo = []
    for pol, ctxlen in CONDITIONS:
        for it in items:
            for rep in range(REPETITIONS):
                key = f"{pol}|{ctxlen}|{it.id}|{rep}"
                if key in existing:
                    continue
                seed = hash(f"{DOMAIN}|{pol}|{ctxlen}|{it.id}|{rep}") & 0xFFFFFFFF
                todo.append((pol, ctxlen, it, rep, seed))

    print(f"Total scheduled: {len(CONDITIONS) * len(items) * REPETITIONS}")
    print(f"Remaining: {len(todo)}")
    if not todo:
        return

    sem = asyncio.Semaphore(CONCURRENCY)
    done = 0
    t0 = time.perf_counter()

    async with httpx.AsyncClient() as client:
        batch_size = CONCURRENCY * 4
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]

            async def one(pol, ctxlen, item, rep, seed):
                msgs = build_messages(domain, pol, ctxlen, item, seed)
                content, elapsed, lp = await call_ollama_logprobs(client, msgs, sem)
                parsed = parse_yes_no(content)
                p_yes, p_no = extract_first_token_probs(lp)
                first_top = lp[0].get("top_logprobs", []) if lp else []
                return {
                    "domain": DOMAIN,
                    "model": MODEL,
                    "polarity": pol,
                    "context_length": ctxlen,
                    "test_item_id": item.id,
                    "test_item_text": item.text[:500],
                    "test_item_category": item.category,
                    "test_item_ground_truth": item.ground_truth,
                    "repetition": rep,
                    "raw_response": content[:2000] if isinstance(content, str) else "",
                    "parsed_response": parsed,
                    "p_yes": p_yes,
                    "p_no": p_no,
                    "first_token_logprobs": first_top,
                    "response_time_ms": round(elapsed, 2),
                    "seed": seed,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            results = await asyncio.gather(*(one(*c) for c in batch))
            with open(out_file, "a") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            done += len(results)
            rate = done / (time.perf_counter() - t0)
            print(f"  {done}/{len(todo)}  {rate:.2f}/s")

    print(f"\nDone. {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
