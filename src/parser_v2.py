"""Symmetric yes/no response parser (v2).

Replaces the asymmetric parser used during data collection. The original
parser (src/parser.py) had 12 "no" patterns vs 10 "yes" patterns, broader
"no" regexes (e.g. `\\bavoid\\b`, `\\bpoor\\s+choice\\b`), and a priority
order that checked "no" patterns before "yes" patterns on every fallback
loop. On mixed or hedged outputs that asymmetry biased the parse toward
"no".

v2 design:
  - Symmetric pattern lists (same count, mirrored specificity).
  - Each layer SCORES both yes and no matches in parallel and returns the
    winner if the margin is clear; ties produce None at that layer and fall
    through to the next.
  - First-word and weighted-position fallback unchanged in spirit.

Usage:
    from src.parser_v2 import parse_yes_no
    label = parse_yes_no(raw_response)  # "yes", "no", or None
"""
from __future__ import annotations

import re

YES_FIRST_WORDS = {"yes", "yeah", "yep", "absolutely", "definitely", "certainly", "correct", "affirmative"}
NO_FIRST_WORDS = {"no", "nope", "not", "never", "incorrect", "negative"}

# Mirrored patterns — equal count, equal specificity tier
NO_PATTERNS = [
    r"\bno\b",
    r"\bnot\s+(?:a\s+)?healthy\b",
    r"\bnot\s+production[- ]ready\b",
    r"\bnot\s+appropriate\b",
    r"\bnot\s+recommended\b",
    r"\bunhealthy\b",
    r"\bshould\s*n[o']?t\b",
    r"\bnot\s+a\s+good\b",
    r"\bi\s+would\s+(?:not|n't)\b",
    r"\bthis\s+is\s+not\b",
]

YES_PATTERNS = [
    r"\byes\b",
    r"\b(?:this\s+is\s+(?:a\s+)?)?healthy\b",
    r"\b(?:this\s+is\s+)?production[- ]ready\b",
    r"\b(?:this\s+is\s+)?appropriate\b",
    r"\bi\s+would\s+recommend\b",
    r"\b(?:good|healthy)\s+choice\b",
    r"\bshould\s+(?:be\s+)?(?:approve|accept|allow|merge)d?\b",
    r"\bis\s+a\s+good\b",
    r"\bi\s+would\s+approve\b",
    r"\bthis\s+is\s+(?:fine|ok(?:ay)?|acceptable)\b",
]

CONCLUSION_NO = [
    r"\bmy\s+answer\s+is:?\s*no\b",
    r"\bthe\s+answer\s+is:?\s*no\b",
    r"\bi'?d\s+say:?\s*no\b",
    r"\boverall[,:]?\s*no\b",
    r"\bso[,:]?\s*no\b",
    r"\bverdict:?\s*no\b",
]
CONCLUSION_YES = [
    r"\bmy\s+answer\s+is:?\s*yes\b",
    r"\bthe\s+answer\s+is:?\s*yes\b",
    r"\bi'?d\s+say:?\s*yes\b",
    r"\boverall[,:]?\s*yes\b",
    r"\bso[,:]?\s*yes\b",
    r"\bverdict:?\s*yes\b",
]


def _count_matches(text: str, patterns: list[str]) -> int:
    return sum(1 for p in patterns if re.search(p, text))


def _score_and_decide(text: str, yes_patterns: list[str], no_patterns: list[str]) -> str | None:
    """Return 'yes' / 'no' if one side has strictly more matches, else None."""
    y = _count_matches(text, yes_patterns)
    n = _count_matches(text, no_patterns)
    if y > n:
        return "yes"
    if n > y:
        return "no"
    return None


def parse_yes_no(response: str) -> str | None:
    if not response or response.startswith("ERROR:"):
        return None

    text = response.strip().lower()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not text:
        return None

    # Layer 1: explicit first word (highest confidence)
    first_word = re.split(r"[\s.,!:;]", text, maxsplit=1)[0]
    if first_word in YES_FIRST_WORDS:
        return "yes"
    if first_word in NO_FIRST_WORDS:
        return "no"

    # Layer 2: first sentence — score both sides in parallel
    first_sentence = re.split(r"[.!?\n]", text, maxsplit=1)[0].strip()
    decision = _score_and_decide(first_sentence, YES_PATTERNS, NO_PATTERNS)
    if decision is not None:
        return decision

    # Layer 3: first 3 sentences
    first_3 = ". ".join(re.split(r"[.!?\n]", text)[:3]).strip()
    decision = _score_and_decide(first_3, YES_PATTERNS, NO_PATTERNS)
    if decision is not None:
        return decision

    # Layer 4: explicit conclusion patterns (whole text)
    decision = _score_and_decide(text, CONCLUSION_YES, CONCLUSION_NO)
    if decision is not None:
        return decision

    # Layer 5: position-weighted bare token count across whole text
    yes_positions = [m.start() for m in re.finditer(r"\byes\b", text)]
    no_positions = [m.start() for m in re.finditer(r"\bno\b", text)]
    if not yes_positions and not no_positions:
        return None

    text_len = max(len(text), 1)
    yes_score = sum(1.0 - (pos / text_len) for pos in yes_positions)
    no_score = sum(1.0 - (pos / text_len) for pos in no_positions)

    if no_score > yes_score * 1.2:
        return "no"
    if yes_score > no_score * 1.2:
        return "yes"
    return None
