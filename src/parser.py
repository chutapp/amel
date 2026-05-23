"""Response parser — extracts yes/no from LLM responses."""

import re


def parse_yes_no(response: str) -> str | None:
    """Parse a yes/no answer from an LLM response.

    Returns:
        "yes", "no", or None if unparseable.
    """
    if not response or response.startswith("ERROR:"):
        return None

    text = response.strip().lower()

    # Remove thinking tags (qwen3 uses <think>...</think>)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    if not text:
        return None

    # Check first word (most models start with Yes/No)
    first_word = re.split(r"[\s.,!:;]", text)[0]

    if first_word in ("yes", "yeah", "yep", "absolutely", "definitely", "certainly"):
        return "yes"
    if first_word in ("no", "nope", "not", "never"):
        return "no"

    # Check first sentence for clear signals
    first_sentence = re.split(r"[.!?\n]", text)[0].strip()

    no_patterns = [
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
        r"\bavoid\b",
        r"\bpoor\s+choice\b",
    ]
    yes_patterns = [
        r"\byes\b",
        r"\bthis is a healthy\b",
        r"\bthis is production[- ]ready\b",
        r"\bthis (?:comment )?is appropriate\b",
        r"\bgood choice\b",
        r"\bhealthy (?:choice|meal|option)\b",
        r"\bthis is a good\b",
        r"\bproduction[- ]ready\b",
        r"\bappropriate\b",
        r"\bi would recommend\b",
    ]

    for pattern in no_patterns:
        if re.search(pattern, first_sentence):
            return "no"

    for pattern in yes_patterns:
        if re.search(pattern, first_sentence):
            return "yes"

    # Check first 3 sentences (some models ramble before answering)
    first_3 = ". ".join(re.split(r"[.!?\n]", text)[:3]).strip()

    for pattern in no_patterns:
        if re.search(pattern, first_3):
            return "no"
    for pattern in yes_patterns:
        if re.search(pattern, first_3):
            return "yes"

    # Look for conclusion patterns
    conclusion_patterns_no = [
        r"\bmy answer is:?\s*no\b",
        r"\bthe answer is:?\s*no\b",
        r"\bi'?d say:?\s*no\b",
        r"\boverall[,:]?\s*no\b",
        r"\bso[,:]?\s*no\b",
        r"\bverdict:?\s*no\b",
    ]
    conclusion_patterns_yes = [
        r"\bmy answer is:?\s*yes\b",
        r"\bthe answer is:?\s*yes\b",
        r"\bi'?d say:?\s*yes\b",
        r"\boverall[,:]?\s*yes\b",
        r"\bso[,:]?\s*yes\b",
        r"\bverdict:?\s*yes\b",
    ]

    for pattern in conclusion_patterns_no:
        if re.search(pattern, text):
            return "no"
    for pattern in conclusion_patterns_yes:
        if re.search(pattern, text):
            return "yes"

    # Last resort: weighted count across the whole response
    # Give more weight to yes/no that appear early
    yes_positions = [m.start() for m in re.finditer(r"\byes\b", text)]
    no_positions = [m.start() for m in re.finditer(r"\bno\b", text)]

    if not yes_positions and not no_positions:
        return None

    # Weight by inverse position (earlier = more weight)
    text_len = max(len(text), 1)
    yes_score = sum(1.0 - (pos / text_len) for pos in yes_positions)
    no_score = sum(1.0 - (pos / text_len) for pos in no_positions)

    if no_score > yes_score * 1.2:  # Need clear margin
        return "no"
    if yes_score > no_score * 1.2:
        return "yes"

    return None
