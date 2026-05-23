"""Conversation builder for the AMEL experiment."""

import random
from src.domains.base import BaseDomain, Item
from src.config import ExperimentCondition


def build_messages(
    domain: BaseDomain,
    polarity: str,
    context_length: int,
    test_item: Item,
    seed: int,
) -> list[dict[str, str]]:
    """Build a complete message list for one experimental condition.

    Args:
        domain: The domain definition.
        polarity: "no_saturated", "yes_saturated", "neutral", or "baseline".
        context_length: Number of prior turns (0 for baseline).
        test_item: The test item to evaluate at the end.
        seed: Random seed for reproducible context item selection.

    Returns:
        List of message dicts with "role" and "content" keys.
    """
    messages = [{"role": "system", "content": domain.system_prompt}]

    if polarity != "baseline" and context_length > 0:
        context_items = _select_context_items(domain, polarity, context_length, seed)
        for item in context_items:
            # User asks about the context item
            messages.append({
                "role": "user",
                "content": domain.format_question(item.text),
            })
            # Assistant responds with the ground truth answer
            messages.append({
                "role": "assistant",
                "content": domain.format_response(item.ground_truth, item),
            })

    # Final user message: the test item
    messages.append({
        "role": "user",
        "content": domain.format_question(test_item.text),
    })

    return messages


def _select_context_items(
    domain: BaseDomain,
    polarity: str,
    context_length: int,
    seed: int,
) -> list[Item]:
    """Select context items based on polarity and desired length.

    For no_saturated: 90% no-items, 10% yes-items
    For yes_saturated: 90% yes-items, 10% no-items
    For neutral: 50% each
    """
    rng = random.Random(seed)

    no_items = domain.get_context_items_no()
    yes_items = domain.get_context_items_yes()

    if polarity == "no_saturated":
        n_no = int(context_length * 0.9)
        n_yes = context_length - n_no
    elif polarity == "yes_saturated":
        n_yes = int(context_length * 0.9)
        n_no = context_length - n_yes
    elif polarity == "neutral":
        n_no = context_length // 2
        n_yes = context_length - n_no
    else:
        return []

    # Sample with replacement if needed (for long contexts)
    selected_no = rng.choices(no_items, k=n_no)
    selected_yes = rng.choices(yes_items, k=n_yes)

    # Combine and shuffle (but keep the overall ratio)
    combined = selected_no + selected_yes
    rng.shuffle(combined)

    return combined


def build_messages_positional(
    domain: BaseDomain,
    polarity: str,
    total_turns: int,
    biased_positions: list[int],
    test_item: Item,
    seed: int,
) -> list[dict[str, str]]:
    """Build a conversation with biased turns placed at specific positions.

    Non-biased positions get neutral (50/50) context items.
    Biased positions get items matching the polarity (all same-polarity).

    Args:
        domain: The domain definition.
        polarity: "no_saturated" or "yes_saturated" — applied at biased_positions.
        total_turns: Total conversation turns (e.g., 50).
        biased_positions: List of 0-indexed positions to place biased turns.
        test_item: The test item to evaluate at the end.
        seed: Random seed for reproducible context item selection.

    Returns:
        List of message dicts with "role" and "content" keys.
    """
    rng = random.Random(seed)

    no_items = domain.get_context_items_no()
    yes_items = domain.get_context_items_yes()

    biased_set = set(biased_positions)

    messages = [{"role": "system", "content": domain.system_prompt}]

    for pos in range(total_turns):
        if pos in biased_set:
            # Biased turn: use items matching the polarity
            if polarity == "no_saturated":
                item = rng.choice(no_items)
            else:
                item = rng.choice(yes_items)
        else:
            # Neutral turn: 50/50 chance of yes or no item
            if rng.random() < 0.5:
                item = rng.choice(no_items)
            else:
                item = rng.choice(yes_items)

        messages.append({
            "role": "user",
            "content": domain.format_question(item.text),
        })
        messages.append({
            "role": "assistant",
            "content": domain.format_response(item.ground_truth, item),
        })

    # Final test item
    messages.append({
        "role": "user",
        "content": domain.format_question(test_item.text),
    })

    return messages
