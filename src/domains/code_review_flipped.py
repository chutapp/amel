"""Flipped code review domain — question asks 'Should this code be rejected?'

Semantic mapping is inverted:
- 'yes' = reject (code is bad)
- 'no'  = approve (code is good)

This disentangles token-level ('no' is stickier) from semantic-level
(rejection is stickier from RLHF) explanations of the negativity asymmetry.
"""

from src.domains.base import BaseDomain, Item
from src.domains.code_review import CodeReviewDomain


class CodeReviewFlippedDomain(BaseDomain):
    name = "code_review_flipped"
    system_prompt = (
        "You are a senior code reviewer. The user will share a code snippet and ask if "
        "it should be rejected. Answer with 'Yes' or 'No' followed by a brief explanation."
    )
    question_template = "Should this code be rejected?\n\n```\n{item}\n```"

    def _no_reasoning(self, item: Item) -> str:
        return "This code follows best practices and should not be rejected."

    def _yes_reasoning(self, item: Item) -> str:
        return "This code has significant issues and should be rejected."

    def get_context_items_no(self) -> list[Item]:
        """Items that should get 'no' (approve) — clearly good code.

        In the flipped frame, 'no' means 'do not reject' = approve.
        So context_items_no uses the GOOD code from the original domain.
        """
        original = CodeReviewDomain()
        return [
            Item(
                id=item.id.replace("yes_code", "flip_no"),
                text=item.text,
                ground_truth="no",  # flipped: good code → "no, don't reject"
                category="clear_positive",
            )
            for item in original.get_context_items_yes()
        ]

    def get_context_items_yes(self) -> list[Item]:
        """Items that should get 'yes' (reject) — clearly bad code.

        In the flipped frame, 'yes' means 'yes, reject' = reject.
        So context_items_yes uses the BAD code from the original domain.
        """
        original = CodeReviewDomain()
        return [
            Item(
                id=item.id.replace("no_code", "flip_yes"),
                text=item.text,
                ground_truth="yes",  # flipped: bad code → "yes, reject it"
                category="clear_negative",
            )
            for item in original.get_context_items_no()
        ]

    def get_test_items(self) -> list[Item]:
        """Same test items, flipped ground truths.

        Original: 'Is this production-ready?' yes=good, no=bad
        Flipped:  'Should this be rejected?'  yes=bad, no=good
        """
        original = CodeReviewDomain()
        flipped_items = []
        for item in original.get_test_items():
            flipped_gt = "no" if item.ground_truth == "yes" else "yes"
            # Swap category labels to match new ground truth direction
            if item.category == "clear_positive":
                flipped_cat = "clear_negative"  # good code → "no don't reject" = clear_negative in rejection frame
            elif item.category == "clear_negative":
                flipped_cat = "clear_positive"  # bad code → "yes reject" = clear_positive in rejection frame
            else:
                flipped_cat = "ambiguous"
            flipped_items.append(Item(
                id=item.id.replace("test_code", "test_flip"),
                text=item.text,
                ground_truth=flipped_gt,
                category=flipped_cat,
            ))
        return flipped_items
