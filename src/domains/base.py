"""Base domain class."""

from dataclasses import dataclass


@dataclass
class Item:
    """A single item to evaluate."""

    id: str
    text: str
    ground_truth: str  # "yes" or "no"
    category: str  # "clear_positive", "ambiguous", "clear_negative"


class BaseDomain:
    """Base class for experiment domains."""

    name: str = ""
    system_prompt: str = ""
    question_template: str = ""

    def get_context_items_no(self) -> list[Item]:
        """Items that should elicit 'no' responses (for no-saturated context)."""
        raise NotImplementedError

    def get_context_items_yes(self) -> list[Item]:
        """Items that should elicit 'yes' responses (for yes-saturated context)."""
        raise NotImplementedError

    def get_test_items(self) -> list[Item]:
        """Test items with known ground truth across all categories."""
        raise NotImplementedError

    def format_question(self, item_text: str) -> str:
        """Format the question for a given item."""
        return self.question_template.format(item=item_text)

    def format_response(self, answer: str, item: Item) -> str:
        """Generate a realistic assistant response for context building."""
        if answer == "no":
            return f"No. {self._no_reasoning(item)}"
        return f"Yes. {self._yes_reasoning(item)}"

    def _no_reasoning(self, item: Item) -> str:
        return "This is not recommended."

    def _yes_reasoning(self, item: Item) -> str:
        return "This is a good choice."
