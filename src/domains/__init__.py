"""Domain definitions for the AMEL experiment."""

from src.domains.meals import MealDomain
from src.domains.code_review import CodeReviewDomain
from src.domains.content_mod import ContentModerationDomain

ALL_DOMAINS = {
    "meals": MealDomain(),
    "code_review": CodeReviewDomain(),
    "content_moderation": ContentModerationDomain(),
}
