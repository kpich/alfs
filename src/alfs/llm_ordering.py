"""Global LLM trust ordering for the Alfs lexicon.

Higher index = higher trust. A model may only destructively overwrite a sense
if its rank is >= the rank of the model that last wrote it.
Models not in ORDERING get rank 0 (lowest).
"""

ORDERING: list[str] = [
    "qwen2.5:32b",  # rank 1
    "claude-code",  # rank 2
]


def rank(model: str | None) -> int:
    """Rank of a model name (0 if unknown/None, higher = more trusted)."""
    if model is None:
        return 0
    try:
        return ORDERING.index(model) + 1
    except ValueError:
        return 0


def can_overwrite(requesting: str | None, existing: str | None) -> bool:
    """True if requesting model may destructively edit a sense written by existing."""
    return rank(requesting) >= rank(existing)
