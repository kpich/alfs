"""Shared JSON schemas for LLM structured outputs in refinement modules."""

from alfs.data_models.pos import PartOfSpeech

POS_VALUES: list[str] = [p.value for p in PartOfSpeech]

# Generic critic schema: LLM validates its own prior output.
CRITIC_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "is_valid": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["is_valid", "reason"],
}

# Part-of-speech assignment schema.
POS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "pos": {"type": "string", "enum": POS_VALUES},
    },
    "required": ["pos"],
}
