import uuid

from pydantic import BaseModel, ConfigDict, Field

from alfs.data_models.pos import PartOfSpeech


def _new_sense_id() -> str:
    return str(uuid.uuid4())


class Sense(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str = Field(default_factory=_new_sense_id)
    definition: str = Field(min_length=1)
    pos: PartOfSpeech | None = None
    morph_base: str | None = None
    morph_relation: str | None = None
    updated_by_model: str | None = None
    updated_at: str | None = None


class Alf(BaseModel):
    """A single dictionary entry: one word form and its senses."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    form: str
    senses: list[Sense] = []
    spelling_variant_of: str | None = (
        None  # preferred (American) spelling, if this is a variant
    )

    def get_sense(self, key: str) -> str:
        """Return the definition for a sense key like '1' or '2'."""
        return self.senses[parse_sense_key(key)].definition


class Alfs(BaseModel):
    """The full dictionary: word form → entry."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    entries: dict[str, Alf] = {}


# --- Sense key utilities ---


def sense_key(idx: int) -> str:
    """Build a sense key from a 0-based index.

    sense_key(0)  -> "1"
    sense_key(2)  -> "3"
    """
    return str(idx + 1)


def morph_base_form(alf: Alf) -> str | None:
    """Return the morph base form if all senses with a non-None base share the same
    base, else None."""
    if not alf.senses:
        return None
    bases = {s.morph_base for s in alf.senses if s.morph_base is not None}
    if len(bases) == 1:
        return bases.pop()
    return None


def parse_sense_key(key: str) -> int:
    """Parse a sense key to a 0-based index.

    parse_sense_key("1")  -> 0
    parse_sense_key("3")  -> 2
    """
    key = key.strip()
    if not key:
        raise ValueError("Empty sense key")
    top_idx = int(key) - 1  # 1-indexed -> 0-based
    if top_idx < 0:
        raise ValueError(f"Sense key out of range: {key!r}")
    return top_idx
