from pydantic import BaseModel, ConfigDict


class Sense(BaseModel):
    model_config = ConfigDict(frozen=True)

    definition: str
    subsenses: list[str] = []  # up to one level of sub-definitions


class Alf(BaseModel):
    """A single dictionary entry: one word form and its senses."""

    model_config = ConfigDict(frozen=True)

    form: str
    senses: list[Sense] = []

    def get_sense(self, key: str) -> str:
        """Return the definition for a sense key like '2' or '3b'."""
        top_idx, sub_idx = parse_sense_key(key)
        sense = self.senses[top_idx]
        if sub_idx is None:
            return sense.definition
        return sense.subsenses[sub_idx]


class Alfs(BaseModel):
    """The full dictionary: word form → entry."""

    model_config = ConfigDict(frozen=True)

    entries: dict[str, Alf] = {}


# --- Sense key utilities ---


def sense_key(idx: int, sub_idx: int | None = None) -> str:
    """Build a sense key from 0-based indices.

    sense_key(0)     -> "1"
    sense_key(2)     -> "3"
    sense_key(2, 0)  -> "3a"
    sense_key(2, 1)  -> "3b"
    """
    key = str(idx + 1)
    if sub_idx is not None:
        key += chr(ord("a") + sub_idx)
    return key


def parse_sense_key(key: str) -> tuple[int, int | None]:
    """Parse a sense key to 0-based (top_idx, sub_idx).

    parse_sense_key("1")   -> (0, None)
    parse_sense_key("3b")  -> (2, 1)
    """
    key = key.strip()
    if not key:
        raise ValueError("Empty sense key")
    if key[-1].isalpha():
        top_part, sub_char = key[:-1], key[-1].lower()
        sub_idx: int | None = ord(sub_char) - ord("a")
    else:
        top_part, sub_idx = key, None
    top_idx = int(top_part) - 1  # 1-indexed -> 0-based
    if top_idx < 0 or (sub_idx is not None and sub_idx < 0):
        raise ValueError(f"Sense key out of range: {key!r}")
    return top_idx, sub_idx
