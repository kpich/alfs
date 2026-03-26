from enum import IntEnum

from pydantic import BaseModel, ConfigDict


class OccurrenceRating(IntEnum):
    POOR = 0  # doesn't fit / inappropriate
    OKAY = 1  # fits but probably needs a more refined sense
    EXCELLENT = 2  # great fit


class AnnotatedOccurrence(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    byte_offset: int
    sense_key: str
    rating: OccurrenceRating  # validated enum, serializes as int to parquet
    model: str | None = None
    synonyms: list[str] | None = None  # None=missing data, []=no substitutes found
