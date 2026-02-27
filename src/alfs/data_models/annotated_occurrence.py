from enum import IntEnum

from pydantic import BaseModel, ConfigDict


class OccurrenceRating(IntEnum):
    NONE = 0  # no listed sense applies
    POOR = 1
    REASONABLE = 2
    EXCELLENT = 3


class AnnotatedOccurrence(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    byte_offset: int
    sense_key: str
    rating: OccurrenceRating  # validated enum, serializes as int to parquet
