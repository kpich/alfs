from pydantic import BaseModel, ConfigDict


class Occurrence(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    byte_offset: int
