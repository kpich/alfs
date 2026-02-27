from pydantic import BaseModel, ConfigDict


class UpdateTarget(BaseModel):
    model_config = ConfigDict(frozen=True)

    form: str
    sense: str | None = (
        None  # None = induce senses; str sense key = refine that sense (future)
    )
