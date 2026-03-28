"""Pydantic models for CC task files and their outputs."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from alfs.data_models.occurrence import Occurrence


class SenseInfo(BaseModel):
    id: str
    definition: str
    pos: str | None = None


# --- Task models (written to pending/) ---


class CCInductionTask(BaseModel):
    type: Literal["induction"] = "induction"
    id: str
    form: str
    contexts: list[str]
    existing_defs: list[str]
    occurrence_refs: list[Occurrence] = []  # parallel to contexts list


CCTask = Annotated[CCInductionTask, Field(discriminator="type")]


# --- Output models (written to done/) ---


class InductionSense(BaseModel):
    definition: str
    pos: str


class ContextLabel(BaseModel):
    context_idx: int
    sense_idx: int | None  # 1-indexed into new_senses; None = _skip


class CCInductionOutput(BaseModel):
    type: Literal["induction"] = "induction"
    id: str
    form: str
    new_senses: list[InductionSense] = []
    context_labels: list[ContextLabel] = []
    add_to_blocklist: bool = False
    blocklist_reason: str | None = None


CCOutput = Annotated[CCInductionOutput, Field(discriminator="type")]
