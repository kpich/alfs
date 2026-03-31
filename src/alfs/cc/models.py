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


class CCMorphRelBlockTask(BaseModel):
    type: Literal["morphrel_block"] = "morphrel_block"
    id: str
    form: str
    senses: list[SenseInfo]


CCTask = Annotated[CCInductionTask | CCMorphRelBlockTask, Field(discriminator="type")]


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
    occurrence_refs: list[
        Occurrence
    ] = []  # parallel to contexts; copied from task file
    add_to_blocklist: bool = False
    blocklist_reason: str | None = None


class MorphRelEntry(BaseModel):
    sense_idx: int
    morph_base: str
    morph_relation: str
    proposed_definition: str
    promote_to_parent: bool


class CCMorphRelBlockOutput(BaseModel):
    type: Literal["morphrel_block"] = "morphrel_block"
    id: str
    form: str
    action: Literal["morph_rel", "delete"]
    morph_rels: list[MorphRelEntry] = []
    blocklist_reason: str | None = None


CCOutput = Annotated[
    CCInductionOutput | CCMorphRelBlockOutput, Field(discriminator="type")
]
