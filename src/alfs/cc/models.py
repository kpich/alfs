"""Pydantic models for CC task files and their outputs."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class SenseInfo(BaseModel):
    id: str
    definition: str
    subsenses: list[str] | None = None
    pos: str | None = None


class FormInfo(BaseModel):
    form: str
    senses: list[SenseInfo]


# --- Task models (written to pending/) ---


class CCInductionTask(BaseModel):
    type: Literal["induction"] = "induction"
    id: str
    form: str
    contexts: list[str]
    existing_defs: list[str]


class CCRewriteTask(BaseModel):
    type: Literal["rewrite"] = "rewrite"
    id: str
    form: str
    senses: list[SenseInfo]


class CCTrimSenseTask(BaseModel):
    type: Literal["trim_sense"] = "trim_sense"
    id: str
    form: str
    senses: list[SenseInfo]
    examples: list[list[str]]


class CCMorphRedirectTask(BaseModel):
    type: Literal["morph_redirect"] = "morph_redirect"
    id: str
    forms: list[FormInfo]
    inventory_forms: list[str]


class CCDeleteEntryTask(BaseModel):
    type: Literal["delete_entry"] = "delete_entry"
    id: str
    form: str
    senses: list[SenseInfo]
    examples: list[list[str]]


CCTask = Annotated[
    CCInductionTask
    | CCRewriteTask
    | CCTrimSenseTask
    | CCMorphRedirectTask
    | CCDeleteEntryTask,
    Field(discriminator="type"),
]


# --- Output models (written to done/) ---


class InductionSense(BaseModel):
    definition: str
    pos: str


class CCInductionOutput(BaseModel):
    type: Literal["induction"] = "induction"
    id: str
    form: str
    senses: list[InductionSense]


class RewrittenSense(BaseModel):
    definition: str
    subsenses: list[str] | None = None


class CCRewriteOutput(BaseModel):
    type: Literal["rewrite"] = "rewrite"
    id: str
    form: str
    senses: list[RewrittenSense]


class CCTrimSenseOutput(BaseModel):
    type: Literal["trim_sense"] = "trim_sense"
    id: str
    form: str
    sense_num: int | None
    reason: str


class MorphRelation(BaseModel):
    derived_form: str
    derived_sense_idx: int
    base_form: str
    base_sense_idx: int
    relation: str
    proposed_definition: str


class CCMorphRedirectOutput(BaseModel):
    type: Literal["morph_redirect"] = "morph_redirect"
    id: str
    relations: list[MorphRelation]


class CCDeleteEntryOutput(BaseModel):
    type: Literal["delete_entry"] = "delete_entry"
    id: str
    form: str
    should_delete: bool
    reason: str


CCOutput = Annotated[
    CCInductionOutput
    | CCRewriteOutput
    | CCTrimSenseOutput
    | CCMorphRedirectOutput
    | CCDeleteEntryOutput,
    Field(discriminator="type"),
]
