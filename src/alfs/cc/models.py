"""Pydantic models for CC task files and their outputs."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

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


# --- Output models (written to done/) ---


class InductionMorphRel(BaseModel):
    base_form: str
    # e.g. "plural", "past_tense", "past_participle", "present_participle",
    # "3sg_present", "comparative", "superlative"
    relation: str


class InductionSense(BaseModel):
    definition: str
    pos: str
    morph_rel: InductionMorphRel | None = None


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
    action: Literal["morph_rel", "delete", "normalize_case"]
    morph_rels: list[MorphRelEntry] = []
    blocklist_reason: str | None = None
    canonical_form: str | None = None  # required when action == "normalize_case"


class DeletedSenseEntry(BaseModel):
    sense_idx: int
    reason: str


class SenseRewrite(BaseModel):
    sense_idx: int
    definition: str


class PosCorrection(BaseModel):
    sense_idx: int
    pos: str


class CCQCTask(BaseModel):
    type: Literal["qc"] = "qc"
    id: str
    form: str
    senses: list[SenseInfo]


class CCMWETask(BaseModel):
    type: Literal["mwe"] = "mwe"
    id: str
    form: str
    components: list[str]
    pmi: float
    corpus_count: int
    contexts: list[str]
    occurrence_refs: list[Occurrence] = []


CCTask = Annotated[
    CCInductionTask | CCMorphRelBlockTask | CCQCTask | CCMWETask,
    Field(discriminator="type"),
]


class CCQCOutput(BaseModel):
    type: Literal["qc"] = "qc"
    id: str
    form: str
    # Sense-level ops (combinable)
    morph_rels: list[MorphRelEntry] = []
    deleted_senses: list[DeletedSenseEntry] = []
    sense_rewrites: list[SenseRewrite] = []
    pos_corrections: list[PosCorrection] = []
    # Entry-level ops (mutually exclusive with each other and with sense-level)
    delete_entry: bool = False
    delete_entry_reason: str | None = None
    normalize_case: str | None = None  # canonical form, or None
    spelling_variant_of: str | None = None  # preferred form, or None

    @model_validator(mode="after")
    def check_entry_vs_sense_ops(self) -> CCQCOutput:
        n_entry = sum(
            [
                self.delete_entry,
                self.normalize_case is not None,
                self.spelling_variant_of is not None,
            ]
        )
        sense_level = bool(
            self.morph_rels
            or self.deleted_senses
            or self.sense_rewrites
            or self.pos_corrections
        )
        if n_entry > 1:
            raise ValueError("at most one entry-level action allowed")
        if n_entry > 0 and sense_level:
            raise ValueError("entry-level and sense-level actions cannot be combined")
        return self


class CCMWEOutput(BaseModel):
    type: Literal["mwe"] = "mwe"
    id: str
    form: str
    action: Literal["approve", "skip", "blocklist"]
    blocklist_reason: str | None = None
    occurrence_refs: list[Occurrence] = []  # copied from task; used for approve


CCOutput = Annotated[
    CCInductionOutput | CCMorphRelBlockOutput | CCQCOutput | CCMWEOutput,
    Field(discriminator="type"),
]
