"""Typed change request classes for the clerk queue."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from alfs.data_models.alf import Alf, Sense
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore


class AddSensesRequest(BaseModel):
    type: Literal["add_senses"] = "add_senses"
    id: str
    created_at: datetime
    form: str
    new_senses: list[Sense]

    def apply(self, sense_store: SenseStore, occ_store: OccurrenceStore | None) -> None:
        new_senses = self.new_senses
        form = self.form

        def merge(existing: Alf | None) -> Alf:
            if existing is None:
                return Alf(form=form, senses=new_senses)
            existing_defs = {s.definition.strip().lower() for s in existing.senses}
            to_add = [
                s
                for s in new_senses
                if s.definition.strip().lower() not in existing_defs
            ]
            if not to_add:
                return existing
            return existing.model_copy(
                update={"senses": list(existing.senses) + to_add}
            )

        sense_store.update(form, merge)


class RewriteRequest(BaseModel):
    type: Literal["rewrite"] = "rewrite"
    id: str
    created_at: datetime
    form: str
    before: list[Sense]
    after: list[Sense]

    def apply(self, sense_store: SenseStore, occ_store: OccurrenceStore | None) -> None:
        after = self.after
        sense_store.update(
            self.form,
            lambda e: e.model_copy(update={"senses": after}),  # type: ignore[union-attr]
        )


class PosTagRequest(BaseModel):
    type: Literal["pos_tag"] = "pos_tag"
    id: str
    created_at: datetime
    form: str
    before: list[Sense]
    after: list[Sense]

    def apply(self, sense_store: SenseStore, occ_store: OccurrenceStore | None) -> None:
        after = self.after
        sense_store.update(
            self.form,
            lambda e: e.model_copy(update={"senses": after}),  # type: ignore[union-attr]
        )


class UpdatePosRequest(BaseModel):
    type: Literal["update_pos"] = "update_pos"
    id: str
    created_at: datetime
    form: str
    before: list[Sense]
    after: list[Sense]

    def apply(self, sense_store: SenseStore, occ_store: OccurrenceStore | None) -> None:
        after = self.after
        sense_store.update(
            self.form,
            lambda e: e.model_copy(update={"senses": after}),  # type: ignore[union-attr]
        )


class PruneRequest(BaseModel):
    type: Literal["prune"] = "prune"
    id: str
    created_at: datetime
    form: str
    before: list[Sense]
    after: list[Sense]
    removed_ids: list[str]

    def apply(self, sense_store: SenseStore, occ_store: OccurrenceStore | None) -> None:
        after = self.after
        sense_store.update(
            self.form,
            lambda e: e.model_copy(update={"senses": after}),  # type: ignore[union-attr]
        )
        if occ_store is not None:
            for sid in self.removed_ids:
                occ_store.delete_by_sense_id(self.form, sid)


class TrimSenseRequest(BaseModel):
    type: Literal["trim_sense"] = "trim_sense"
    id: str
    created_at: datetime
    form: str
    before: list[Sense]
    after: list[Sense]
    sense_id: str
    reason: str

    def apply(self, sense_store: SenseStore, occ_store: OccurrenceStore | None) -> None:
        after = self.after
        sense_store.update(
            self.form,
            lambda e: e.model_copy(update={"senses": after}),  # type: ignore[union-attr]
        )
        if occ_store is not None:
            occ_store.delete_by_sense_id(self.form, self.sense_id)


class MorphRedirectRequest(BaseModel):
    type: Literal["morph_redirect"] = "morph_redirect"
    id: str
    created_at: datetime
    form: str
    derived_sense_idx: int
    base_form: str
    base_sense_idx: int
    relation: str
    before: Sense
    after: Sense

    def apply(self, sense_store: SenseStore, occ_store: OccurrenceStore | None) -> None:
        idx = self.derived_sense_idx
        after_sense = self.after

        def apply_fn(existing: Alf | None) -> Alf:
            assert existing is not None
            senses = list(existing.senses)
            senses[idx] = after_sense
            return existing.model_copy(update={"senses": senses})

        sense_store.update(self.form, apply_fn)


class SetRedirectRequest(BaseModel):
    type: Literal["set_redirect"] = "set_redirect"
    id: str
    created_at: datetime
    form: str
    redirect_to: str

    def apply(self, sense_store: SenseStore, occ_store: OccurrenceStore | None) -> None:
        redirect_to = self.redirect_to
        form = self.form

        def set_redirect(existing: Alf | None) -> Alf:
            base = existing if existing is not None else Alf(form=form)
            return base.model_copy(update={"redirect": redirect_to})

        sense_store.update(self.form, set_redirect)


class DeleteEntryRequest(BaseModel):
    type: Literal["delete_entry"] = "delete_entry"
    form: str
    reason: str

    def apply(self, sense_store: SenseStore, occ_store: OccurrenceStore | None) -> None:
        if occ_store is not None:
            occ_store.delete_by_form(self.form)
        sense_store.delete(self.form)


ChangeRequest = Annotated[
    AddSensesRequest
    | RewriteRequest
    | PosTagRequest
    | UpdatePosRequest
    | PruneRequest
    | TrimSenseRequest
    | MorphRedirectRequest
    | SetRedirectRequest
    | DeleteEntryRequest,
    Field(discriminator="type"),
]
