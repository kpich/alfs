"""Typed change request classes for the clerk queue."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from alfs.data_models.alf import Alf, Sense
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore
from alfs.llm_ordering import can_overwrite

_log = logging.getLogger(__name__)


class AddSensesRequest(BaseModel):
    type: Literal["add_senses"] = "add_senses"
    id: str
    created_at: datetime
    form: str
    new_senses: list[Sense]

    def apply(self, sense_store: SenseStore, occ_store: OccurrenceStore | None) -> bool:
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
        return True


class RewriteRequest(BaseModel):
    type: Literal["rewrite"] = "rewrite"
    id: str
    created_at: datetime
    form: str
    before: Sense
    after: Sense
    requesting_model: str | None = None

    def apply(self, sense_store: SenseStore, occ_store: OccurrenceStore | None) -> bool:
        if not can_overwrite(self.requesting_model, self.before.updated_by_model):
            _log.warning(
                "Skipping %s for %r: %r cannot overwrite %r",
                self.type,
                self.form,
                self.requesting_model,
                self.before.updated_by_model,
            )
            return False
        after = self.after
        before_id = self.before.id
        sense_store.update(
            self.form,
            lambda e: e.model_copy(  # type: ignore[union-attr]
                update={"senses": [after if s.id == before_id else s for s in e.senses]}  # type: ignore[union-attr]
            ),
        )
        return True


class PosTagRequest(BaseModel):
    type: Literal["pos_tag"] = "pos_tag"
    id: str
    created_at: datetime
    form: str
    before: Sense
    after: Sense
    requesting_model: str | None = None

    def apply(self, sense_store: SenseStore, occ_store: OccurrenceStore | None) -> bool:
        if not can_overwrite(self.requesting_model, self.before.updated_by_model):
            _log.warning(
                "Skipping %s for %r: %r cannot overwrite %r",
                self.type,
                self.form,
                self.requesting_model,
                self.before.updated_by_model,
            )
            return False
        after = self.after
        before_id = self.before.id
        sense_store.update(
            self.form,
            lambda e: e.model_copy(  # type: ignore[union-attr]
                update={"senses": [after if s.id == before_id else s for s in e.senses]}  # type: ignore[union-attr]
            ),
        )
        return True


class UpdatePosRequest(BaseModel):
    type: Literal["update_pos"] = "update_pos"
    id: str
    created_at: datetime
    form: str
    before: Sense
    after: Sense
    requesting_model: str | None = None

    def apply(self, sense_store: SenseStore, occ_store: OccurrenceStore | None) -> bool:
        if not can_overwrite(self.requesting_model, self.before.updated_by_model):
            _log.warning(
                "Skipping %s for %r: %r cannot overwrite %r",
                self.type,
                self.form,
                self.requesting_model,
                self.before.updated_by_model,
            )
            return False
        after = self.after
        before_id = self.before.id
        sense_store.update(
            self.form,
            lambda e: e.model_copy(  # type: ignore[union-attr]
                update={"senses": [after if s.id == before_id else s for s in e.senses]}  # type: ignore[union-attr]
            ),
        )
        return True


class PruneRequest(BaseModel):
    type: Literal["prune"] = "prune"
    id: str
    created_at: datetime
    form: str
    before: list[Sense]
    after: list[Sense]
    removed_ids: list[str]
    requesting_model: str | None = None

    def apply(self, sense_store: SenseStore, occ_store: OccurrenceStore | None) -> bool:
        removed_set = set(self.removed_ids)
        for sense in self.before:
            if sense.id in removed_set and not can_overwrite(
                self.requesting_model, sense.updated_by_model
            ):
                _log.warning(
                    "Skipping %s for %r: %r cannot overwrite %r",
                    self.type,
                    self.form,
                    self.requesting_model,
                    sense.updated_by_model,
                )
                return False
        after = self.after
        sense_store.update(
            self.form,
            lambda e: e.model_copy(update={"senses": after}),  # type: ignore[union-attr]
        )
        if occ_store is not None:
            for sid in self.removed_ids:
                occ_store.delete_by_sense_id(self.form, sid)
        return True


class TrimSenseRequest(BaseModel):
    type: Literal["trim_sense"] = "trim_sense"
    id: str
    created_at: datetime
    form: str
    before: list[Sense]
    after: list[Sense]
    sense_id: str
    reason: str
    requesting_model: str | None = None

    def apply(self, sense_store: SenseStore, occ_store: OccurrenceStore | None) -> bool:
        for sense in self.before:
            if sense.id == self.sense_id:
                if not can_overwrite(self.requesting_model, sense.updated_by_model):
                    _log.warning(
                        "Skipping %s for %r: %r cannot overwrite %r",
                        self.type,
                        self.form,
                        self.requesting_model,
                        sense.updated_by_model,
                    )
                    return False
                break
        after = self.after
        sense_store.update(
            self.form,
            lambda e: e.model_copy(update={"senses": after}),  # type: ignore[union-attr]
        )
        if occ_store is not None:
            occ_store.delete_by_sense_id(self.form, self.sense_id)
        return True


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
    promote_to_parent: bool = True

    def apply(self, sense_store: SenseStore, occ_store: OccurrenceStore | None) -> bool:
        idx = self.derived_sense_idx
        after_sense = self.after
        before_sense = self.before
        base_form = self.base_form

        def apply_fn(existing: Alf | None) -> Alf:
            assert existing is not None
            senses = list(existing.senses)
            senses[idx] = after_sense
            return existing.model_copy(update={"senses": senses})

        sense_store.update(self.form, apply_fn)

        if self.promote_to_parent:
            # Promote original sense content to parent.
            # morph_base/morph_relation/updated_by_model/updated_at are not copied —
            # the promoted sense is an independent sense on the base form.
            promoted = Sense(
                definition=before_sense.definition,
                pos=before_sense.pos,
            )

            def add_to_parent(parent: Alf | None) -> Alf:
                if parent is None:
                    return Alf(form=base_form, senses=[promoted])
                return parent.model_copy(
                    update={"senses": list(parent.senses) + [promoted]}
                )

            sense_store.update(base_form, add_to_parent)
        return True


class SetRedirectRequest(BaseModel):
    type: Literal["set_redirect"] = "set_redirect"
    id: str
    created_at: datetime
    form: str
    redirect_to: str

    def apply(self, sense_store: SenseStore, occ_store: OccurrenceStore | None) -> bool:
        redirect_to = self.redirect_to
        form = self.form

        def set_redirect(existing: Alf | None) -> Alf:
            base = existing if existing is not None else Alf(form=form)
            return base.model_copy(update={"redirect": redirect_to})

        sense_store.update(self.form, set_redirect)
        return True


class SetSpellingVariantRequest(BaseModel):
    type: Literal["set_spelling_variant"] = "set_spelling_variant"
    id: str
    created_at: datetime
    form: str
    preferred_form: str

    def apply(self, sense_store: SenseStore, occ_store: OccurrenceStore | None) -> bool:
        preferred = self.preferred_form
        sense_store.update(
            self.form,
            lambda e: e.model_copy(update={"spelling_variant_of": preferred}),  # type: ignore[union-attr]
        )
        return True


class ClearRedirectSensesRequest(BaseModel):
    type: Literal["clear_redirect_senses"] = "clear_redirect_senses"
    id: str
    created_at: datetime
    form: str

    def apply(self, sense_store: SenseStore, occ_store: OccurrenceStore | None) -> bool:
        sense_store.update(self.form, lambda e: e.model_copy(update={"senses": []}))  # type: ignore[union-attr]
        return True


class DeleteEntryRequest(BaseModel):
    type: Literal["delete_entry"] = "delete_entry"
    id: str
    created_at: datetime
    form: str
    reason: str
    requesting_model: str | None = None

    def apply(self, sense_store: SenseStore, occ_store: OccurrenceStore | None) -> bool:
        existing = sense_store.read(self.form)
        if existing is not None:
            for sense in existing.senses:
                if not can_overwrite(self.requesting_model, sense.updated_by_model):
                    _log.warning(
                        "Skipping delete_entry for %r: %r cannot overwrite %r",
                        self.form,
                        self.requesting_model,
                        sense.updated_by_model,
                    )
                    return False
        if occ_store is not None:
            occ_store.delete_by_form(self.form)
        sense_store.delete(self.form)
        return True


ChangeRequest = Annotated[
    AddSensesRequest
    | RewriteRequest
    | PosTagRequest
    | UpdatePosRequest
    | PruneRequest
    | TrimSenseRequest
    | MorphRedirectRequest
    | SetRedirectRequest
    | SetSpellingVariantRequest
    | ClearRedirectSensesRequest
    | DeleteEntryRequest,
    Field(discriminator="type"),
]
