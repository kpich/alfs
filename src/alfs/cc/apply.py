"""Apply CC skill outputs as clerk requests.

Usage:
    python -m alfs.cc.apply \\
        --cc-tasks-dir ../cc_tasks \\
        --senses-db ../alfs_data/senses.db \\
        --queue-dir ../clerk_queue \\
        [--labeled-db ../alfs_data/labeled.db] \\
        [--blocklist-file ../alfs_data/blocklist.yaml]
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import uuid

from pydantic import TypeAdapter

from alfs.cc.models import (
    CCInductionOutput,
    CCMorphRelBlockOutput,
    CCOutput,
    CCQCOutput,
)
from alfs.clerk.queue import enqueue
from alfs.clerk.request import (
    AddSensesRequest,
    DeleteEntryRequest,
    MorphRedirectRequest,
    PruneRequest,
    RewriteRequest,
    SetSpellingVariantRequest,
    UpdatePosRequest,
)
from alfs.data_models.alf import Sense
from alfs.data_models.blocklist import Blocklist
from alfs.data_models.occurrence import Occurrence
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.pos import PartOfSpeech
from alfs.data_models.sense_store import SenseStore

_output_adapter: TypeAdapter[CCOutput] = TypeAdapter(CCOutput)
_SKIP_SENSE_KEY = "_skip"

_MORPH_TEMPLATES: dict[str, str] = {
    "plural": "Plural of {base}.",
    "3sg_present": "Third-person singular present tense of {base}.",
    "past_tense": "Past tense of {base}.",
    "past_participle": "Past participle of {base}.",
    "present_participle": "Present participle of {base}.",
    "comparative": "Comparative form of {base}.",
    "superlative": "Superlative form of {base}.",
}


def _morph_definition(relation: str, base_form: str) -> str:
    template = _MORPH_TEMPLATES.get(relation)
    if template is not None:
        return template.format(base=base_form)
    return f"{relation.replace('_', ' ').capitalize()} of {base_form}."


def _apply_induction(
    output: CCInductionOutput,
    sense_store: SenseStore,
    queue_dir: Path,
    occ_store: OccurrenceStore | None,
    blocklist: Blocklist | None,
) -> bool:
    # Handle blocklist decision
    if output.add_to_blocklist:
        if blocklist is not None:
            blocklist.add(output.form, output.blocklist_reason)
            print(f"  added {output.form!r} to blocklist: {output.blocklist_reason}")
        if occ_store is not None:
            occ_store.delete_by_form(output.form)
            print(f"  deleted labeled occurrences for {output.form!r}")
        return True

    # Apply context labels
    if output.context_labels and occ_store is not None:
        occurrence_refs: list[Occurrence] = output.occurrence_refs
        if not occurrence_refs:
            print(
                f"  warning: no occurrence_refs in output for {output.form!r}, "
                f"skipping context labels"
            )
        else:
            rows = []
            for label in output.context_labels:
                idx = label.context_idx
                if idx < 0 or idx >= len(occurrence_refs):
                    continue
                occ = occurrence_refs[idx]
                if label.sense_idx is None:
                    # _skip label
                    rows.append(
                        (
                            output.form,
                            occ.doc_id,
                            occ.byte_offset,
                            _SKIP_SENSE_KEY,
                            0,
                            None,
                        )
                    )
                else:
                    # Sense assignment (1-indexed into new_senses)
                    sense_key = str(label.sense_idx)
                    rows.append(
                        (
                            output.form,
                            occ.doc_id,
                            occ.byte_offset,
                            sense_key,
                            2,
                            None,
                        )
                    )
            if rows:
                occ_store.upsert_many(rows, "claude-code")
                print(f"  labeled {len(rows)} occurrence(s) for {output.form!r}")

    # Enqueue new senses
    entry = sense_store.read(output.form)
    existing_defs = (
        {s.definition.strip().lower() for s in entry.senses} if entry else set()
    )

    new_senses: list[Sense] = []
    # base_form -> list of senses to add to the base form
    base_senses: dict[str, list[Sense]] = {}

    for s in output.new_senses:
        try:
            pos = PartOfSpeech(s.pos) if s.pos else None
        except ValueError:
            pos = None

        if s.morph_rel is not None:
            # Add semantic sense to the base form
            mr = s.morph_rel
            base_entry = sense_store.read(mr.base_form)
            base_existing = (
                {b.definition.strip().lower() for b in base_entry.senses}
                if base_entry
                else set()
            )
            if s.definition.strip().lower() not in base_existing:
                base_senses.setdefault(mr.base_form, []).append(
                    Sense(
                        definition=s.definition,
                        pos=pos,
                        updated_by_model="claude-code",
                    )
                )
            # Add morph-redirect sense to derived form
            morph_def = _morph_definition(mr.relation, mr.base_form)
            if morph_def.strip().lower() not in existing_defs:
                new_senses.append(
                    Sense(
                        definition=morph_def,
                        pos=pos,
                        morph_base=mr.base_form,
                        morph_relation=mr.relation,
                        updated_by_model="claude-code",
                    )
                )
        else:
            if s.definition.strip().lower() in existing_defs:
                continue
            new_senses.append(
                Sense(definition=s.definition, pos=pos, updated_by_model="claude-code")
            )

    for base_form, senses in base_senses.items():
        enqueue(
            AddSensesRequest(
                id=str(uuid.uuid4()),
                created_at=datetime.now(UTC),
                form=base_form,
                new_senses=senses,
            ),
            queue_dir,
        )
        print(f"  queued {len(senses)} new sense(s) for base form {base_form!r}")

    if not new_senses and not base_senses:
        print(f"  skipped induction for {output.form!r}: no new senses")
        return True

    if new_senses:
        request = AddSensesRequest(
            id=str(uuid.uuid4()),
            created_at=datetime.now(UTC),
            form=output.form,
            new_senses=new_senses,
        )
        enqueue(request, queue_dir)
        print(f"  queued {len(new_senses)} new sense(s) for {output.form!r}")
    return True


def _apply_morphrel_block(
    output: CCMorphRelBlockOutput,
    sense_store: SenseStore,
    queue_dir: Path,
    occ_store: OccurrenceStore | None,
    blocklist: Blocklist | None,
) -> bool:
    form = output.form

    if output.action == "morph_rel":
        alf = sense_store.read(form)
        if alf is None:
            print(f"  skipped morph_rel for {form!r}: form not found in sense store")
            return True
        for entry in output.morph_rels:
            if sense_store.read(entry.morph_base) is None:
                print(
                    f"  skipped morph_rel sense {entry.sense_idx} for {form!r}: "
                    f"base {entry.morph_base!r} not in sense store"
                )
                continue
            if entry.sense_idx >= len(alf.senses):
                print(
                    f"  skipped morph_rel sense {entry.sense_idx} for {form!r}: "
                    f"index out of range"
                )
                continue
            before = alf.senses[entry.sense_idx]
            after = before.model_copy(
                update={
                    "definition": entry.proposed_definition,
                    "morph_base": entry.morph_base,
                    "morph_relation": entry.morph_relation,
                    "updated_by_model": "claude-code",
                }
            )
            request = MorphRedirectRequest(
                id=str(uuid.uuid4()),
                created_at=datetime.now(UTC),
                form=form,
                derived_sense_idx=entry.sense_idx,
                base_form=entry.morph_base,
                base_sense_idx=0,
                relation=entry.morph_relation,
                before=before,
                after=after,
                promote_to_parent=entry.promote_to_parent,
            )
            enqueue(request, queue_dir)
            print(
                f"  queued morph_rel sense {entry.sense_idx} for {form!r}"
                f" → {entry.morph_base!r}"
            )

    elif output.action == "delete":
        reason = output.blocklist_reason or "deleted by cc-morphrel-redirect-block"
        if blocklist is not None:
            blocklist.add(form, reason)
            print(f"  added {form!r} to blocklist: {reason}")
        if occ_store is not None:
            occ_store.delete_by_form(form)
            print(f"  deleted labeled occurrences for {form!r}")
        enqueue(
            DeleteEntryRequest(
                id=str(uuid.uuid4()),
                created_at=datetime.now(UTC),
                form=form,
                reason=reason,
                requesting_model="claude-code",
            ),
            queue_dir,
        )
        print(f"  queued delete for {form!r}")

    elif output.action == "normalize_case":
        canonical = output.canonical_form
        if not canonical:
            print(f"  skipped normalize_case for {form!r}: no canonical_form specified")
            return True
        reason = f"case variant of {canonical}"
        if occ_store is not None:
            occ_store.delete_by_form(form)
            print(f"  deleted labeled occurrences for {form!r}")
        existing_alf = sense_store.read(form)
        if sense_store.read(canonical) is None and existing_alf is not None:
            enqueue(
                AddSensesRequest(
                    id=str(uuid.uuid4()),
                    created_at=datetime.now(UTC),
                    form=canonical,
                    new_senses=list(existing_alf.senses),
                ),
                queue_dir,
            )
            print(f"  queued migrate senses from {form!r} to {canonical!r}")
        else:
            print(f"  {canonical!r} already exists; dropping {form!r}")
        enqueue(
            DeleteEntryRequest(
                id=str(uuid.uuid4()),
                created_at=datetime.now(UTC),
                form=form,
                reason=reason,
                requesting_model="claude-code",
            ),
            queue_dir,
        )
        print(f"  queued delete for {form!r}")

    return True


def _apply_qc(
    output: CCQCOutput,
    sense_store: SenseStore,
    queue_dir: Path,
    occ_store: OccurrenceStore | None,
    blocklist: Blocklist | None,
) -> bool:
    form = output.form

    if output.delete_entry:
        reason = output.delete_entry_reason or "deleted by cc-qc"
        if blocklist is not None:
            blocklist.add(form, reason)
            print(f"  added {form!r} to blocklist: {reason}")
        if occ_store is not None:
            occ_store.delete_by_form(form)
            print(f"  deleted labeled occurrences for {form!r}")
        enqueue(
            DeleteEntryRequest(
                id=str(uuid.uuid4()),
                created_at=datetime.now(UTC),
                form=form,
                reason=reason,
                requesting_model="claude-code",
            ),
            queue_dir,
        )
        print(f"  queued delete for {form!r}")
        return True

    if output.normalize_case:
        canonical = output.normalize_case
        reason = f"case variant of {canonical}"
        if occ_store is not None:
            occ_store.delete_by_form(form)
            print(f"  deleted labeled occurrences for {form!r}")
        existing_alf = sense_store.read(form)
        if sense_store.read(canonical) is None and existing_alf is not None:
            enqueue(
                AddSensesRequest(
                    id=str(uuid.uuid4()),
                    created_at=datetime.now(UTC),
                    form=canonical,
                    new_senses=list(existing_alf.senses),
                ),
                queue_dir,
            )
            print(f"  queued migrate senses from {form!r} to {canonical!r}")
        else:
            print(f"  {canonical!r} already exists; dropping {form!r}")
        enqueue(
            DeleteEntryRequest(
                id=str(uuid.uuid4()),
                created_at=datetime.now(UTC),
                form=form,
                reason=reason,
                requesting_model="claude-code",
            ),
            queue_dir,
        )
        print(f"  queued delete for {form!r}")
        return True

    if output.spelling_variant_of:
        preferred = output.spelling_variant_of
        enqueue(
            SetSpellingVariantRequest(
                id=str(uuid.uuid4()),
                created_at=datetime.now(UTC),
                form=form,
                preferred_form=preferred,
            ),
            queue_dir,
        )
        print(f"  queued spelling_variant {form!r} → {preferred!r}")
        return True

    alf = sense_store.read(form)
    if alf is None:
        print(f"  skipped qc for {form!r}: form not found in sense store")
        return True

    for entry in output.morph_rels:
        if sense_store.read(entry.morph_base) is None:
            print(
                f"  skipped morph_rel sense {entry.sense_idx} for {form!r}: "
                f"base {entry.morph_base!r} not in sense store"
            )
            continue
        if entry.sense_idx >= len(alf.senses):
            print(
                f"  skipped morph_rel sense {entry.sense_idx} for {form!r}: "
                f"index out of range"
            )
            continue
        before = alf.senses[entry.sense_idx]
        after = before.model_copy(
            update={
                "definition": entry.proposed_definition,
                "morph_base": entry.morph_base,
                "morph_relation": entry.morph_relation,
                "updated_by_model": "claude-code",
            }
        )
        enqueue(
            MorphRedirectRequest(
                id=str(uuid.uuid4()),
                created_at=datetime.now(UTC),
                form=form,
                derived_sense_idx=entry.sense_idx,
                base_form=entry.morph_base,
                base_sense_idx=0,
                relation=entry.morph_relation,
                before=before,
                after=after,
                promote_to_parent=entry.promote_to_parent,
            ),
            queue_dir,
        )
        print(
            f"  queued morph_rel sense {entry.sense_idx} for {form!r}"
            f" → {entry.morph_base!r}"
        )

    for rw in output.sense_rewrites:
        if rw.sense_idx >= len(alf.senses):
            print(
                f"  skipped rewrite sense {rw.sense_idx} for {form!r}: "
                f"index out of range"
            )
            continue
        before = alf.senses[rw.sense_idx]
        after = before.model_copy(
            update={"definition": rw.definition, "updated_by_model": "claude-code"}
        )
        enqueue(
            RewriteRequest(
                id=str(uuid.uuid4()),
                created_at=datetime.now(UTC),
                form=form,
                before=before,
                after=after,
                requesting_model="claude-code",
            ),
            queue_dir,
        )
        print(f"  queued rewrite sense {rw.sense_idx} for {form!r}")

    for pc in output.pos_corrections:
        if pc.sense_idx >= len(alf.senses):
            print(
                f"  skipped pos_correction sense {pc.sense_idx} for {form!r}: "
                f"index out of range"
            )
            continue
        try:
            pos = PartOfSpeech(pc.pos)
        except ValueError:
            print(
                f"  skipped pos_correction sense {pc.sense_idx} for {form!r}: "
                f"invalid pos {pc.pos!r}"
            )
            continue
        before = alf.senses[pc.sense_idx]
        after = before.model_copy(
            update={"pos": pos, "updated_by_model": "claude-code"}
        )
        enqueue(
            UpdatePosRequest(
                id=str(uuid.uuid4()),
                created_at=datetime.now(UTC),
                form=form,
                before=before,
                after=after,
                requesting_model="claude-code",
            ),
            queue_dir,
        )
        print(f"  queued pos_correction sense {pc.sense_idx} for {form!r}")

    if output.deleted_senses:
        deleted_idxs = {d.sense_idx for d in output.deleted_senses}
        invalid = sorted(i for i in deleted_idxs if i >= len(alf.senses))
        if invalid:
            print(
                f"  skipping out-of-range deleted_senses indices {invalid} for {form!r}"
            )
            deleted_idxs -= set(invalid)
        if deleted_idxs:
            deleted_ids = [alf.senses[i].id for i in sorted(deleted_idxs)]
            before_list = list(alf.senses)
            after_list = [s for i, s in enumerate(alf.senses) if i not in deleted_idxs]
            reasons = "; ".join(
                f"sense {d.sense_idx}: {d.reason}"
                for d in output.deleted_senses
                if d.sense_idx in deleted_idxs
            )
            enqueue(
                PruneRequest(
                    id=str(uuid.uuid4()),
                    created_at=datetime.now(UTC),
                    form=form,
                    before=before_list,
                    after=after_list,
                    removed_ids=deleted_ids,
                    requesting_model="claude-code",
                ),
                queue_dir,
            )
            print(
                f"  queued prune {len(deleted_idxs)} sense(s) for {form!r}: {reasons}"
            )

    return True


def run(
    cc_tasks_dir: str | Path,
    senses_db: str | Path,
    queue_dir: str | Path,
    labeled_db: str | Path | None = None,
    blocklist_file: str | Path | None = None,
) -> None:
    done_dir = Path(cc_tasks_dir) / "done"
    if not done_dir.exists():
        print("No done/ directory found.")
        return

    sense_store = SenseStore(Path(senses_db))
    queue_path = Path(queue_dir)

    occ_store: OccurrenceStore | None = None
    if labeled_db:
        occ_store = OccurrenceStore(Path(labeled_db))

    blocklist: Blocklist | None = None
    if blocklist_file:
        blocklist = Blocklist(Path(blocklist_file))

    files = sorted(done_dir.glob("*/*.json"))
    if not files:
        print("No output files in done/.")
        return

    for f in files:
        try:
            output = _output_adapter.validate_json(f.read_bytes())
        except Exception as exc:
            print(f"  error parsing {f.name}: {exc}")
            continue

        ok: bool
        if isinstance(output, CCInductionOutput):
            ok = _apply_induction(
                output,
                sense_store,
                queue_path,
                occ_store,
                blocklist,
            )
        elif isinstance(output, CCMorphRelBlockOutput):
            ok = _apply_morphrel_block(
                output,
                sense_store,
                queue_path,
                occ_store,
                blocklist,
            )
        elif isinstance(output, CCQCOutput):
            ok = _apply_qc(
                output,
                sense_store,
                queue_path,
                occ_store,
                blocklist,
            )
        else:
            print(f"  unknown output type in {f.name}")
            ok = False

        if ok:
            f.unlink()

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply CC skill outputs as clerk requests"
    )
    parser.add_argument(
        "--cc-tasks-dir", required=True, help="Path to cc_tasks directory"
    )
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument(
        "--queue-dir", required=True, help="Path to clerk queue directory"
    )
    parser.add_argument(
        "--labeled-db", default=None, help="Path to labeled.db (optional)"
    )
    parser.add_argument(
        "--blocklist-file", default=None, help="Path to blocklist.yaml (optional)"
    )
    args = parser.parse_args()

    run(
        args.cc_tasks_dir,
        args.senses_db,
        args.queue_dir,
        args.labeled_db,
        args.blocklist_file,
    )


if __name__ == "__main__":
    main()
