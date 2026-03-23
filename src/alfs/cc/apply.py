"""Apply CC skill outputs as clerk requests.

Usage:
    python -m alfs.cc.apply \\
        --cc-tasks-dir ../cc_tasks \\
        --senses-db ../alfs_data/senses.db \\
        --labeled-db ../alfs_data/labeled.db \\
        --queue-dir ../clerk_queue
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import uuid

from pydantic import TypeAdapter

from alfs.cc.models import (
    CCDeleteEntryOutput,
    CCInductionOutput,
    CCMorphRedirectOutput,
    CCOutput,
    CCRewriteOutput,
    CCSpellingVariantOutput,
    CCTrimSenseOutput,
)
from alfs.clerk.queue import enqueue
from alfs.clerk.request import (
    AddSensesRequest,
    DeleteEntryRequest,
    MorphRedirectRequest,
    RewriteRequest,
    SetSpellingVariantRequest,
    TrimSenseRequest,
)
from alfs.data_models.alf import Sense
from alfs.data_models.pos import PartOfSpeech
from alfs.data_models.sense_store import SenseStore

_output_adapter: TypeAdapter[CCOutput] = TypeAdapter(CCOutput)


def _apply_induction(
    output: CCInductionOutput,
    sense_store: SenseStore,
    queue_dir: Path,
) -> bool:
    entry = sense_store.read(output.form)
    existing_defs = (
        {s.definition.strip().lower() for s in entry.senses} if entry else set()
    )

    new_senses: list[Sense] = []
    for s in output.senses:
        if s.definition.strip().lower() in existing_defs:
            continue
        try:
            pos = PartOfSpeech(s.pos) if s.pos else None
        except ValueError:
            pos = None
        new_senses.append(
            Sense(definition=s.definition, pos=pos, updated_by_model="claude-code")
        )

    if not new_senses:
        print(f"  skipped induction for {output.form!r}: no new senses")
        return True

    request = AddSensesRequest(
        id=str(uuid.uuid4()),
        created_at=datetime.now(UTC),
        form=output.form,
        new_senses=new_senses,
    )
    enqueue(request, queue_dir)
    print(f"  queued {len(new_senses)} new sense(s) for {output.form!r}")
    return True


def _apply_rewrite(
    output: CCRewriteOutput,
    sense_store: SenseStore,
    queue_dir: Path,
) -> bool:
    entry = sense_store.read(output.form)
    if not entry or not entry.senses:
        print(f"  skipped rewrite for {output.form!r}: no entry in store")
        return False

    if len(output.senses) != len(entry.senses):
        print(
            f"  skipped rewrite for {output.form!r}: sense count mismatch"
            f" ({len(output.senses)} vs {len(entry.senses)})"
        )
        return False

    after = [
        Sense(
            id=entry.senses[i].id,
            definition=s.definition,
            subsenses=s.subsenses or None,
            pos=entry.senses[i].pos,
            updated_by_model="claude-code",
        )
        for i, s in enumerate(output.senses)
    ]

    request = RewriteRequest(
        id=str(uuid.uuid4()),
        created_at=datetime.now(UTC),
        form=output.form,
        before=list(entry.senses),
        after=after,
        requesting_model="claude-code",
    )
    enqueue(request, queue_dir)
    print(f"  queued rewrite for {output.form!r}")
    return True


def _apply_trim_sense(
    output: CCTrimSenseOutput,
    sense_store: SenseStore,
    queue_dir: Path,
) -> bool:
    if output.sense_num is None:
        print(f"  skipped trim for {output.form!r}: all senses distinct")
        return True

    entry = sense_store.read(output.form)
    if not entry or not entry.senses:
        print(f"  skipped trim for {output.form!r}: no entry in store")
        return False

    if not (1 <= output.sense_num <= len(entry.senses)):
        print(
            f"  skipped trim for {output.form!r}: sense_num {output.sense_num}"
            f" out of range (have {len(entry.senses)})"
        )
        return False

    deleted_sense = entry.senses[output.sense_num - 1]
    remaining = [s for s in entry.senses if s.id != deleted_sense.id]

    request = TrimSenseRequest(
        id=str(uuid.uuid4()),
        created_at=datetime.now(UTC),
        form=output.form,
        before=list(entry.senses),
        after=remaining,
        sense_id=deleted_sense.id,
        reason=output.reason,
        requesting_model="claude-code",
    )
    enqueue(request, queue_dir)
    print(f"  queued trim for {output.form!r}: sense {output.sense_num}")
    return True


def _apply_morph_redirect(
    output: CCMorphRedirectOutput,
    sense_store: SenseStore,
    queue_dir: Path,
) -> bool:
    queued = 0
    for rel in output.relations:
        derived_entry = sense_store.read(rel.derived_form)
        base_entry = sense_store.read(rel.base_form)

        if not derived_entry or not derived_entry.senses:
            print(f"  skipped morph: {rel.derived_form!r} not in store")
            continue
        if not base_entry or not base_entry.senses:
            print(f"  skipped morph: {rel.base_form!r} not in store")
            continue

        if rel.derived_sense_idx < 0 or rel.derived_sense_idx >= len(
            derived_entry.senses
        ):
            print(
                f"  skipped morph: {rel.derived_form!r} sense {rel.derived_sense_idx}"
                f" out of range"
            )
            continue
        if rel.base_sense_idx < 0 or rel.base_sense_idx >= len(base_entry.senses):
            print(
                f"  skipped morph: {rel.base_form!r} sense {rel.base_sense_idx}"
                f" out of range"
            )
            continue

        derived_sense = derived_entry.senses[rel.derived_sense_idx]
        after_sense = derived_sense.model_copy(
            update={
                "definition": rel.proposed_definition,
                "morph_base": rel.base_form,
                "morph_relation": rel.relation,
                "updated_by_model": "claude-code",
            }
        )

        request = MorphRedirectRequest(
            id=str(uuid.uuid4()),
            created_at=datetime.now(UTC),
            form=rel.derived_form,
            derived_sense_idx=rel.derived_sense_idx,
            base_form=rel.base_form,
            base_sense_idx=rel.base_sense_idx,
            relation=rel.relation,
            before=derived_sense,
            after=after_sense,
            promote_to_parent=rel.promote_to_parent,
        )
        enqueue(request, queue_dir)
        queued += 1

    if queued:
        print(f"  queued {queued} morph redirect(s)")
    return True


def _apply_spelling_variant(
    output: CCSpellingVariantOutput,
    sense_store: SenseStore,
    queue_dir: Path,
) -> bool:
    queued = 0
    for pair in output.confirmed:
        variant_entry = sense_store.read(pair.variant_form)
        preferred_entry = sense_store.read(pair.preferred_form)

        if not variant_entry:
            print(f"  skipped spelling variant: {pair.variant_form!r} not in store")
            continue
        if not preferred_entry:
            print(f"  skipped spelling variant: {pair.preferred_form!r} not in store")
            continue

        request = SetSpellingVariantRequest(
            id=str(uuid.uuid4()),
            created_at=datetime.now(UTC),
            form=pair.variant_form,
            preferred_form=pair.preferred_form,
        )
        enqueue(request, queue_dir)
        queued += 1

    if queued:
        print(f"  queued {queued} spelling variant link(s)")
    return True


def _apply_delete_entry(output: CCDeleteEntryOutput, queue_dir: Path) -> bool:
    if not output.should_delete:
        print(f"  skipped delete for {output.form!r}: judged worth keeping")
        return True
    request = DeleteEntryRequest(
        id=str(uuid.uuid4()),
        created_at=datetime.now(UTC),
        form=output.form,
        reason=output.reason,
        requesting_model="claude-code",
    )
    enqueue(request, queue_dir)
    print(f"  queued delete for {output.form!r}")
    return True


def run(
    cc_tasks_dir: str | Path,
    senses_db: str | Path,
    queue_dir: str | Path,
) -> None:
    done_dir = Path(cc_tasks_dir) / "done"
    if not done_dir.exists():
        print("No done/ directory found.")
        return

    sense_store = SenseStore(Path(senses_db))
    queue_path = Path(queue_dir)

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
            ok = _apply_induction(output, sense_store, queue_path)
        elif isinstance(output, CCRewriteOutput):
            ok = _apply_rewrite(output, sense_store, queue_path)
        elif isinstance(output, CCTrimSenseOutput):
            ok = _apply_trim_sense(output, sense_store, queue_path)
        elif isinstance(output, CCMorphRedirectOutput):
            ok = _apply_morph_redirect(output, sense_store, queue_path)
        elif isinstance(output, CCSpellingVariantOutput):
            ok = _apply_spelling_variant(output, sense_store, queue_path)
        elif isinstance(output, CCDeleteEntryOutput):
            ok = _apply_delete_entry(output, queue_path)
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
    args = parser.parse_args()

    run(args.cc_tasks_dir, args.senses_db, args.queue_dir)


if __name__ == "__main__":
    main()
