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

from alfs.cc.models import CCInductionOutput, CCOutput
from alfs.clerk.queue import enqueue
from alfs.clerk.request import AddSensesRequest
from alfs.data_models.alf import Sense
from alfs.data_models.blocklist import Blocklist
from alfs.data_models.occurrence import Occurrence
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.pos import PartOfSpeech
from alfs.data_models.sense_store import SenseStore

_output_adapter: TypeAdapter[CCOutput] = TypeAdapter(CCOutput)
_SKIP_SENSE_KEY = "_skip"


def _apply_induction(
    output: CCInductionOutput,
    sense_store: SenseStore,
    queue_dir: Path,
    occ_store: OccurrenceStore | None,
    blocklist: Blocklist | None,
    pending_induction_dir: Path,
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

    # Apply context labels (requires the original task file for occurrence_refs)
    if output.context_labels and occ_store is not None:
        task_path = pending_induction_dir / f"{output.id}.json"
        if task_path.exists():
            try:
                from alfs.cc.models import CCInductionTask

                task_adapter: TypeAdapter[CCInductionTask] = TypeAdapter(
                    CCInductionTask
                )
                task = task_adapter.validate_json(task_path.read_bytes())
                occurrence_refs: list[Occurrence] = task.occurrence_refs
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
                task_path.unlink()
            except Exception as exc:
                print(
                    f"  warning: could not apply context labels "
                    f"for {output.form!r}: {exc}"
                )
        else:
            print(
                f"  warning: task file not found for {output.form!r} "
                f"(id={output.id}), skipping context labels"
            )

    # Enqueue new senses
    entry = sense_store.read(output.form)
    existing_defs = (
        {s.definition.strip().lower() for s in entry.senses} if entry else set()
    )

    new_senses: list[Sense] = []
    for s in output.new_senses:
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


def run(
    cc_tasks_dir: str | Path,
    senses_db: str | Path,
    queue_dir: str | Path,
    labeled_db: str | Path | None = None,
    blocklist_file: str | Path | None = None,
) -> None:
    done_dir = Path(cc_tasks_dir) / "done"
    pending_induction_dir = Path(cc_tasks_dir) / "pending" / "induction"
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
                pending_induction_dir,
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
