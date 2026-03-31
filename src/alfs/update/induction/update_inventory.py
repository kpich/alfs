"""Augment senses.db with new senses for a single form (never overwrites, only appends).

Usage:
    python -m alfs.update.induction.update_inventory \\
        --senses-file {form}_senses.json --senses-db senses.db \\
        --queue-dir ../clerk_queue
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path
import uuid

from alfs.clerk.queue import enqueue
from alfs.clerk.request import AddSensesRequest
from alfs.data_models.alf import Alf


def merge_entry(existing: Alf, new: Alf) -> Alf:
    """Return existing Alf with any genuinely new senses from new appended."""
    existing_defs = {s.definition.strip().lower() for s in existing.senses}
    new_senses = [
        s for s in new.senses if s.definition.strip().lower() not in existing_defs
    ]
    if not new_senses:
        return existing
    return Alf(
        form=existing.form,
        senses=list(existing.senses) + new_senses,
    )


def run(senses_file: str | Path, senses_db: str | Path, queue_dir: str | Path) -> None:
    alf = Alf.model_validate_json(Path(senses_file).read_text())
    if not alf.senses:
        print(f"No new senses for '{alf.form}' (empty senses file)")
        return

    request = AddSensesRequest(
        id=str(uuid.uuid4()),
        created_at=datetime.now(UTC),
        form=alf.form,
        new_senses=list(alf.senses),
    )
    enqueue(request, Path(queue_dir))
    print(f"  Queued add_senses for '{alf.form}' ({len(alf.senses)} senses)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment senses.db with new senses for a single form"
    )
    parser.add_argument(
        "--senses-file",
        required=True,
        help="Path to {form}_senses.json from INDUCE_SENSES",
    )
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument(
        "--queue-dir", required=True, help="Path to clerk queue directory"
    )
    args = parser.parse_args()

    run(args.senses_file, args.senses_db, args.queue_dir)


if __name__ == "__main__":
    main()
