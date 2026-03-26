"""Clear stale senses from redirect entries.

Redirect entries should never have definitions. This script finds any entry
where ``alf.redirect`` is set but ``alf.senses`` is non-empty and enqueues
a ClearRedirectSensesRequest for each.

Usage:
    python -m alfs.update.refinement.cleanup \
        --senses-db ../alfs_data/senses.db \
        --queue-dir ../clerk_queue
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path
import uuid

from alfs.clerk.queue import enqueue
from alfs.clerk.request import ClearRedirectSensesRequest
from alfs.data_models.sense_store import SenseStore


def run(senses_db: str | Path, queue_dir: str | Path) -> int:
    store = SenseStore(Path(senses_db))
    queue_dir = Path(queue_dir)
    queued = 0
    for form, alf in store.all_entries().items():
        if alf.redirect and alf.senses:
            request = ClearRedirectSensesRequest(
                id=str(uuid.uuid4()),
                created_at=datetime.now(UTC),
                form=form,
            )
            enqueue(request, queue_dir)
            print(f"  queued clear for redirect: {form!r} → {alf.redirect!r}")
            queued += 1
    print(f"Queued {queued} entries.")
    return queued


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clear stale senses from redirect entries"
    )
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument("--queue-dir", required=True, help="Path to clerk queue dir")
    args = parser.parse_args()
    run(args.senses_db, args.queue_dir)


if __name__ == "__main__":
    main()
