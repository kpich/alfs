"""Identify low-quality senses and queue them for removal via clerk.

Usage:
    python -m alfs.update.refinement.prune \\
        --senses-db ../alfs_data/senses.db \\
        --labeled-db ../alfs_data/labeled.db \\
        --queue-dir ../clerk_queue \\
        [--n 5]
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path
import uuid

import polars as pl

from alfs.clerk.queue import enqueue
from alfs.clerk.request import PruneRequest
from alfs.data_models.alf import sense_key
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Queue prune changes for low-quality senses via clerk"
    )
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument("--labeled-db", required=True, help="Path to labeled.db")
    parser.add_argument(
        "--queue-dir", required=True, help="Path to clerk queue directory"
    )
    parser.add_argument("--n", type=int, default=5, help="Max senses to prune")
    args = parser.parse_args()

    sense_store = SenseStore(Path(args.senses_db))
    occ_store = OccurrenceStore(Path(args.labeled_db))
    queue_dir = Path(args.queue_dir)

    labeled_df = occ_store.to_polars()
    all_entries = sense_store.all_entries()

    # Compute per-sense quality stats
    stats = (
        labeled_df.group_by(["form", "sense_key"])
        .agg(
            [
                pl.len().alias("total"),
                (pl.col("rating") < 3).sum().alias("n_lt3"),
            ]
        )
        .with_columns((pl.col("n_lt3") / pl.col("total")).alias("pct_lt3"))
        .filter(pl.col("pct_lt3") > 0.20)
        .sort("pct_lt3", descending=True)
        .head(args.n)
    )

    if stats.is_empty():
        print("No senses qualify for pruning.")
        return

    # Filter to forms present in senses.db, non-redirect, with >1 sense
    eligible_rows = []
    for row in stats.iter_rows(named=True):
        form = row["form"]
        sk = row["sense_key"]
        alf = all_entries.get(form)
        if alf is None:
            continue
        if alf.redirect:
            continue
        if len(alf.senses) <= 1:
            continue
        # sense_key is a UUID (top-level) or UUID+letter (subsense); extract base UUID
        sense_id = sk[:-1] if len(sk) == 37 and sk[-1].isalpha() else sk
        sense = next((s for s in alf.senses if s.id == sense_id), None)
        if sense is None:
            continue
        eligible_rows.append({**row, "sense_id": sense_id})

    if not eligible_rows:
        print("No eligible senses after filtering.")
        return

    # Group by form
    by_form: dict[str, list[dict]] = {}  # type: ignore[type-arg]
    for row in eligible_rows:
        by_form.setdefault(row["form"], []).append(row)

    queued = 0
    for form, bad_rows in by_form.items():
        alf = all_entries[form]
        bad_ids = {r["sense_id"] for r in bad_rows}
        remaining = [s for s in alf.senses if s.id not in bad_ids]

        # Safety check: must leave at least 1 sense
        if not remaining:
            print(f"  {form!r}: skipping — would remove all senses")
            continue

        removed_ids = list(bad_ids)

        request = PruneRequest(
            id=str(uuid.uuid4()),
            created_at=datetime.now(UTC),
            form=form,
            before=list(alf.senses),
            after=remaining,
            removed_ids=removed_ids,
        )
        enqueue(request, queue_dir)

        for r in bad_rows:
            pct = round(r["pct_lt3"] * 100)
            idx = next(i for i, s in enumerate(alf.senses) if s.id == r["sense_id"])
            display_sk = sense_key(idx)
            print(
                f"  {form!r}: removing sense {display_sk} ({pct}% <3, n={r['total']})"
            )
        queued += 1

    print(f"Queued {queued} prune change{'s' if queued != 1 else ''}.")


if __name__ == "__main__":
    main()
