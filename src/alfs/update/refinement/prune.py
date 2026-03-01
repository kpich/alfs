"""Identify low-quality senses and queue them for human removal approval.

Usage:
    python -m alfs.update.refinement.prune \\
        --senses-db ../alfs_data/senses.db \\
        --labeled-db ../alfs_data/labeled.db \\
        --changes-db ../alfs_data/changes.db \\
        [--n 5]
"""

import argparse
from datetime import datetime
from pathlib import Path
import uuid

import polars as pl

from alfs.data_models.alf import sense_key
from alfs.data_models.change_store import Change, ChangeStatus, ChangeStore, ChangeType
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Queue prune changes for low-quality senses"
    )
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument("--labeled-db", required=True, help="Path to labeled.db")
    parser.add_argument("--changes-db", required=True, help="Path to changes.db")
    parser.add_argument("--n", type=int, default=5, help="Max senses to prune")
    args = parser.parse_args()

    sense_store = SenseStore(Path(args.senses_db))
    occ_store = OccurrenceStore(Path(args.labeled_db))
    change_store = ChangeStore(Path(args.changes_db))

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
        # Validate sense_key maps to an actual sense index
        try:
            top_idx = int(sk) - 1  # sense_key is 1-based numeric string like "2"
        except ValueError:
            continue
        if top_idx < 0 or top_idx >= len(alf.senses):
            continue
        eligible_rows.append({**row, "top_idx": top_idx})

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
        bad_keys = {r["top_idx"] for r in bad_rows}
        remaining = [s for i, s in enumerate(alf.senses) if i not in bad_keys]

        # Safety check: must leave at least 1 sense
        if not remaining:
            print(f"  {form!r}: skipping — would remove all senses")
            continue

        removed_info = [
            {"index": r["top_idx"], "pct_lt3": r["pct_lt3"], "total": r["total"]}
            for r in bad_rows
        ]

        change = Change(
            id=str(uuid.uuid4()),
            type=ChangeType.prune,
            form=form,
            data={
                "before": [s.model_dump() for s in alf.senses],
                "after": [s.model_dump() for s in remaining],
                "removed": removed_info,
            },
            status=ChangeStatus.pending,
            created_at=datetime.utcnow(),
        )
        change_store.add(change)

        for r in bad_rows:
            pct = round(r["pct_lt3"] * 100)
            sk = sense_key(r["top_idx"])
            print(f"  {form!r}: removing sense {sk} ({pct}% <3, n={r['total']})")
        queued += 1

    print(f"Queued {queued} prune change{'s' if queued != 1 else ''}.")


if __name__ == "__main__":
    main()
