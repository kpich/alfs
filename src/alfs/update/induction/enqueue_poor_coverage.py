"""Enqueue forms with poorly-labeled occurrences into the induction queue.

Finds forms in labeled.db that have occurrences with rating 0 or 1 (and are
not already marked as _skip), then adds them (sorted by bad-occurrence count)
to the induction queue with those specific occurrence refs attached.
Running this multiple times is idempotent: forms already in the queue are skipped.

Usage:
    python -m alfs.update.induction.enqueue_poor_coverage \\
        --labeled-db ../alfs_data/labeled.db \\
        --queue-file ../alfs_data/induction_queue.yaml \\
        --blocklist-file ../alfs_data/blocklist.yaml \\
        [--top-n 500] [--min-bad 1] [--max-per-form 10]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from alfs.data_models.blocklist import Blocklist
from alfs.data_models.induction_queue import InductionQueue
from alfs.data_models.occurrence import Occurrence
from alfs.data_models.occurrence_store import OccurrenceStore

_SKIP_KEY = "_skip"


def run(
    labeled_db: str | Path,
    queue_file: str | Path,
    blocklist_file: str | Path,
    top_n: int = 500,
    min_bad: int = 1,
    max_per_form: int = 10,
) -> int:
    """Enqueue forms with poor-quality labeled occurrences. Returns count added."""
    occ_store = OccurrenceStore(Path(labeled_db))
    df = occ_store.to_polars()

    if df.is_empty():
        print("No labeled occurrences found.")
        return 0

    # Filter to poor-quality, non-skip occurrences
    bad_df = df.filter(
        pl.col("rating").is_in([0, 1]) & (pl.col("sense_key") != _SKIP_KEY)
    )

    if bad_df.is_empty():
        print("No poorly-labeled occurrences found.")
        return 0

    # Build exclusion sets
    blocklist_forms = set(Blocklist(Path(blocklist_file)).load().keys())
    queued_forms = {e.form for e in InductionQueue(Path(queue_file)).load()}
    excluded = blocklist_forms | queued_forms

    # Group by form, count bad occurrences, sort descending
    grouped = (
        bad_df.filter(~pl.col("form").is_in(list(excluded)))
        .group_by("form")
        .agg(
            [
                pl.len().alias("n_bad"),
                pl.struct("doc_id", "byte_offset").alias("occurrences"),
            ]
        )
        .filter(pl.col("n_bad") >= min_bad)
        .sort("n_bad", descending=True)
        .head(top_n)
    )

    if grouped.is_empty():
        print("No qualifying forms to enqueue.")
        return 0

    forms = grouped["form"].to_list()
    occs_by_form: dict[str, list[Occurrence]] = {}
    for row in grouped.iter_rows(named=True):
        form = row["form"]
        occ_list = row["occurrences"][:max_per_form]
        occs_by_form[form] = [
            Occurrence(doc_id=o["doc_id"], byte_offset=o["byte_offset"])
            for o in occ_list
        ]

    added = InductionQueue(Path(queue_file)).add_forms(forms, occs_by_form)
    print(
        f"Enqueued {added} forms with poor coverage "
        f"(top by bad-occurrence count, min_bad={min_bad})."
    )
    for row in grouped.head(10).iter_rows(named=True):
        print(f"  {row['form']!r}  ({row['n_bad']} bad occurrences)")
    if len(forms) > 10:
        print(f"  ... and {len(forms) - 10} more")
    return added


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enqueue forms with poorly-labeled occurrences"
    )
    parser.add_argument("--labeled-db", required=True)
    parser.add_argument("--queue-file", required=True)
    parser.add_argument("--blocklist-file", required=True)
    parser.add_argument("--top-n", type=int, default=500)
    parser.add_argument("--min-bad", type=int, default=1)
    parser.add_argument("--max-per-form", type=int, default=10)
    args = parser.parse_args()

    run(
        args.labeled_db,
        args.queue_file,
        args.blocklist_file,
        args.top_n,
        args.min_bad,
        args.max_per_form,
    )


if __name__ == "__main__":
    main()
