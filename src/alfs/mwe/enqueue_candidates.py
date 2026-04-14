"""Populate the MWE candidate queue from PMI results.

Reads precomputed PMI results, filters out known/blocklisted forms,
and adds candidates to the MWE queue for lexicographer review.

Usage:
    python -m alfs.mwe.enqueue_candidates \
        --pmi-results pmi_results.parquet \
        --senses-db ../alfs_data/senses.db \
        --blocklist-file ../alfs_data/blocklist.yaml \
        --mwe-queue-file ../alfs_data/mwe_queue.yaml \
        [--top-n 200] [--seg-data-dir by_prefix/ --n-occurrence-refs 3]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import polars as pl

from alfs.data_models.blocklist import Blocklist
from alfs.data_models.mwe_queue import MWEQueue, MWEQueueEntry
from alfs.data_models.occurrence import Occurrence
from alfs.data_models.sense_store import SenseStore
from alfs.mwe.find_occurrences import find_mwe_occurrences, load_all_seg_data


def run(
    pmi_results: Path,
    senses_db: Path,
    blocklist_file: Path,
    mwe_queue_file: Path,
    *,
    top_n: int = 200,
    seg_data_dir: Path | None = None,
    n_occurrence_refs: int = 3,
    seed: int | None = None,
) -> int:
    """Enqueue top-N MWE candidates by PMI. Returns count added."""
    pmi_df = pl.read_parquet(str(pmi_results))
    if len(pmi_df) == 0:
        print("No PMI candidates.")
        return 0

    # Build exclusion sets
    known_forms_lower = {f.lower() for f in SenseStore(senses_db).all_forms()}
    blocklist_lower = {f.lower() for f in Blocklist(blocklist_file).load()}
    queued_lower = {e.form.lower() for e in MWEQueue(mwe_queue_file).load()}
    excluded = known_forms_lower | blocklist_lower | queued_lower

    # Filter candidates
    candidates = pmi_df.filter(~pl.col("form").str.to_lowercase().is_in(excluded)).head(
        top_n
    )

    if len(candidates) == 0:
        print("No new MWE candidates to enqueue.")
        return 0

    # Optionally sample occurrence refs
    all_tokens = None
    if seg_data_dir and n_occurrence_refs > 0:
        all_tokens = load_all_seg_data(seg_data_dir)

    rng = random.Random(seed)
    entries: list[MWEQueueEntry] = []
    for row in candidates.iter_rows(named=True):
        occs: list[Occurrence] = []
        if all_tokens is not None:
            try:
                all_occs = find_mwe_occurrences(all_tokens, row["components"])
                sampled = rng.sample(all_occs, min(n_occurrence_refs, len(all_occs)))
                occs = sampled
            except Exception:
                pass
        entries.append(
            MWEQueueEntry(
                form=row["form"],
                components=row["components"],
                pmi=row["pmi"],
                corpus_count=row["count"],
                occurrences=occs,
            )
        )

    added = MWEQueue(mwe_queue_file).add_candidates(entries)
    print(f"Enqueued {added} MWE candidates (top by PMI).")
    for entry in entries[:10]:
        print(
            f"  {entry.form!r:30s} count={entry.corpus_count:6d}  pmi={entry.pmi:.2f}"
        )
    if len(entries) > 10:
        print(f"  ... and {len(entries) - 10} more")
    return added


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate MWE candidate queue from PMI results"
    )
    parser.add_argument("--pmi-results", required=True, help="Path to PMI parquet")
    parser.add_argument("--senses-db", required=True)
    parser.add_argument("--blocklist-file", required=True)
    parser.add_argument("--mwe-queue-file", required=True)
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument(
        "--seg-data-dir",
        default=None,
        help="Path to by_prefix/ dir (for sampling occurrence refs)",
    )
    parser.add_argument("--n-occurrence-refs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run(
        Path(args.pmi_results),
        Path(args.senses_db),
        Path(args.blocklist_file),
        Path(args.mwe_queue_file),
        top_n=args.top_n,
        seg_data_dir=Path(args.seg_data_dir) if args.seg_data_dir else None,
        n_occurrence_refs=args.n_occurrence_refs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
