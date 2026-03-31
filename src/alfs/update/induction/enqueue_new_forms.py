"""Enqueue top-N unseen corpus forms into the induction queue.

Finds word forms in the corpus that have no entry in senses.db and are not
in the blocklist, then adds them (sorted by frequency) to the induction queue.
Running this multiple times is idempotent: forms already in the queue are skipped.

Usage:
    python -m alfs.update.induction.enqueue_new_forms \\
        --seg-data-dir by_prefix/ \\
        --senses-db ../alfs_data/senses.db \\
        --queue-file ../alfs_data/induction_queue.yaml \\
        --blocklist-file ../alfs_data/blocklist.yaml \\
        [--top-n 500] [--min-count 5] [--n-occurrence-refs 3] [--seed 42]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import re

import polars as pl

from alfs.data_models.blocklist import Blocklist
from alfs.data_models.induction_queue import InductionQueue
from alfs.data_models.occurrence import Occurrence
from alfs.data_models.sense_store import SenseStore
from alfs.seg.aggregate_occurrences import prefix as form_prefix

_WORD_RE = re.compile(r"[a-zA-Z]")


def run(
    seg_data_dir: str | Path,
    senses_db: str | Path,
    queue_file: str | Path,
    blocklist_file: str | Path,
    top_n: int = 500,
    min_count: int = 5,
    n_occurrence_refs: int = 3,
    seed: int | None = None,
) -> int:
    """Enqueue top-N unseen forms by corpus frequency. Returns count added."""
    seg_dir = Path(seg_data_dir)
    parquet_files = list(seg_dir.glob("*/occurrences.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No occurrences.parquet files found in {seg_dir}. Run `make seg` first."
        )

    # Count total corpus occurrences per form
    total_counts = (
        pl.concat([pl.scan_parquet(str(f)).select("form") for f in parquet_files])
        .group_by("form")
        .agg(pl.len().alias("total"))
        .collect()
    )

    # Filter to word-like forms with enough occurrences
    total_counts = total_counts.filter(pl.col("total") >= min_count)

    # Build exclusion sets
    known_forms = set(SenseStore(Path(senses_db)).all_forms())
    # Also exclude lowercase of any known form so that e.g. a "POTS" entry
    # prevents "pots" from being enqueued as a new form (parquets are lowercased).
    known_forms_lower = {f.lower() for f in known_forms}
    blocklist_forms = set(Blocklist(Path(blocklist_file)).load().keys())
    queued_forms = {e.form for e in InductionQueue(Path(queue_file)).load()}
    excluded = known_forms | known_forms_lower | blocklist_forms | queued_forms

    # Filter candidates: word-like forms not in any exclusion set
    candidates = (
        total_counts.filter(
            pl.col("form").map_elements(
                lambda f: bool(_WORD_RE.search(f)), return_dtype=pl.Boolean
            )
        )
        .filter(~pl.col("form").is_in(list(excluded)))
        .sort("total", descending=True)
        .head(top_n)
    )

    forms = candidates["form"].to_list()
    counts = dict(
        zip(candidates["form"].to_list(), candidates["total"].to_list(), strict=False)
    )

    if not forms:
        print("No new forms to enqueue.")
        return 0

    # Optionally sample occurrence refs for each form
    rng = random.Random(seed)
    occs_by_form: dict[str, list[Occurrence]] = {}
    if n_occurrence_refs > 0:
        for form in forms:
            occ_path = seg_dir / form_prefix(form) / "occurrences.parquet"
            if not occ_path.exists():
                continue
            try:
                df = pl.read_parquet(str(occ_path)).filter(
                    pl.col("form") == form.lower()
                )
                rows = df.select(["doc_id", "byte_offset"]).to_dicts()
                sampled = rng.sample(rows, min(n_occurrence_refs, len(rows)))
                occs_by_form[form] = [
                    Occurrence(doc_id=r["doc_id"], byte_offset=r["byte_offset"])
                    for r in sampled
                ]
            except Exception:
                pass

    added = InductionQueue(Path(queue_file)).add_forms(forms, occs_by_form)
    print(f"Enqueued {added} new forms (top by frequency, min_count={min_count}).")
    for form in forms[:10]:
        print(f"  {form!r}  ({counts.get(form, 0)} occurrences)")
    if len(forms) > 10:
        print(f"  ... and {len(forms) - 10} more")
    return added


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enqueue top-N unseen corpus forms into the induction queue"
    )
    parser.add_argument("--seg-data-dir", required=True)
    parser.add_argument("--senses-db", required=True)
    parser.add_argument("--queue-file", required=True)
    parser.add_argument("--blocklist-file", required=True)
    parser.add_argument("--top-n", type=int, default=500)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--n-occurrence-refs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run(
        args.seg_data_dir,
        args.senses_db,
        args.queue_file,
        args.blocklist_file,
        args.top_n,
        args.min_count,
        args.n_occurrence_refs,
        args.seed,
    )


if __name__ == "__main__":
    main()
