"""Python pipeline runner for the full update cycle.

Replaces nextflow/update.nf. Runs sense induction + inventory update + labeling,
with a clerk drain as the sync point between inventory and labeling.

Usage:
    python -m alfs.update.run_update \\
        --seg-data-dir by_prefix/ --docs docs.parquet \\
        --senses-db senses.db --labeled-db labeled.db \\
        --queue-dir ../clerk_queue [--model llama3.1:8b] [--top-n 10] \\
        [--context-chars 150] [--max-samples 20] [--max-occurrences 100] \\
        [--workers 4]
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import tempfile
import traceback

from alfs.clerk.queue import drain
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore
from alfs.update.induction import induce_senses, update_inventory
from alfs.update.labeling import label_occurrences, select_targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full update pipeline")
    parser.add_argument("--seg-data-dir", required=True, help="Path to by_prefix/")
    parser.add_argument("--docs", required=True, help="Path to docs.parquet")
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument("--labeled-db", required=True, help="Path to labeled.db")
    parser.add_argument("--queue-dir", required=True, help="Path to clerk queue dir")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--context-chars", type=int, default=150)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-occurrences", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    senses_db = Path(args.senses_db)
    labeled_db = Path(args.labeled_db)
    queue_dir = Path(args.queue_dir)

    with tempfile.TemporaryDirectory(prefix="alfs_update_") as tmpdir:
        targets_dir = Path(tmpdir) / "targets"
        senses_dir = Path(tmpdir) / "senses"
        targets_dir.mkdir()
        senses_dir.mkdir()

        # Phase 1: select targets
        print("=== Phase 1: Select targets ===")
        target_files = select_targets.run(
            seg_data_dir=args.seg_data_dir,
            top_n=args.top_n,
            output_dir=targets_dir,
            senses_db=senses_db if senses_db.exists() else None,
            labeled_db=labeled_db if labeled_db.exists() else None,
        )
        print(f"Selected {len(target_files)} targets.")

        # Phase 2: parallel induction (critic inline in run_induce)
        print("=== Phase 2: Induce senses ===")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    induce_senses.run,
                    tf,
                    args.seg_data_dir,
                    args.docs,
                    senses_dir / tf.name,
                    args.model,
                    args.context_chars,
                    args.max_samples,
                    senses_db if senses_db.exists() else None,
                    labeled_db if labeled_db.exists() else None,
                ): tf
                for tf in target_files
            }
            for future in as_completed(futures):
                tf = futures[future]
                try:
                    future.result()
                except Exception:
                    print(f"  ERROR inducing senses for {tf.name}:")
                    traceback.print_exc()

        # Phase 3: update inventory (enqueue)
        print("=== Phase 3: Update inventory ===")
        for sf in senses_dir.glob("*.json"):
            try:
                update_inventory.run(sf, senses_db, queue_dir)
            except Exception:
                print(f"  ERROR updating inventory for {sf.name}:")
                traceback.print_exc()

        # Phase 4: drain clerk (sync point — ensures senses.db is up to date)
        print("=== Phase 4: Drain clerk queue ===")
        drain(
            queue_dir,
            SenseStore(senses_db),
            OccurrenceStore(labeled_db) if labeled_db.exists() else None,
            workers=args.workers,
        )
        print("Clerk queue drained.")

        # Phase 5: parallel labeling
        print("=== Phase 5: Label occurrences ===")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures_label = {
                pool.submit(
                    label_occurrences.run,
                    tf,
                    args.seg_data_dir,
                    args.docs,
                    senses_db,
                    labeled_db,
                    args.model,
                    args.context_chars,
                    args.max_occurrences,
                ): tf
                for tf in target_files
            }
            for future in as_completed(futures_label):
                tf = futures_label[future]
                try:
                    future.result()
                except Exception:
                    print(f"  ERROR labeling {tf.name}:")
                    traceback.print_exc()

    print("=== Update pipeline complete ===")


if __name__ == "__main__":
    main()
