"""Orchestrate relabeling: generate targets then label each form.

Usage:
    python -m alfs.update.labeling.run_relabel \\
        --senses-db senses.db --labeled-db labeled.db \\
        --docs docs.parquet --seg-data-dir by_prefix/ \\
        [--nwords 5] [--model gemma2:9b] [--context-chars 150] [--max-occurrences 20]
"""

import argparse
from pathlib import Path
import tempfile

from alfs.update.labeling import label_occurrences
from alfs.update.labeling.generate_relabel_targets import generate_targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel forms in labeled.db")
    parser.add_argument("--senses-db", required=True)
    parser.add_argument("--labeled-db", required=True)
    parser.add_argument("--docs", required=True)
    parser.add_argument("--seg-data-dir", required=True)
    parser.add_argument("--nwords", type=int, default=5)
    parser.add_argument("--model", default="gemma2:9b")
    parser.add_argument("--context-chars", type=int, default=150)
    parser.add_argument("--max-occurrences", type=int, default=20)
    parser.add_argument(
        "--log-dir", default=None, help="Directory for instance-tagging change log"
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        target_files = generate_targets(
            senses_db=Path(args.senses_db),
            output_dir=Path(tmp),
            labeled_db=Path(args.labeled_db),
            nwords=args.nwords,
        )
        for target_file in target_files:
            label_occurrences.run(
                target_file=target_file,
                seg_data_dir=args.seg_data_dir,
                docs=args.docs,
                senses_db=args.senses_db,
                labeled_db=args.labeled_db,
                model=args.model,
                context_chars=args.context_chars,
                max_occurrences=args.max_occurrences,
                log_dir=args.log_dir,
            )


if __name__ == "__main__":
    main()
