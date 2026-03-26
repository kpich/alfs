"""Orchestrate labeling of new forms: generate targets then label each form.

Usage:
    python -m alfs.update.labeling.run_label_new \\
        --senses-db senses.db --labeled-db labeled.db \\
        --docs docs.parquet --seg-data-dir by_prefix/ \\
        [--nwords 10] [--model gemma2:9b] [--context-chars 150] [--max-occurrences 10]
"""

import argparse

from alfs.update.labeling.run_relabel import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Label new (unlabeled) forms")
    parser.add_argument("--senses-db", required=True)
    parser.add_argument("--labeled-db", required=True)
    parser.add_argument("--docs", required=True)
    parser.add_argument("--seg-data-dir", required=True)
    parser.add_argument("--nwords", type=int, default=10)
    parser.add_argument("--model", default="gemma2:9b")
    parser.add_argument("--context-chars", type=int, default=150)
    parser.add_argument("--max-occurrences", type=int, default=10)
    parser.add_argument(
        "--log-dir", default=None, help="Directory for instance-tagging change log"
    )
    args = parser.parse_args()
    run(
        senses_db=args.senses_db,
        labeled_db=args.labeled_db,
        new_only=True,  # filter targets to forms not yet in labeled.db
        docs=args.docs,
        seg_data_dir=args.seg_data_dir,
        nwords=args.nwords,
        model=args.model,
        context_chars=args.context_chars,
        max_occurrences=args.max_occurrences,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
