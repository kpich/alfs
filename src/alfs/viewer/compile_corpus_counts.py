"""Pre-compute corpus frequency counts for viewer compilation.

Usage:
    python -m alfs.viewer.compile_corpus_counts \\
        --senses-db senses.db --by-prefix-dir seg_data/by_prefix \\
        --output corpus_counts.json
"""

import argparse
import json
from pathlib import Path

import polars as pl

from alfs.data_models.sense_store import SenseStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute corpus counts for viewer")
    parser.add_argument("--senses-db", required=True)
    parser.add_argument("--by-prefix-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sense_store = SenseStore(Path(args.senses_db))
    alfs_forms = list(sense_store.all_entries().keys())

    corpus_df = (
        pl.scan_parquet(
            str(Path(args.by_prefix_dir) / "**" / "*.parquet"),
            schema={"form": pl.String},
        )
        .filter(pl.col("form").is_in(alfs_forms))
        .group_by("form")
        .agg(pl.len().alias("count"))
        .collect()
    )
    corpus_counts = dict(
        zip(corpus_df["form"].to_list(), corpus_df["count"].to_list(), strict=False)
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(corpus_counts))
    print(f"Wrote corpus counts for {len(corpus_counts)} forms → {args.output}")


if __name__ == "__main__":
    main()
