"""Pre-compute corpus frequency counts for viewer compilation.

Usage:
    python -m alfs.viewer.compile_corpus_counts \\
        --senses-db senses.db --by-prefix-dir seg_data/by_prefix \\
        --output corpus_counts.json \\
        [--top-forms-output top_corpus_forms.json --top-forms-n 500]
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
    parser.add_argument(
        "--top-forms-output",
        help="Output all-corpus form counts, case-normalized (no senses.db filter)",
    )
    args = parser.parse_args()

    sense_store = SenseStore(Path(args.senses_db))
    alfs_forms = list(sense_store.all_entries().keys())

    parquet_glob = str(Path(args.by_prefix_dir) / "**" / "*.parquet")

    # Single scan: count all forms in corpus, excluding all-punctuation tokens
    all_df = (
        pl.scan_parquet(
            parquet_glob,
            schema={"form": pl.String},
            extra_columns="ignore",
        )
        .filter(pl.col("form").str.contains(r"[a-zA-Z0-9]"))
        .group_by("form")
        .agg(pl.len().alias("count"))
        .collect()
    )

    total_tokens = int(all_df["count"].sum())

    # corpus_counts: filtered to senses.db forms + total
    corpus_df = all_df.filter(pl.col("form").is_in(alfs_forms))
    corpus_counts: dict[str, int] = dict(
        zip(corpus_df["form"].to_list(), corpus_df["count"].to_list(), strict=False)
    )
    corpus_counts["_total"] = total_tokens

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(corpus_counts))
    print(
        f"Wrote corpus counts for {len(corpus_counts) - 1} forms "
        f"({total_tokens:,} total tokens) → {args.output}"
    )

    # top_forms: all corpus forms, case-normalized to lowercase, for coverage chart
    if args.top_forms_output:
        top_df = (
            all_df.with_columns(pl.col("form").str.to_lowercase().alias("form_lower"))
            .group_by("form_lower")
            .agg(pl.col("count").sum())
            .rename({"form_lower": "form"})
            .sort("count", descending=True)
        )
        top_forms: dict[str, int] = dict(
            zip(top_df["form"].to_list(), top_df["count"].to_list(), strict=False)
        )
        Path(args.top_forms_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.top_forms_output).write_text(json.dumps(top_forms))
        print(
            f"Wrote {len(top_forms)} corpus forms (lowercase) → {args.top_forms_output}"
        )


if __name__ == "__main__":
    main()
