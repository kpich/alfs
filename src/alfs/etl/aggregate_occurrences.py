"""Aggregate raw occurrences into per-prefix parquet files.

Usage:
    python -m alfs.etl.aggregate_occurrences \
        --occurrences raw_occurrences.parquet --output-dir by_prefix/
"""

import argparse
import os

import polars as pl


def _prefix(form: str) -> str:
    if form and form[0].lower() in "abcdefghijklmnopqrstuvwxyz":
        return form[0].lower()
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate occurrences by prefix")
    parser.add_argument(
        "--occurrences", required=True, help="Path to raw_occurrences.parquet"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for by_prefix layout"
    )
    args = parser.parse_args()

    print(f"Loading occurrences from {args.occurrences}...")
    df = pl.read_parquet(args.occurrences)
    print(f"Loaded {len(df)} occurrences")

    df = df.with_columns(
        pl.col("form")
        .map_elements(
            _prefix,
            return_dtype=pl.String,
        )
        .alias("prefix")
    )

    prefixes = df["prefix"].unique().sort().to_list()
    print(f"Writing {len(prefixes)} prefix groups to {args.output_dir}...")

    for prefix in prefixes:
        group = (
            df.filter(pl.col("prefix") == prefix)
            .drop("prefix")
            .sort(["form", "doc_id", "byte_offset"])
        )
        prefix_dir = os.path.join(args.output_dir, prefix)
        os.makedirs(prefix_dir, exist_ok=True)
        out_path = os.path.join(prefix_dir, "occurrences.parquet")
        group.write_parquet(out_path)
        print(f"  {prefix}: {len(group)} rows → {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
