"""Aggregate raw occurrences into per-prefix parquet files.

Usage:
    python -m alfs.seg.aggregate_occurrences \
        --occurrences raw_occurrences.parquet --output-dir by_prefix/ [--merge]
"""

import argparse
from pathlib import Path

import polars as pl


def prefix(form: str) -> str:
    if form and form[0].lower() in "abcdefghijklmnopqrstuvwxyz":
        return form[0].lower()
    return "other"


def aggregate(df: pl.DataFrame, output_dir: Path, merge: bool = False) -> None:
    """Write df to by_prefix layout under output_dir.

    merge=True: read existing parquet, concat, re-sort, write.
    merge=False: overwrite (original behaviour).

    Forms are lowercased before writing so that case variants (e.g. "Dogs",
    "DOGS") all map to the same canonical occurrence pool ("dogs"). Byte offsets
    remain valid — context extraction uses the offset directly against original
    text, not a string search for the form.
    """
    df = df.with_columns(pl.col("form").str.to_lowercase())
    df = df.with_columns(
        pl.col("form")
        .map_elements(
            prefix,
            return_dtype=pl.String,
        )
        .alias("prefix")
    )

    prefixes = df["prefix"].unique().sort().to_list()
    print(f"Writing {len(prefixes)} prefix groups to {output_dir}...")

    for pfx in prefixes:
        group = (
            df.filter(pl.col("prefix") == pfx)
            .drop("prefix")
            .sort(["form", "doc_id", "byte_offset"])
        )
        prefix_dir = output_dir / pfx
        prefix_dir.mkdir(parents=True, exist_ok=True)
        out_path = prefix_dir / "occurrences.parquet"

        if merge and out_path.exists():
            existing = pl.read_parquet(out_path)
            group = pl.concat([existing, group]).sort(["form", "doc_id", "byte_offset"])

        group.write_parquet(out_path)
        print(f"  {pfx}: {len(group)} rows → {out_path}")

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate occurrences by prefix")
    parser.add_argument(
        "--occurrences",
        nargs="+",
        required=True,
        help="Path(s) to occurrences parquet file(s)",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for by_prefix layout"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge into existing parquets instead of overwriting",
    )
    args = parser.parse_args()

    print(f"Loading occurrences from {args.occurrences}...")
    df = pl.read_parquet(args.occurrences)
    print(f"Loaded {len(df)} occurrences")

    aggregate(df, Path(args.output_dir), merge=args.merge)


if __name__ == "__main__":
    main()
