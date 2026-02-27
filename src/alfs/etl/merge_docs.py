"""Merge multiple Parquet doc files into one.

Usage:
    python -m alfs.etl.merge_docs \
        --inputs wikibooks.parquet wikisource.parquet \
        --output docs.parquet
"""

import argparse

import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Parquet doc files")
    parser.add_argument(
        "--inputs", nargs="+", required=True, help="Input Parquet files"
    )
    parser.add_argument("--output", required=True, help="Output Parquet file path")
    args = parser.parse_args()

    df = pl.concat([pl.read_parquet(f) for f in args.inputs])
    df.write_parquet(args.output)
    print(f"Merged {len(args.inputs)} files → {args.output} ({df.shape})")


if __name__ == "__main__":
    main()
