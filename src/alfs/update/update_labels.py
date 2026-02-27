"""Merge new labeled parquet files into consolidated labeled.parquet.

Usage:
    python -m alfs.update.update_labels \\
        --labeled-data labeled.parquet --new-dir new_labeled/ \\
        --output labeled.parquet
"""

import argparse
from pathlib import Path

import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge new labeled files into consolidated labeled.parquet"
    )
    parser.add_argument(
        "--labeled-data", required=True, help="Path to existing labeled.parquet"
    )
    parser.add_argument(
        "--new-dir", required=True, help="Directory with new *_labeled.parquet files"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    labeled_path = Path(args.labeled_data)
    if labeled_path.exists():
        existing = pl.read_parquet(str(labeled_path))
    else:
        existing = pl.DataFrame(
            {
                "form": [],
                "doc_id": [],
                "byte_offset": [],
                "sense_key": [],
                "rating": [],
            },
            schema={
                "form": pl.String,
                "doc_id": pl.String,
                "byte_offset": pl.Int64,
                "sense_key": pl.String,
                "rating": pl.Int64,
            },
        )

    new_files = list(Path(args.new_dir).glob("**/*_labeled.parquet"))
    if not new_files:
        print("No new labeled files found; writing existing data unchanged.")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        existing.write_parquet(args.output)
        return

    new_dfs = [pl.read_parquet(str(f)) for f in new_files]
    # Put new data after existing so newest rows win on dedup (keep="last")
    combined = pl.concat([existing] + new_dfs)
    deduped = combined.unique(subset=["form", "doc_id", "byte_offset"], keep="last")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    deduped.write_parquet(args.output)
    print(f"Merged {len(new_dfs)} new files; {len(deduped)} total labeled rows")
    print(f"→ {args.output}")


if __name__ == "__main__":
    main()
