"""Split docs into fixed-size shards for parallel segmentation.

Filters out already-segmented doc_ids from the by_prefix layout, then
writes new docs as shard_NNNN.parquet files (one per shard_size docs).

Usage:
    python -m alfs.seg.split_docs \
        --docs docs.parquet \
        --seg-data-dir ../seg_data/by_prefix \
        --shard-size 1000
"""

import argparse
from pathlib import Path

import polars as pl


def get_segmented_doc_ids(seg_data_dir: Path) -> set[str]:
    parquets = list(seg_data_dir.glob("*/occurrences.parquet"))
    if not parquets:
        return set()
    doc_ids: set[str] = set()
    for p in parquets:
        df = pl.read_parquet(p, columns=["doc_id"])
        doc_ids.update(df["doc_id"].to_list())
    return doc_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split docs into shards for parallel segmentation"
    )
    parser.add_argument("--docs", required=True, help="Path to docs.parquet")
    parser.add_argument(
        "--seg-data-dir",
        required=True,
        help="Existing by_prefix directory (used to find already-segmented doc_ids)",
    )
    parser.add_argument(
        "--shard-size", type=int, default=1000, help="Number of docs per shard"
    )
    args = parser.parse_args()

    seg_data_dir = Path(args.seg_data_dir)
    segmented_ids = get_segmented_doc_ids(seg_data_dir)
    print(f"Already segmented: {len(segmented_ids)} docs")

    print(f"Loading docs from {args.docs}...")
    all_docs = pl.read_parquet(args.docs)
    new_docs = all_docs.filter(~pl.col("doc_id").is_in(list(segmented_ids)))
    print(f"New docs to segment: {len(new_docs)}")

    if len(new_docs) == 0:
        print("Nothing to do.")
        return

    n_shards = (len(new_docs) + args.shard_size - 1) // args.shard_size
    print(f"Writing {n_shards} shards of up to {args.shard_size} docs...")

    for i in range(n_shards):
        shard = new_docs[i * args.shard_size : (i + 1) * args.shard_size]
        out_path = f"shard_{i:04d}.parquet"
        shard.write_parquet(out_path)
        print(f"  {out_path}: {len(shard)} docs")

    print("Done.")


if __name__ == "__main__":
    main()
