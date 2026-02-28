"""Validate that labeled.parquet byte_offsets still resolve to the expected form."""

import argparse
import sys

import polars as pl


def validate(labeled: pl.DataFrame, docs: pl.DataFrame) -> pl.DataFrame:
    """Return rows from labeled where the byte_offset no longer resolves to form.

    Returns a DataFrame with the same schema as labeled, containing only stale rows.
    Empty result = all labels valid.

    Orphaned labels (doc_id not in docs) are NOT flagged — that's expected behavior
    when docs are removed.
    """
    docs_map = dict(zip(docs["doc_id"].to_list(), docs["text"].to_list(), strict=False))
    stale_indices = []
    for i, row in enumerate(labeled.iter_rows(named=True)):
        text = docs_map.get(row["doc_id"])
        if text is None:
            continue  # orphaned label — not flagged
        char_offset = len(text.encode()[: row["byte_offset"]].decode())
        form = row["form"]
        if text[char_offset : char_offset + len(form)] != form:
            stale_indices.append(i)
    return labeled[stale_indices]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate labeled.parquet against docs.parquet"
    )
    parser.add_argument("--labeled", required=True, help="Path to labeled.parquet")
    parser.add_argument("--docs", required=True, help="Path to docs.parquet")
    args = parser.parse_args()

    labeled = pl.read_parquet(args.labeled)
    docs = pl.read_parquet(args.docs)

    stale = validate(labeled, docs)
    total = len(labeled)
    n_stale = len(stale)

    print(f"{total} labels checked, {n_stale} stale")
    if n_stale > 0:
        print(stale)
        sys.exit(1)


if __name__ == "__main__":
    main()
