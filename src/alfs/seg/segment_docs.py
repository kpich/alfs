"""Tokenize docs and emit (form, doc_id, byte_offset) tuples.

Usage:
    python -m alfs.etl.segment_docs \
        --docs docs.parquet --output raw_occurrences.parquet
"""

import argparse

import polars as pl
import spacy


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment docs into occurrences")
    parser.add_argument("--docs", required=True, help="Path to docs.parquet")
    parser.add_argument(
        "--output", required=True, help="Output path for raw_occurrences.parquet"
    )
    args = parser.parse_args()

    print(f"Loading docs from {args.docs}...")
    df = pl.read_parquet(args.docs)
    print(f"Loaded {len(df)} docs")

    nlp = spacy.load("en_core_web_sm")

    rows: list[dict[str, object]] = []
    for row in df.iter_rows(named=True):
        doc_id: str = row["doc_id"]
        text: str = row["text"]
        spacy_doc = nlp(text)
        for token in spacy_doc:
            byte_offset = len(text[: token.idx].encode())
            rows.append(
                {"form": token.text, "doc_id": doc_id, "byte_offset": byte_offset}
            )

    print(f"Writing {len(rows)} occurrences to {args.output}...")
    out_df = pl.DataFrame(
        rows, schema={"form": pl.String, "doc_id": pl.String, "byte_offset": pl.Int64}
    )
    out_df.write_parquet(args.output)
    print(f"Done. Shape: {out_df.shape}")


if __name__ == "__main__":
    main()
