"""Tokenize docs and emit (form, doc_id, byte_offset) tuples.

Usage:
    python -m alfs.etl.segment_docs \
        --docs docs.parquet --output raw_occurrences.parquet
"""

import argparse
from collections.abc import Iterator

import polars as pl
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import spacy

CHUNK_SIZE = 800_000

PA_SCHEMA = pa.schema(
    [
        ("form", pa.string()),
        ("doc_id", pa.string()),
        ("byte_offset", pa.int64()),
    ]
)


def iter_chunks(text: str) -> Iterator[tuple[str, int]]:
    """Yield (chunk_text, chunk_start_chars) pairs."""
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            # back up to last whitespace
            ws = text.rfind(" ", start, end)
            if ws > start:
                end = ws + 1
        yield text[start:end], start
        start = end


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment docs into occurrences")
    parser.add_argument("--docs", required=True, help="Path to docs.parquet")
    parser.add_argument(
        "--output", required=True, help="Output path for raw_occurrences.parquet"
    )
    parser.add_argument(
        "--shard-index", type=int, default=0, help="Shard index (0-based)"
    )
    parser.add_argument(
        "--num-shards", type=int, default=1, help="Total number of shards"
    )
    args = parser.parse_args()

    print(f"Loading docs from {args.docs}...")
    df = pl.read_parquet(args.docs)
    print(f"Loaded {len(df)} docs")
    df = df[args.shard_index :: args.num_shards]
    print(f"Shard {args.shard_index}/{args.num_shards}: {len(df)} docs")

    nlp = spacy.load("en_core_web_sm")

    with pq.ParquetWriter(args.output, PA_SCHEMA) as writer:
        for row in df.iter_rows(named=True):
            doc_id: str = row["doc_id"]
            text: str = row["text"]
            rows: list[dict[str, object]] = []
            for chunk, chunk_start_chars in iter_chunks(text):
                chunk_start_bytes = len(text[:chunk_start_chars].encode())
                spacy_doc = nlp(chunk)
                for token in spacy_doc:
                    byte_offset = chunk_start_bytes + len(chunk[: token.idx].encode())
                    rows.append(
                        {
                            "form": token.text,
                            "doc_id": doc_id,
                            "byte_offset": byte_offset,
                        }
                    )
            table = pa.table(
                {
                    "form": [r["form"] for r in rows],
                    "doc_id": [r["doc_id"] for r in rows],
                    "byte_offset": [r["byte_offset"] for r in rows],
                },
                schema=PA_SCHEMA,
            )
            writer.write_table(table)

    print(f"Done writing {args.output}")


if __name__ == "__main__":
    main()
