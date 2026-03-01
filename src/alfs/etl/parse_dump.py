"""Parse a JSONL page dump and write docs to Parquet.

Usage:
    python -m alfs.etl.parse_dump \
        --pages pages.jsonl \
        --source wikibooks \
        --shard-index 0 \
        --num-shards 4 \
        --output docs.parquet
"""

import argparse
import hashlib
import json
from urllib.parse import quote

import mwparserfromhell
import polars as pl

from alfs.data_models.doc import Doc

BASE_URLS = {
    "wikibooks": "https://en.wikibooks.org/wiki/",
    "wikisource": "https://en.wikisource.org/wiki/",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse JSONL page dump to Parquet")
    parser.add_argument("--pages", required=True, help="Path to JSONL from stream_dump")
    parser.add_argument("--output", required=True, help="Output Parquet file path")
    parser.add_argument(
        "--source",
        required=True,
        choices=list(BASE_URLS),
        help="Source corpus name",
    )
    parser.add_argument(
        "--shard-index", type=int, default=0, help="Shard index (0-based)"
    )
    parser.add_argument(
        "--num-shards", type=int, default=1, help="Total number of shards"
    )
    args = parser.parse_args()

    base_url = BASE_URLS[args.source]

    with open(args.pages) as f:
        all_pages = [json.loads(line) for line in f]
    shard = all_pages[args.shard_index :: args.num_shards]
    n_total = len(all_pages)
    print(
        f"Shard {args.shard_index}/{args.num_shards}: {len(shard)} of {n_total} pages"
    )

    docs: list[Doc] = []
    for page in shard:
        text = mwparserfromhell.parse(page["wikitext"]).strip_code().strip()
        doc_id = hashlib.sha256(text.encode()).hexdigest()[:8]
        title = page["title"]
        source_url = f"{base_url}{quote(title.replace(' ', '_'))}"
        docs.append(
            Doc(
                doc_id=doc_id,
                title=title,
                author=page["author"],
                year=page["year"],
                text=text,
                source_url=source_url,
                source=args.source,
            )
        )

    print(f"Writing {len(docs)} docs to {args.output}...")
    df = pl.DataFrame([d.model_dump() for d in docs])
    df.write_parquet(args.output)
    print(f"Done. Shape: {df.shape}")


if __name__ == "__main__":
    main()
