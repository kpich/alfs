"""Parse a Wikibooks XML dump and write sampled docs to Parquet.

Usage:
    python -m alfs.etl.parse_dump \
        --dump dump.xml.bz2 --num-docs 10 --seed 42 --output docs.parquet
"""

import argparse
import bz2
import random
from urllib.parse import quote
import xml.etree.ElementTree as ET

import mwparserfromhell
import polars as pl

from alfs.data_models.doc import Doc

NS = "http://www.mediawiki.org/xml/export-0.11/"


def stream_pages(dump_path: str) -> list[dict]:
    pages = []
    with bz2.open(dump_path) as f:
        for _event, elem in ET.iterparse(f, events=["end"]):
            if elem.tag != f"{{{NS}}}page":
                continue

            ns_elem = elem.find(f"{{{NS}}}ns")
            if ns_elem is None or ns_elem.text != "0":
                elem.clear()
                continue

            title_elem = elem.find(f"{{{NS}}}title")
            title = title_elem.text if title_elem is not None else ""
            if not title or "/" in title:
                elem.clear()
                continue

            revision = elem.find(f"{{{NS}}}revision")
            if revision is None:
                elem.clear()
                continue

            text_elem = revision.find(f"{{{NS}}}text")
            wikitext = text_elem.text if text_elem is not None else ""
            if not wikitext:
                elem.clear()
                continue

            timestamp_elem = revision.find(f"{{{NS}}}timestamp")
            timestamp = timestamp_elem.text if timestamp_elem is not None else ""
            year = int(timestamp[:4]) if timestamp else 0

            contributor = revision.find(f"{{{NS}}}contributor")
            username_elem = (
                contributor.find(f"{{{NS}}}username")
                if contributor is not None
                else None
            )
            author = username_elem.text if username_elem is not None else "unknown"

            pages.append(
                {"title": title, "wikitext": wikitext, "year": year, "author": author}
            )
            elem.clear()

    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Wikibooks XML dump to Parquet")
    parser.add_argument("--dump", required=True, help="Path to .xml.bz2 dump file")
    parser.add_argument(
        "--num-docs", type=int, required=True, help="Number of docs to sample"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--output", required=True, help="Output Parquet file path")
    args = parser.parse_args()

    print(f"Streaming pages from {args.dump}...")
    all_pages = stream_pages(args.dump)
    print(f"Found {len(all_pages)} top-level pages")

    sampled = random.Random(args.seed).sample(
        all_pages, min(args.num_docs, len(all_pages))
    )
    print(f"Sampled {len(sampled)} pages, parsing wikitext...")

    docs: list[Doc] = []
    for page in sampled:
        text = mwparserfromhell.parse(page["wikitext"]).strip_code().strip()
        title = page["title"]
        source_url = f"https://en.wikibooks.org/wiki/{quote(title.replace(' ', '_'))}"
        docs.append(
            Doc(
                title=title,
                author=page["author"],
                year=page["year"],
                text=text,
                source_url=source_url,
            )
        )

    print(f"Writing {len(docs)} docs to {args.output}...")
    df = pl.DataFrame([d.model_dump() for d in docs])
    df.write_parquet(args.output)
    print(f"Done. Shape: {df.shape}")


if __name__ == "__main__":
    main()
