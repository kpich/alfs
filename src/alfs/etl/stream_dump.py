"""Stream a MediaWiki XML dump to newline-delimited JSON.

Usage:
    python -m alfs.etl.stream_dump \
        --dump dump.xml.bz2 \
        --source wikibooks \
        --output pages.jsonl
"""

import argparse
import bz2
from collections.abc import Iterator
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

NS = "http://www.mediawiki.org/xml/export-0.11/"


def stream_pages(dump_path: Path, source: str) -> Iterator[dict]:
    """Yield page dicts from a MediaWiki XML dump (namespace 0, non-redirects)."""
    with bz2.open(dump_path) as f:
        for _event, elem in ET.iterparse(f, events=["end"]):
            if elem.tag != f"{{{NS}}}page":
                continue

            ns_elem = elem.find(f"{{{NS}}}ns")
            if ns_elem is None or ns_elem.text != "0":
                elem.clear()
                continue

            if elem.find(f"{{{NS}}}redirect") is not None:
                elem.clear()
                continue

            title_elem = elem.find(f"{{{NS}}}title")
            title = title_elem.text if title_elem is not None else ""
            if not title:
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
            year = int(timestamp[:4]) if timestamp else None
            if source == "wikisource" and wikitext:
                m = re.search(r"\|\s*year\s*=\s*(\d{4})", wikitext)
                if m:
                    year = int(m.group(1))

            contributor = revision.find(f"{{{NS}}}contributor")
            username_elem = (
                contributor.find(f"{{{NS}}}username")
                if contributor is not None
                else None
            )
            author = username_elem.text if username_elem is not None else None

            yield {
                "title": title,
                "wikitext": wikitext,
                "year": year,
                "author": author,
                "source": source,
            }
            elem.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream MediaWiki XML dump to JSONL")
    parser.add_argument("--dump", required=True, help="Path to .xml.bz2 dump file")
    from alfs.etl.sources import SOURCES

    parser.add_argument(
        "--source",
        required=True,
        choices=[name for name, s in SOURCES.items() if s.type == "mediawiki"],
        help="Source corpus name",
    )
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    args = parser.parse_args()

    count = 0
    with open(args.output, "w") as out:
        for page in stream_pages(Path(args.dump), args.source):
            out.write(json.dumps(page) + "\n")
            count += 1

    print(f"Found {count} pages, wrote to {args.output}")


if __name__ == "__main__":
    main()
