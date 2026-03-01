"""Stream a MediaWiki XML dump to newline-delimited JSON.

Usage:
    python -m alfs.etl.stream_dump \
        --dump dump.xml.bz2 \
        --source wikibooks \
        --output pages.jsonl
"""

import argparse
import bz2
import json
import xml.etree.ElementTree as ET

NS = "http://www.mediawiki.org/xml/export-0.11/"

BASE_URLS = {
    "wikibooks": "https://en.wikibooks.org/wiki/",
    "wikisource": "https://en.wikisource.org/wiki/",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream MediaWiki XML dump to JSONL")
    parser.add_argument("--dump", required=True, help="Path to .xml.bz2 dump file")
    parser.add_argument(
        "--source",
        required=True,
        choices=list(BASE_URLS),
        help="Source corpus name",
    )
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    args = parser.parse_args()

    count = 0
    with bz2.open(args.dump) as f, open(args.output, "w") as out:
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
            year = int(timestamp[:4]) if timestamp else None

            contributor = revision.find(f"{{{NS}}}contributor")
            username_elem = (
                contributor.find(f"{{{NS}}}username")
                if contributor is not None
                else None
            )
            author = username_elem.text if username_elem is not None else None

            record = {
                "title": title,
                "wikitext": wikitext,
                "year": year,
                "author": author,
                "source": args.source,
            }
            out.write(json.dumps(record) + "\n")
            count += 1
            elem.clear()

    print(f"Found {count} pages, wrote to {args.output}")


if __name__ == "__main__":
    main()
