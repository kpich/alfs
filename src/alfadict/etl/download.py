"""Download Wikibooks pages by title and write to Parquet.

Usage:
    python -m alfadict.etl.download --titles titles.txt --output docs.parquet
"""

import argparse
from urllib.parse import quote

import mwparserfromhell
import polars as pl
import requests

from alfadict.data_models.doc import Doc

API_URL = "https://en.wikibooks.org/w/api.php"
HEADERS = {"User-Agent": "alfadict/0.1 (https://github.com/alfadict/alfadict; bot)"}


def fetch_doc(title: str) -> Doc | None:
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions",
        "rvprop": "content|timestamp|user",
        "rvlimit": 1,
        "rvdir": "newer",
        "format": "json",
    }

    response = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    data = response.json()

    pages = data["query"]["pages"]
    page = next(iter(pages.values()))

    if "missing" in page:
        print(f"  Skipping missing page: {title}")
        return None

    revisions = page.get("revisions")
    if not revisions:
        print(f"  Skipping page with no revisions: {title}")
        return None

    rev = revisions[0]
    wikitext = rev.get("*") or rev.get("content", "")
    author = rev.get("user", "unknown")
    timestamp = rev.get("timestamp", "")
    year = int(timestamp[:4]) if timestamp else 0

    parsed = mwparserfromhell.parse(wikitext)
    text = parsed.strip_code().strip()

    source_url = f"https://en.wikibooks.org/wiki/{quote(title.replace(' ', '_'))}"

    return Doc(
        title=title,
        author=author,
        year=year,
        text=text,
        source_url=source_url,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Wikibooks pages to Parquet")
    parser.add_argument(
        "--titles", required=True, help="Input file with one title per line"
    )
    parser.add_argument("--output", required=True, help="Output Parquet file path")
    args = parser.parse_args()

    with open(args.titles) as f:
        titles = [line.strip() for line in f if line.strip()]

    print(f"Downloading {len(titles)} pages...")
    docs: list[Doc] = []
    for i, title in enumerate(titles, 1):
        print(f"  [{i}/{len(titles)}] {title}")
        doc = fetch_doc(title)
        if doc is not None:
            docs.append(doc)

    print(f"Downloaded {len(docs)} docs, writing to {args.output}...")
    df = pl.DataFrame([d.model_dump() for d in docs])
    df.write_parquet(args.output)
    print(f"Done. Shape: {df.shape}")


if __name__ == "__main__":
    main()
