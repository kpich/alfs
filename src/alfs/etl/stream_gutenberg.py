"""Stream Project Gutenberg books from a cached RDF catalog + individual text fetch.

Reads the gutenberg-catalog.tar.bz2 RDF catalog, filters to English plain-text
books, and lazily fetches each book's plain text from the Gutenberg cache mirror.
"""

from collections.abc import Iterator
from pathlib import Path
import re
import tarfile
import urllib.request
from xml.etree import ElementTree as ET

RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
DCTERMS_NS = "http://purl.org/dc/terms/"
DCAM_NS = "http://purl.org/dc/dcam/"
PGTERMS_NS = "http://www.gutenberg.org/2009/pgterms/"


def _get_text(elem: ET.Element, tag: str) -> str | None:
    child = elem.find(tag)
    return child.text if child is not None else None


def _fetch_book_text(book_id: str) -> str | None:
    """Fetch plain text for a book from the Gutenberg cache mirror."""
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None

    # Strip PG boilerplate
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start = raw.find(start_marker)
    if start != -1:
        start = raw.find("\n", start) + 1
    else:
        start = 0
    end = raw.find(end_marker)
    if end == -1:
        end = len(raw)

    return raw[start:end].strip()


def stream_gutenberg(catalog_path: Path) -> Iterator[dict]:
    """Yield page dicts from a Gutenberg RDF catalog + fetched book texts."""
    with tarfile.open(catalog_path, "r:bz2") as tar:
        for member in tar:
            if not member.name.endswith(".rdf"):
                continue

            f = tar.extractfile(member)
            if f is None:
                continue

            try:
                tree = ET.parse(f)
            except ET.ParseError:
                continue

            root = tree.getroot()
            ebook = root.find(f"{{{RDF_NS}}}Description")
            if ebook is None:
                continue

            # Filter to English text
            lang_elem = ebook.find(f".//{{{DCTERMS_NS}}}language//{{{RDF_NS}}}value")
            if lang_elem is None or lang_elem.text != "en":
                continue

            type_elem = ebook.find(f".//{{{DCTERMS_NS}}}type//{{{RDF_NS}}}value")
            if type_elem is None or type_elem.text != "Text":
                continue

            # Check for plain UTF-8 text file
            has_plain_text = False
            for fmt in ebook.findall(f".//{{{DCTERMS_NS}}}hasFormat"):
                media_type = fmt.find(f".//{{{DCTERMS_NS}}}format//{{{RDF_NS}}}value")
                if media_type is not None and "text/plain" in (media_type.text or ""):
                    has_plain_text = True
                    break
            if not has_plain_text:
                continue

            # Extract book ID from the about attribute
            about = ebook.get(f"{{{RDF_NS}}}about", "")
            m = re.search(r"/(\d+)$", about)
            if not m:
                continue
            book_id = m.group(1)

            year: int | None = (
                None  # Gutenberg release date != original publication year
            )

            # Extract title
            title = (
                _get_text(ebook, f"{{{DCTERMS_NS}}}title") or f"Gutenberg #{book_id}"
            )

            # Extract author
            creator = ebook.find(f".//{{{PGTERMS_NS}}}agent/{{{PGTERMS_NS}}}name")
            author = creator.text if creator is not None else None

            # Fetch text lazily
            text = _fetch_book_text(book_id)
            if not text:
                continue

            yield {
                "title": title,
                "wikitext": text,
                "year": year,
                "author": author,
                "source": "gutenberg",
                "source_url": f"https://www.gutenberg.org/ebooks/{book_id}",
            }
