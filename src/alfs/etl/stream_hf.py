"""Stream a HuggingFace dataset as page dicts.

Used for CC-News and any other HuggingFace-hosted corpus.
"""

from collections.abc import Iterator


def stream_hf(dataset_name: str) -> Iterator[dict]:
    """Yield page dicts from a HuggingFace dataset (streaming mode)."""
    from datasets import load_dataset  # type: ignore[import-untyped]

    ds = load_dataset(dataset_name, split="train", streaming=True)

    for item in ds:
        text = item.get("text", "") or ""
        if not text.strip():
            continue

        title = item.get("title", "") or ""
        date = item.get("date", "") or ""
        year: int | None = (
            int(date[:4]) if date and len(date) >= 4 and date[:4].isdigit() else None
        )
        author = item.get("author", None) or None
        url = item.get("url", "") or ""

        yield {
            "title": title,
            "wikitext": text,
            "year": year,
            "author": author,
            "source": dataset_name,
            "source_url": url,
        }
