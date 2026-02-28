"""Shared corpus utilities for fetching labeled instances."""

import polars as pl


def _extract_context(text: str, byte_offset: int, form: str, context_chars: int) -> str:
    char_offset = len(text.encode()[:byte_offset].decode())
    start = max(0, char_offset - context_chars)
    end = char_offset + len(form) + context_chars
    return text[start:end]


def fetch_instances(
    form: str,
    sense_key: str,
    labeled: pl.DataFrame,
    docs: pl.DataFrame,
    *,
    min_rating: int = 3,
    context_chars: int = 150,
    max_instances: int = 10,
) -> list[str]:
    """Return context snippets for high-confidence labeled occurrences of a sense."""
    filtered = (
        labeled.filter(pl.col("form") == form)
        .filter(pl.col("sense_key") == sense_key)
        .filter(pl.col("rating") >= min_rating)
        .head(max_instances)
    )
    docs_map = dict(zip(docs["doc_id"].to_list(), docs["text"].to_list(), strict=False))
    results = []
    for row in filtered.iter_rows(named=True):
        text = docs_map.get(row["doc_id"], "")
        if text:
            results.append(
                _extract_context(text, row["byte_offset"], form, context_chars)
            )
    return results
