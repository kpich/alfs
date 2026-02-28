"""Shared corpus utilities for fetching labeled instances."""

import html as _html

import polars as pl


def _extract_context(
    text: str, byte_offset: int, form: str, context_chars: int, bold_form: bool = False
) -> str:
    char_offset = len(text.encode()[:byte_offset].decode())
    start = max(0, char_offset - context_chars)
    end = char_offset + len(form) + context_chars
    snippet = text[start:end]
    if bold_form:
        wp = char_offset - start
        return (
            _html.escape(snippet[:wp])
            + "<strong>"
            + _html.escape(snippet[wp : wp + len(form)])
            + "</strong>"
            + _html.escape(snippet[wp + len(form) :])
        )
    return snippet


def fetch_instances(
    form: str,
    sense_key: str,
    labeled: pl.DataFrame,
    docs: pl.DataFrame,
    *,
    min_rating: int = 3,
    context_chars: int = 150,
    max_instances: int = 10,
    bold_form: bool = False,
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
                _extract_context(
                    text, row["byte_offset"], form, context_chars, bold_form
                )
            )
    return results
