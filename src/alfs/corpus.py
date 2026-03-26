"""Shared corpus utilities for fetching labeled instances."""

import html as _html

import polars as pl

from alfs.encoding import context_window as _context_window


def _extract_context(
    text: str, byte_offset: int, form: str, context_chars: int, bold_form: bool = False
) -> str:
    snippet, wp = _context_window(text, byte_offset, form, context_chars)
    if bold_form:
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
    min_rating: int = 2,
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
    if filtered.is_empty():
        return []
    needed_ids = filtered["doc_id"].unique().to_list()
    docs_subset = docs.filter(pl.col("doc_id").is_in(needed_ids))
    docs_map = dict(
        zip(
            docs_subset["doc_id"].to_list(), docs_subset["text"].to_list(), strict=False
        )
    )
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
