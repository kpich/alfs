"""Tests for PMI computation."""

from __future__ import annotations

import polars as pl

from alfs.mwe.compute_pmi import compute_bigram_pmi, compute_hyphen_trigram_pmi


def _make_tokens(rows: list[tuple[str, str, int]]) -> pl.LazyFrame:
    return (
        pl.DataFrame(
            {
                "form": [r[0] for r in rows],
                "doc_id": [r[1] for r in rows],
                "byte_offset": [r[2] for r in rows],
            },
        )
        .sort(["doc_id", "byte_offset"])
        .lazy()
    )


def test_bigram_pmi_basic():
    # Build a corpus where "a priori" co-occurs frequently relative to
    # the individual frequencies of "a" and "priori".
    rows: list[tuple[str, str, int]] = []
    offset = 0
    for i in range(50):
        doc_id = f"d{i}"
        rows.append(("a", doc_id, 0))
        rows.append(("priori", doc_id, 2))
        offset += 10
    # Add some noise: "a" appearing without "priori"
    for i in range(50, 60):
        doc_id = f"d{i}"
        rows.append(("a", doc_id, 0))
        rows.append(("cat", doc_id, 2))
    # Add "cat" appearing alone
    for i in range(60, 70):
        doc_id = f"d{i}"
        rows.append(("cat", doc_id, 0))
        rows.append(("dog", doc_id, 4))

    tokens = _make_tokens(rows)
    result = compute_bigram_pmi(tokens, min_count=5, min_pmi=0.0)

    # "a priori" should be in results with positive PMI
    forms = result["form"].to_list()
    assert "a priori" in forms

    # Get its PMI
    a_priori_row = result.filter(pl.col("form") == "a priori")
    assert a_priori_row["pmi"][0] > 0


def test_bigram_pmi_filters_low_count():
    rows: list[tuple[str, str, int]] = []
    for i in range(3):  # only 3 occurrences
        rows.append(("rare", f"d{i}", 0))
        rows.append(("word", f"d{i}", 5))
    tokens = _make_tokens(rows)
    result = compute_bigram_pmi(tokens, min_count=5, min_pmi=0.0)
    assert len(result) == 0


def test_bigram_pmi_filters_non_word():
    # Bigrams like "." + "The" should be filtered out
    rows: list[tuple[str, str, int]] = []
    for i in range(20):
        rows.append((".", f"d{i}", 0))
        rows.append(("The", f"d{i}", 2))
    tokens = _make_tokens(rows)
    result = compute_bigram_pmi(tokens, min_count=5, min_pmi=0.0)
    # Should not contain any bigram starting with "."
    if len(result) > 0:
        for form in result["form"].to_list():
            assert not form.startswith(".")


def test_hyphen_trigram_pmi():
    rows: list[tuple[str, str, int]] = []
    for i in range(30):
        doc_id = f"d{i}"
        rows.append(("well", doc_id, 0))
        rows.append(("-", doc_id, 4))
        rows.append(("known", doc_id, 5))
    # noise
    for i in range(30, 50):
        doc_id = f"d{i}"
        rows.append(("well", doc_id, 0))
        rows.append(("done", doc_id, 5))
    for i in range(50, 70):
        doc_id = f"d{i}"
        rows.append(("known", doc_id, 0))
        rows.append(("fact", doc_id, 6))

    tokens = _make_tokens(rows)
    result = compute_hyphen_trigram_pmi(tokens, min_count=5, min_pmi=0.0)

    forms = result["form"].to_list()
    assert "well-known" in forms
    wk = result.filter(pl.col("form") == "well-known")
    assert wk["pmi"][0] > 0
    assert wk["components"].to_list()[0] == ["well", "-", "known"]
