"""Tests for MWE occurrence finding."""

from __future__ import annotations

import polars as pl
import pytest

from alfs.mwe.find_occurrences import (
    find_mwe_occurrences,
    mwe_form_from_components,
)


def _make_tokens(rows: list[tuple[str, str, int]]) -> pl.LazyFrame:
    """Build a LazyFrame from (form, doc_id, byte_offset) tuples."""
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


def test_find_bigram():
    tokens = _make_tokens(
        [
            ("a", "d1", 0),
            ("priori", "d1", 2),
            ("a", "d1", 20),
            ("cat", "d1", 22),
            ("a", "d2", 0),
            ("priori", "d2", 2),
        ]
    )
    occs = find_mwe_occurrences(tokens, ["a", "priori"])
    assert len(occs) == 2
    assert {(o.doc_id, o.byte_offset) for o in occs} == {("d1", 0), ("d2", 0)}


def test_find_bigram_no_cross_doc():
    tokens = _make_tokens(
        [
            ("a", "d1", 100),
            ("priori", "d2", 0),
        ]
    )
    occs = find_mwe_occurrences(tokens, ["a", "priori"])
    assert len(occs) == 0


def test_find_trigram():
    tokens = _make_tokens(
        [
            ("well", "d1", 0),
            ("-", "d1", 4),
            ("known", "d1", 5),
            ("well", "d1", 20),
            ("done", "d1", 25),
        ]
    )
    occs = find_mwe_occurrences(tokens, ["well", "-", "known"])
    assert len(occs) == 1
    assert occs[0].doc_id == "d1"
    assert occs[0].byte_offset == 0


def test_find_trigram_no_match():
    tokens = _make_tokens(
        [
            ("well", "d1", 0),
            ("-", "d1", 4),
            ("done", "d1", 5),
        ]
    )
    occs = find_mwe_occurrences(tokens, ["well", "-", "known"])
    assert len(occs) == 0


def test_case_insensitive():
    tokens = _make_tokens(
        [
            ("A", "d1", 0),
            ("Priori", "d1", 2),
        ]
    )
    occs = find_mwe_occurrences(tokens, ["a", "priori"], case_sensitive=False)
    assert len(occs) == 1


def test_unsupported_ngram_length():
    tokens = _make_tokens([("a", "d1", 0)])
    with pytest.raises(ValueError, match="Only bigram and trigram"):
        find_mwe_occurrences(tokens, ["a", "b", "c", "d"])


def test_mwe_form_phrasal():
    assert mwe_form_from_components(["a", "priori"]) == "a priori"
    assert mwe_form_from_components(["take", "care"]) == "take care"


def test_mwe_form_contraction():
    assert mwe_form_from_components(["wo", "n't"]) == "won't"
    assert mwe_form_from_components(["I", "'ll"]) == "I'll"
    assert mwe_form_from_components(["ca", "n't"]) == "can't"


def test_mwe_form_hyphenated():
    assert mwe_form_from_components(["well", "-", "known"]) == "well-known"
    assert mwe_form_from_components(["e", "-", "mail"]) == "e-mail"


def test_mwe_form_empty():
    assert mwe_form_from_components([]) == ""
