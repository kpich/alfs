"""Find MWE occurrences by joining adjacent unigram tokens in seg data.

Given a sequence of component tokens (e.g. ["a", "priori"] or
["well", "-", "known"]), finds all locations in the corpus where those
tokens appear consecutively.

Usage:
    from alfs.mwe.find_occurrences import find_mwe_occurrences, load_all_seg_data

    all_tokens = load_all_seg_data(Path("by_prefix"))
    occs = find_mwe_occurrences(all_tokens, ["a", "priori"])
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from alfs.data_models.occurrence import Occurrence


def load_all_seg_data(seg_data_dir: Path) -> pl.LazyFrame:
    """Load all by_prefix/*/occurrences.parquet into a single LazyFrame.

    Sorted by (doc_id, byte_offset) so consecutive rows within a doc
    represent adjacent tokens.
    """
    parquet_files = sorted(seg_data_dir.glob("*/occurrences.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No occurrences.parquet files found in {seg_data_dir}")
    return pl.concat([pl.scan_parquet(str(f)) for f in parquet_files]).sort(
        ["doc_id", "byte_offset"]
    )


def _build_bigram_df(tokens: pl.LazyFrame) -> pl.LazyFrame:
    """Pair each token with its successor within the same document."""
    return tokens.with_columns(
        pl.col("form").shift(-1).over("doc_id").alias("next_form"),
        pl.col("byte_offset").shift(-1).over("doc_id").alias("next_byte_offset"),
    )


def _build_trigram_df(tokens: pl.LazyFrame) -> pl.LazyFrame:
    """Pair each token with its two successors within the same document."""
    return tokens.with_columns(
        pl.col("form").shift(-1).over("doc_id").alias("next_form"),
        pl.col("byte_offset").shift(-1).over("doc_id").alias("next_byte_offset"),
        pl.col("form").shift(-2).over("doc_id").alias("next2_form"),
        pl.col("byte_offset").shift(-2).over("doc_id").alias("next2_byte_offset"),
    )


def find_mwe_occurrences(
    all_tokens: pl.LazyFrame,
    components: list[str],
    *,
    case_sensitive: bool = True,
) -> list[Occurrence]:
    """Find all corpus locations where components appear as consecutive tokens.

    Returns Occurrence(doc_id, byte_offset) for each match, where byte_offset
    is the offset of the first component token.
    """
    if len(components) == 2:
        return _find_bigram(all_tokens, components, case_sensitive=case_sensitive)
    elif len(components) == 3:
        return _find_trigram(all_tokens, components, case_sensitive=case_sensitive)
    else:
        raise ValueError(
            f"Only bigram and trigram MWEs are supported, "
            f"got {len(components)} components"
        )


def _form_filter(col_name: str, target: str, case_sensitive: bool) -> pl.Expr:
    if case_sensitive:
        return pl.col(col_name) == target
    return pl.col(col_name).str.to_lowercase() == target.lower()


def _find_bigram(
    all_tokens: pl.LazyFrame,
    components: list[str],
    *,
    case_sensitive: bool = True,
) -> list[Occurrence]:
    df = _build_bigram_df(all_tokens)
    matches = (
        df.filter(
            _form_filter("form", components[0], case_sensitive)
            & _form_filter("next_form", components[1], case_sensitive)
        )
        .select(["doc_id", "byte_offset"])
        .collect()
    )
    return [
        Occurrence(doc_id=row[0], byte_offset=row[1]) for row in matches.iter_rows()
    ]


def _find_trigram(
    all_tokens: pl.LazyFrame,
    components: list[str],
    *,
    case_sensitive: bool = True,
) -> list[Occurrence]:
    df = _build_trigram_df(all_tokens)
    matches = (
        df.filter(
            _form_filter("form", components[0], case_sensitive)
            & _form_filter("next_form", components[1], case_sensitive)
            & _form_filter("next2_form", components[2], case_sensitive)
        )
        .select(["doc_id", "byte_offset"])
        .collect()
    )
    return [
        Occurrence(doc_id=row[0], byte_offset=row[1]) for row in matches.iter_rows()
    ]


def mwe_form_from_components(components: list[str]) -> str:
    """Reconstruct the MWE surface form from its component tokens.

    Tokens are joined without space when they start with an apostrophe
    (contractions like wo + n't → won't) or when the component is a
    punctuation character like a hyphen (well + - + known → well-known).
    Otherwise joined with a space (a + priori → a priori).
    """
    if not components:
        return ""
    parts = [components[0]]
    for tok in components[1:]:
        # No space before tokens containing an apostrophe (contractions:
        # wo + n't → won't, I + 'll → I'll) or around hyphens.
        if "'" in tok or tok == "-" or (parts and parts[-1] == "-"):
            parts.append(tok)
        else:
            parts.append(" ")
            parts.append(tok)
    return "".join(parts)
