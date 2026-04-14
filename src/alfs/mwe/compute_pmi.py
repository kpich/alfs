"""Compute PMI for bigrams and hyphen-bridged trigrams in the corpus.

Scans all segmentation data, counts unigram and bigram co-occurrences,
and computes pointwise mutual information (PMI) for each pair.  Hyphenated
compounds (X + "-" + Y) are detected as trigrams and scored as a single
MWE candidate.

Usage:
    python -m alfs.mwe.compute_pmi \
        --seg-data-dir by_prefix/ --output pmi_results.parquet \
        [--min-count 10] [--min-pmi 5.0]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re

import polars as pl

from alfs.mwe.find_occurrences import load_all_seg_data

_WORD_RE = re.compile(r"[a-zA-Z]")


def _has_letter(s: str) -> bool:
    return bool(_WORD_RE.search(s))


def compute_bigram_pmi(
    all_tokens: pl.LazyFrame,
    *,
    min_count: int = 10,
    min_pmi: float = 5.0,
) -> pl.DataFrame:
    """Compute PMI for all adjacent bigrams.

    Returns a DataFrame with columns:
        form (str): reconstructed MWE surface form
        components (list[str]): component tokens
        count (int): bigram frequency
        pmi (float): pointwise mutual information
    """
    # Materialize for counting
    tokens = all_tokens.collect()
    n_total = len(tokens)

    # Unigram counts
    unigram_counts = tokens.group_by("form").agg(pl.len().alias("count"))
    unigram_map: dict[str, int] = dict(
        zip(
            unigram_counts["form"].to_list(),
            unigram_counts["count"].to_list(),
            strict=False,
        )
    )

    # Bigram extraction: pair each token with its successor in same doc
    bigrams = (
        tokens.lazy()
        .with_columns(
            pl.col("form").shift(-1).over("doc_id").alias("next_form"),
        )
        .filter(pl.col("next_form").is_not_null())
        .group_by(["form", "next_form"])
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") >= min_count)
        .collect()
    )

    # Filter: both tokens must contain a letter
    bigrams = bigrams.filter(
        pl.col("form").map_elements(_has_letter, return_dtype=pl.Boolean)
        & pl.col("next_form").map_elements(_has_letter, return_dtype=pl.Boolean)
    )

    # Compute PMI
    log2_n = math.log2(n_total)
    rows = []
    for row in bigrams.iter_rows(named=True):
        w1, w2, count = row["form"], row["next_form"], row["count"]
        c1 = unigram_map.get(w1, 0)
        c2 = unigram_map.get(w2, 0)
        if c1 == 0 or c2 == 0:
            continue
        pmi = math.log2(count) - math.log2(c1) - math.log2(c2) + log2_n
        if pmi >= min_pmi:
            rows.append(
                {
                    "form": f"{w1} {w2}"
                    if not w2.startswith("'") and "'" not in w2
                    else f"{w1}{w2}",
                    "components": [w1, w2],
                    "count": count,
                    "pmi": pmi,
                }
            )

    if not rows:
        return pl.DataFrame(
            schema={
                "form": pl.String,
                "components": pl.List(pl.String),
                "count": pl.Int64,
                "pmi": pl.Float64,
            }
        )
    return pl.DataFrame(rows).sort("pmi", descending=True)


def compute_hyphen_trigram_pmi(
    all_tokens: pl.LazyFrame,
    *,
    min_count: int = 10,
    min_pmi: float = 5.0,
) -> pl.DataFrame:
    """Compute PMI for hyphen-bridged trigrams (X + "-" + Y).

    PMI is computed for the (X, Y) outer pair, treating the hyphen as
    transparent.  The components list includes all three tokens.
    """
    tokens = all_tokens.collect()
    n_total = len(tokens)

    unigram_counts = tokens.group_by("form").agg(pl.len().alias("count"))
    unigram_map: dict[str, int] = dict(
        zip(
            unigram_counts["form"].to_list(),
            unigram_counts["count"].to_list(),
            strict=False,
        )
    )

    # Trigram extraction: token + next + next2, filtering for hyphen in the middle
    trigrams = (
        tokens.lazy()
        .with_columns(
            pl.col("form").shift(-1).over("doc_id").alias("mid_form"),
            pl.col("form").shift(-2).over("doc_id").alias("next2_form"),
        )
        .filter(
            pl.col("mid_form").is_not_null()
            & pl.col("next2_form").is_not_null()
            & (pl.col("mid_form") == "-")
        )
        .group_by(["form", "next2_form"])
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") >= min_count)
        .collect()
    )

    # Filter: both outer tokens must contain a letter
    trigrams = trigrams.filter(
        pl.col("form").map_elements(_has_letter, return_dtype=pl.Boolean)
        & pl.col("next2_form").map_elements(_has_letter, return_dtype=pl.Boolean)
    )

    log2_n = math.log2(n_total)
    rows = []
    for row in trigrams.iter_rows(named=True):
        w1, w3, count = row["form"], row["next2_form"], row["count"]
        c1 = unigram_map.get(w1, 0)
        c3 = unigram_map.get(w3, 0)
        if c1 == 0 or c3 == 0:
            continue
        pmi = math.log2(count) - math.log2(c1) - math.log2(c3) + log2_n
        if pmi >= min_pmi:
            rows.append(
                {
                    "form": f"{w1}-{w3}",
                    "components": [w1, "-", w3],
                    "count": count,
                    "pmi": pmi,
                }
            )

    if not rows:
        return pl.DataFrame(
            schema={
                "form": pl.String,
                "components": pl.List(pl.String),
                "count": pl.Int64,
                "pmi": pl.Float64,
            }
        )
    return pl.DataFrame(rows).sort("pmi", descending=True)


def compute_pmi(
    seg_data_dir: Path,
    *,
    min_count: int = 10,
    min_pmi: float = 5.0,
) -> pl.DataFrame:
    """Compute PMI for all bigrams and hyphen-bridged trigrams.

    Returns combined DataFrame sorted by PMI descending.
    """
    all_tokens = load_all_seg_data(seg_data_dir)

    print("Computing bigram PMI...")
    bigrams = compute_bigram_pmi(all_tokens, min_count=min_count, min_pmi=min_pmi)
    print(f"  {len(bigrams)} bigram candidates")

    print("Computing hyphen-bridged trigram PMI...")
    trigrams = compute_hyphen_trigram_pmi(
        all_tokens, min_count=min_count, min_pmi=min_pmi
    )
    print(f"  {len(trigrams)} hyphen-trigram candidates")

    combined = pl.concat([bigrams, trigrams])
    # Deduplicate: if a hyphenated form also appears as a bigram (e.g. "well -"),
    # keep the one with higher PMI
    combined = (
        combined.sort("pmi", descending=True)
        .unique(subset=["form"], keep="first")
        .sort("pmi", descending=True)
    )
    print(f"  {len(combined)} total candidates after dedup")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute PMI for corpus bigrams and hyphen-bridged trigrams"
    )
    parser.add_argument("--seg-data-dir", required=True, help="Path to by_prefix/ dir")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--min-pmi", type=float, default=5.0)
    args = parser.parse_args()

    result = compute_pmi(
        Path(args.seg_data_dir),
        min_count=args.min_count,
        min_pmi=args.min_pmi,
    )
    result.write_parquet(args.output)
    print(f"Wrote {len(result)} candidates to {args.output}")

    if len(result) > 0:
        print("\nTop 20 by PMI:")
        for row in result.head(20).iter_rows(named=True):
            print(
                f"  {row['form']!r:30s} count={row['count']:6d}  pmi={row['pmi']:.2f}"
            )


if __name__ == "__main__":
    main()
