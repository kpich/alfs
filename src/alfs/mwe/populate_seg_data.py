"""Populate by_prefix seg data with MWE occurrences.

Scans SenseStore for multi-word expression forms (those that spaCy
tokenizes into >1 token), finds their corpus occurrences by joining
adjacent unigrams in existing seg data, and merges the results into the
by_prefix layout so the batch labeling pipeline can discover them.

Usage:
    python -m alfs.mwe.populate_seg_data \
        --senses-db ../alfs_data/senses.db \
        --seg-data-dir ../seg_data/by_prefix
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import spacy

from alfs.data_models.sense_store import SenseStore
from alfs.mwe.find_occurrences import find_mwe_occurrences, load_all_seg_data
from alfs.seg.aggregate_occurrences import aggregate


def _tokenize_form(nlp: spacy.language.Language, form: str) -> list[str]:
    """Tokenize a form using spaCy, returning component tokens."""
    return [tok.text for tok in nlp(form)]


def find_mwe_forms(
    nlp: spacy.language.Language, forms: list[str]
) -> list[tuple[str, list[str]]]:
    """Return (form, components) pairs for forms that tokenize to >1 token."""
    mwe_forms = []
    for form in forms:
        components = _tokenize_form(nlp, form)
        if len(components) > 1:
            mwe_forms.append((form, components))
    return mwe_forms


def populate(senses_db: Path, seg_data_dir: Path) -> int:
    """Find MWE occurrences and merge them into seg data.

    Returns the number of new occurrences added.
    """
    store = SenseStore(senses_db)
    all_forms = store.all_forms()

    nlp = spacy.load("en_core_web_sm")
    nlp.select_pipes(enable=[])  # disable all pipes, tokenizer only

    mwe_forms = find_mwe_forms(nlp, all_forms)
    if not mwe_forms:
        print("No MWE forms found in SenseStore.")
        return 0

    print(f"Found {len(mwe_forms)} MWE forms in SenseStore.")

    try:
        all_tokens = load_all_seg_data(seg_data_dir)
    except FileNotFoundError:
        print(f"No seg data found in {seg_data_dir}, nothing to do.")
        return 0

    # Collect all MWE occurrences into a single DataFrame
    rows: list[dict[str, str | int]] = []
    for form, components in mwe_forms:
        if len(components) not in (2, 3):
            print(f"  Skipping {form!r}: {len(components)} components (unsupported)")
            continue
        occs = find_mwe_occurrences(all_tokens, components)
        print(f"  {form!r}: {len(occs)} occurrences")
        for occ in occs:
            rows.append(
                {"form": form, "doc_id": occ.doc_id, "byte_offset": occ.byte_offset}
            )

    if not rows:
        print("No MWE occurrences found.")
        return 0

    new_df = pl.DataFrame(
        rows, schema={"form": pl.String, "doc_id": pl.String, "byte_offset": pl.Int64}
    )

    # Deduplicate against existing seg data: read existing rows for these
    # forms and remove any that already exist.
    mwe_form_set = {form for form, _ in mwe_forms}
    existing_df = (
        all_tokens.filter(pl.col("form").is_in(mwe_form_set))
        .select(["form", "doc_id", "byte_offset"])
        .collect()
    )

    if len(existing_df) > 0:
        # Anti-join to keep only truly new occurrences
        before = len(new_df)
        new_df = new_df.join(
            existing_df, on=["form", "doc_id", "byte_offset"], how="anti"
        )
        print(f"Filtered {before - len(new_df)} already-present occurrences.")

    if len(new_df) == 0:
        print("All occurrences already present in seg data.")
        return 0

    print(f"Merging {len(new_df)} new MWE occurrences into seg data...")
    aggregate(new_df, seg_data_dir, merge=True)
    return len(new_df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate seg data with MWE occurrences"
    )
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument(
        "--seg-data-dir", required=True, help="Path to by_prefix seg data directory"
    )
    args = parser.parse_args()

    n = populate(Path(args.senses_db), Path(args.seg_data_dir))
    print(f"Done. {n} new occurrences added.")


if __name__ == "__main__":
    main()
