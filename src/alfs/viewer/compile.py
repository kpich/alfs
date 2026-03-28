"""Compile viewer data from senses.db + labeled.db + docs.parquet.

Usage (full, single-process):
    python -m alfs.viewer.compile \\
        --senses-db senses.db --labeled-db labeled.db --docs docs.parquet \\
        --corpus-counts corpus_counts.json --output data.json

Usage (batch mode, for parallel NF):
    python -m alfs.viewer.compile \\
        --senses-db senses.db --labeled-db labeled.db --docs docs.parquet \\
        --corpus-counts corpus_counts.json \\
        --batch-idx 0 --num-batches 8 --output entries_0.json
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import polars as pl

from alfs.corpus import fetch_instances
from alfs.data_models.alf import Alfs, sense_key
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore
from alfs.viewer.stats import compute_year_kde


def compile_entries(
    alfs: Alfs,
    labeled: pl.DataFrame,
    docs: pl.DataFrame,
    timestamps: dict[str, str | None] | None = None,
    batch_forms: set[str] | None = None,
) -> dict:
    """Build viewer entries dict for the given forms (all forms if batch_forms is None).

    Returns entries without percentile — caller assigns percentile after merging.
    """
    uuid_to_pos: dict[str, str] = {}
    for _form, alf in alfs.entries.items():
        if alf.redirect is not None:
            continue
        for i, sense in enumerate(alf.senses):
            uuid_to_pos[sense.id] = sense_key(i)

    if uuid_to_pos:
        trans = pl.DataFrame(
            {
                "sense_key": list(uuid_to_pos.keys()),
                "pos_key": list(uuid_to_pos.values()),
            }
        )
        labeled = (
            labeled.join(trans, on="sense_key", how="left")
            .with_columns(pl.coalesce(["pos_key", "sense_key"]).alias("sense_key"))
            .drop("pos_key")
        )

    docs_with_year = (
        docs.select(["doc_id", "year"])
        .drop_nulls("year")
        .with_columns(pl.col("year").cast(pl.Int32))
    )
    redirect_map = {
        form: alf.redirect
        for form, alf in alfs.entries.items()
        if alf.redirect is not None
    }

    def apply_redirect(df: pl.DataFrame) -> pl.DataFrame:
        if not redirect_map:
            return df
        rdf = pl.DataFrame(
            {
                "form": list(redirect_map.keys()),
                "canonical": list(redirect_map.values()),
            }
        )
        return (
            df.join(rdf, on="form", how="left")
            .with_columns(pl.coalesce(["canonical", "form"]).alias("form"))
            .drop("canonical")
        )

    # Numerator: rating>=1 occurrences per (form, sense_key, year)
    joined = apply_redirect(
        labeled.filter(pl.col("rating") >= 1).join(
            docs_with_year, on="doc_id", how="inner"
        )
    )
    counts = joined.group_by(["form", "sense_key", "year"]).agg(pl.len().alias("count"))

    # Denominator: total labeled (any rating) across ALL words per year (global corpus)
    all_joined = labeled.join(docs_with_year, on="doc_id", how="inner")
    year_totals_df = all_joined.group_by("year").agg(pl.len().alias("total"))
    global_year_totals: dict[int, int] = {}
    for row in year_totals_df.iter_rows(named=True):
        global_year_totals[row["year"]] = row["total"]

    # sense_counts per form for senses_bar: {form: {sense_key: count}}
    sense_counts_df = (
        labeled.filter(pl.col("rating") >= 1)
        .group_by(["form", "sense_key"])
        .agg(pl.len().alias("count"))
    )
    sense_counts_per_form: dict[str, dict[str, int]] = defaultdict(dict)
    for row in sense_counts_df.iter_rows(named=True):
        sense_counts_per_form[row["form"]][row["sense_key"]] = row["count"]

    # sense_year_counts per form: {sense_key: {year: count}}
    sense_year_counts_per_form: dict[str, dict[str, dict[int, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in counts.iter_rows(named=True):
        sense_year_counts_per_form[row["form"]][row["sense_key"]][row["year"]] = row[
            "count"
        ]

    entries: dict[str, dict] = {}
    for form, alf in alfs.entries.items():
        if alf.redirect is not None:
            continue
        if batch_forms is not None and form not in batch_forms:
            continue
        senses = []
        for top_idx, sense in enumerate(alf.senses):
            sense_entry: dict = {
                "key": sense_key(top_idx),
                "definition": sense.definition,
                "pos": sense.pos.value if sense.pos else None,
            }
            if sense.morph_base is not None:
                sense_entry["morph_base"] = sense.morph_base
            if sense.morph_relation is not None:
                sense_entry["morph_relation"] = sense.morph_relation
            sense_entry["instances"] = fetch_instances(
                form,
                sense_key(top_idx),
                labeled,
                docs,
                min_rating=1,
                context_chars=60,
                max_instances=3,
                bold_form=True,
            )
            senses.append(sense_entry)

        form_sense_counts = sense_counts_per_form.get(form, {})
        total_positive = sum(form_sense_counts.values())
        senses_bar = (
            [
                {
                    "key": sense_key(top_idx),
                    "pos": sense.pos.value if sense.pos else None,
                    "proportion": form_sense_counts.get(sense_key(top_idx), 0)
                    / total_positive,
                }
                for top_idx, sense in enumerate(alf.senses)
                if form_sense_counts.get(sense_key(top_idx), 0) > 0
            ]
            if total_positive > 0
            else []
        )

        by_year_kde = compute_year_kde(
            sense_year_counts_per_form.get(form, {}),
            global_year_totals,
        )
        entries[form] = {
            "senses": senses,
            "senses_bar": senses_bar,
            "by_year_kde": dict(by_year_kde),
            "updated_at": (timestamps or {}).get(form),
        }

    return entries


def assign_percentiles(entries: dict, corpus_counts: dict[str, int]) -> None:
    """Assign percentile rank in-place based on corpus frequency across all entries."""
    sorted_forms = sorted(entries, key=lambda f: corpus_counts.get(f, 0), reverse=True)
    n = len(sorted_forms)
    for rank, form in enumerate(sorted_forms, start=1):
        entries[form]["percentile"] = math.ceil(rank / n * 100) if n else 100


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile viewer data")
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument("--labeled-db", required=True, help="Path to labeled.db")
    parser.add_argument("--docs", required=True, help="Path to docs.parquet")
    parser.add_argument(
        "--corpus-counts", required=True, help="Pre-computed corpus_counts.json"
    )
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument("--batch-idx", type=int, help="Batch index (0-based)")
    parser.add_argument("--num-batches", type=int, help="Total number of batches")
    args = parser.parse_args()

    sense_store = SenseStore(Path(args.senses_db))
    entries_dict = sense_store.all_entries()
    alfs = Alfs(entries=entries_dict)
    timestamps = sense_store.all_timestamps()

    occ_store = OccurrenceStore(Path(args.labeled_db))
    labeled = occ_store.to_polars()
    docs = pl.read_parquet(args.docs)

    corpus_counts: dict[str, int] = json.loads(Path(args.corpus_counts).read_text())

    batch_forms: set[str] | None = None
    if args.batch_idx is not None:
        num_batches = args.num_batches or 1
        all_forms = sorted(entries_dict.keys())
        batch_size = math.ceil(len(all_forms) / num_batches)
        start = args.batch_idx * batch_size
        batch_forms = set(all_forms[start : start + batch_size])

    entries = compile_entries(alfs, labeled, docs, timestamps, batch_forms)

    # Only assign percentiles in single-process (non-batch) mode
    if batch_forms is None:
        assign_percentiles(entries, corpus_counts)

    output = {"entries": entries}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"Wrote {len(entries)} entries → {args.output}")


if __name__ == "__main__":
    main()
