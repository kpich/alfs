"""Compile viewer data from senses.db + labeled.db + docs.parquet.

Usage:
    python -m alfs.viewer.compile \\
        --senses-db senses.db --labeled-db labeled.db --docs docs.parquet \\
        --output data.json
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


def compile_entries(
    alfs: Alfs,
    labeled: pl.DataFrame,
    docs: pl.DataFrame,
    corpus_counts: dict[str, int],
    timestamps: dict[str, str | None] | None = None,
) -> dict:
    """Build the viewer entries dict, skipping redirect forms."""
    joined = labeled.filter(pl.col("rating") >= 1).join(
        docs.select(["doc_id", "year"])
        .drop_nulls("year")
        .with_columns(pl.col("year").cast(pl.Int32)),
        on="doc_id",
        how="inner",
    )
    redirect_map = {
        form: alf.redirect
        for form, alf in alfs.entries.items()
        if alf.redirect is not None
    }
    if redirect_map:
        rdf = pl.DataFrame(
            {
                "form": list(redirect_map.keys()),
                "canonical": list(redirect_map.values()),
            }
        )
        joined = (
            joined.join(rdf, on="form", how="left")
            .with_columns(pl.coalesce(["canonical", "form"]).alias("form"))
            .drop("canonical")
        )
    counts = joined.group_by(["form", "sense_key", "year"]).agg(pl.len().alias("count"))

    by_year_per_form: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in counts.iter_rows(named=True):
        by_year_per_form[row["form"]][str(row["year"])][row["sense_key"]] = row["count"]

    entries: dict[str, dict] = {}
    for form, alf in alfs.entries.items():
        if alf.redirect is not None:
            continue
        senses = []
        for top_idx, sense in enumerate(alf.senses):
            sense_entry: dict = {
                "key": sense_key(top_idx),
                "definition": sense.definition,
                "pos": sense.pos.value if sense.pos else None,
            }
            if sense.subsenses:
                sense_entry["subsenses"] = [
                    {"key": sense_key(top_idx, sub_idx), "definition": defn}
                    for sub_idx, defn in enumerate(sense.subsenses)
                ]
            if sense.morph_base is not None:
                sense_entry["morph_base"] = sense.morph_base
            if sense.morph_relation is not None:
                sense_entry["morph_relation"] = sense.morph_relation
            sense_entry["instances"] = fetch_instances(
                form,
                sense_key(top_idx),
                labeled,
                docs,
                min_rating=3,
                context_chars=60,
                max_instances=3,
                bold_form=True,
            )
            senses.append(sense_entry)

        entries[form] = {
            "senses": senses,
            "by_year": dict(by_year_per_form.get(form, {})),
            "updated_at": (timestamps or {}).get(form),
        }

    sorted_forms = sorted(entries, key=lambda f: corpus_counts.get(f, 0), reverse=True)
    n = len(sorted_forms)
    for rank, form in enumerate(sorted_forms, start=1):
        entries[form]["percentile"] = math.ceil(rank / n * 100) if n else 100

    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile viewer data")
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument("--labeled-db", required=True, help="Path to labeled.db")
    parser.add_argument("--docs", required=True, help="Path to docs.parquet")
    parser.add_argument("--by-prefix-dir", required=True, help="Path to by_prefix/ dir")
    parser.add_argument("--output", required=True, help="Path to output data.json")
    args = parser.parse_args()

    sense_store = SenseStore(Path(args.senses_db))
    entries_dict = sense_store.all_entries()
    alfs = Alfs(entries=entries_dict)
    timestamps = sense_store.all_timestamps()

    occ_store = OccurrenceStore(Path(args.labeled_db))
    labeled = occ_store.to_polars()
    docs = pl.read_parquet(args.docs)

    alfs_forms = list(alfs.entries.keys())
    corpus_df = (
        pl.scan_parquet(str(Path(args.by_prefix_dir) / "**" / "*.parquet"))
        .filter(pl.col("form").is_in(alfs_forms))
        .group_by("form")
        .agg(pl.len().alias("count"))
        .collect()
    )
    corpus_counts = dict(
        zip(corpus_df["form"].to_list(), corpus_df["count"].to_list(), strict=False)
    )

    entries = compile_entries(alfs, labeled, docs, corpus_counts, timestamps)

    output = {"entries": entries}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"Wrote {len(entries)} entries → {args.output}")


if __name__ == "__main__":
    main()
