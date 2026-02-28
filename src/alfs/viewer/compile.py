"""Compile viewer data from alfs.json + labeled.parquet + docs.parquet.

Usage:
    python -m alfs.viewer.compile \\
        --alfs alfs.json --labeled labeled.parquet --docs docs.parquet \\
        --output data.json
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import polars as pl

from alfs.data_models.alf import Alfs, sense_key


def compile_entries(
    alfs: Alfs,
    labeled: pl.DataFrame,
    docs: pl.DataFrame,
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
            }
            if sense.subsenses:
                sense_entry["subsenses"] = [
                    {"key": sense_key(top_idx, sub_idx), "definition": defn}
                    for sub_idx, defn in enumerate(sense.subsenses)
                ]
            senses.append(sense_entry)

        entries[form] = {
            "senses": senses,
            "by_year": dict(by_year_per_form.get(form, {})),
        }

    total_counts = {
        form: sum(
            count for yr_data in entry["by_year"].values() for count in yr_data.values()
        )
        for form, entry in entries.items()
    }
    sorted_forms = sorted(entries, key=lambda f: total_counts[f], reverse=True)
    n = len(sorted_forms)
    for rank, form in enumerate(sorted_forms, start=1):
        entries[form]["percentile"] = math.ceil(rank / n * 100) if n else 100

    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile viewer data")
    parser.add_argument("--alfs", required=True, help="Path to alfs.json")
    parser.add_argument("--labeled", required=True, help="Path to labeled.parquet")
    parser.add_argument("--docs", required=True, help="Path to docs.parquet")
    parser.add_argument("--output", required=True, help="Path to output data.json")
    args = parser.parse_args()

    alfs = Alfs.model_validate_json(Path(args.alfs).read_text())
    labeled = pl.read_parquet(args.labeled)
    docs = pl.read_parquet(args.docs)

    entries = compile_entries(alfs, labeled, docs)

    output = {"entries": entries}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"Wrote {len(entries)} entries → {args.output}")


if __name__ == "__main__":
    main()
