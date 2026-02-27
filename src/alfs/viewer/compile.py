"""Compile viewer data from alfs.json + labeled.parquet + docs.parquet.

Usage:
    python -m alfs.viewer.compile \\
        --alfs alfs.json --labeled labeled.parquet --docs docs.parquet \\
        --output data.json
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import polars as pl

from alfs.data_models.alf import Alfs, sense_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile viewer data")
    parser.add_argument("--alfs", required=True, help="Path to alfs.json")
    parser.add_argument("--labeled", required=True, help="Path to labeled.parquet")
    parser.add_argument("--docs", required=True, help="Path to docs.parquet")
    parser.add_argument("--output", required=True, help="Path to output data.json")
    args = parser.parse_args()

    alfs = Alfs.model_validate_json(Path(args.alfs).read_text())

    labeled = pl.read_parquet(args.labeled).filter(pl.col("rating") >= 1)

    docs = (
        pl.read_parquet(args.docs)
        .select(["doc_id", "year"])
        .drop_nulls("year")
        .with_columns(pl.col("year").cast(pl.Int32))
    )

    joined = labeled.join(docs, on="doc_id", how="inner")

    # Group by (form, sense_key, year) and count
    counts = joined.group_by(["form", "sense_key", "year"]).agg(pl.len().alias("count"))

    # Build by_year per form: {year_str: {sense_key: count}}
    by_year_per_form: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in counts.iter_rows(named=True):
        by_year_per_form[row["form"]][str(row["year"])][row["sense_key"]] = row["count"]

    entries: dict[str, dict] = {}
    for form, alf in alfs.entries.items():
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

    output = {"entries": entries}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"Wrote {len(entries)} entries → {args.output}")


if __name__ == "__main__":
    main()
