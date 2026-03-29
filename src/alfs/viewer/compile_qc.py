"""Compile QC data from labeled.db for viewer dashboard.

Usage:
    # Rating histogram stats:
    python -m alfs.viewer.compile_qc \\
        --mode stats --labeled-db labeled.db --output qc_stats.json

    # Instance contexts for a given rating:
    python -m alfs.viewer.compile_qc \\
        --mode instances --labeled-db labeled.db --docs docs.parquet \\
        --senses-db senses.db --rating 0 --output qc_0.json

    # Staleness lag histogram data:
    python -m alfs.viewer.compile_qc \\
        --mode lag --labeled-db labeled.db --senses-db senses.db --output qc_lag.json
"""

import argparse
import json
from pathlib import Path

import polars as pl

from alfs.corpus import _extract_context
from alfs.data_models.alf import Alfs, sense_key
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore


def _translate_uuids(labeled: pl.DataFrame, alfs: Alfs) -> pl.DataFrame:
    """Replace UUID sense_keys with positional keys (e.g. "1", "2")."""
    uuid_to_pos: dict[str, str] = {}
    for _form, alf in alfs.entries.items():
        if alf.redirect is not None:
            continue
        for i, sense in enumerate(alf.senses):
            uuid_to_pos[sense.id] = sense_key(i)
    if not uuid_to_pos:
        return labeled
    trans = pl.DataFrame(
        {
            "sense_key": list(uuid_to_pos.keys()),
            "pos_key": list(uuid_to_pos.values()),
        }
    )
    return (
        labeled.join(trans, on="sense_key", how="left")
        .with_columns(pl.coalesce(["pos_key", "sense_key"]).alias("sense_key"))
        .drop("pos_key")
    )


def compile_qc_stats(labeled: pl.DataFrame) -> dict:
    counts = labeled.group_by("rating").agg(pl.len().alias("count")).sort("rating")
    rating_counts = {
        str(row["rating"]): row["count"] for row in counts.iter_rows(named=True)
    }
    for r in ("0", "1", "2"):
        rating_counts.setdefault(r, 0)

    def rating_dist(group_cols: list[str]) -> dict[str, list[int]]:
        all_groups = labeled.select(group_cols).unique()
        dist: dict[str, list[int]] = {}
        for r in (0, 1, 2):
            per_group = (
                labeled.filter(pl.col("rating") == r)
                .group_by(group_cols)
                .agg(pl.len().alias("n"))
            )
            dist[str(r)] = (
                all_groups.join(per_group, on=group_cols, how="left")
                .fill_null(0)["n"]
                .to_list()
            )
        return dist

    return {
        "rating_counts": rating_counts,
        "word_rating_dist": rating_dist(["form"]),
        "sense_rating_dist": rating_dist(["form", "sense_key"]),
    }


def compile_qc_lag(labeled: pl.DataFrame, sense_store: SenseStore) -> dict:
    ts = sense_store.max_sense_updated_at_by_form()
    if not ts:
        return {"pct_stale": None, "stale": 0, "total_with_ts": 0, "lag_days": []}

    ts_df = pl.DataFrame(
        {"form": list(ts.keys()), "sense_updated_at": list(ts.values())}
    )
    merged = labeled.join(ts_df, on="form", how="inner")
    merged = merged.filter(
        pl.col("updated_at").is_not_null() & pl.col("sense_updated_at").is_not_null()
    )
    if merged.is_empty():
        return {"pct_stale": None, "stale": 0, "total_with_ts": 0, "lag_days": []}

    fmt = "%Y-%m-%d %H:%M:%S"
    merged = merged.with_columns(
        (
            (
                pl.col("updated_at").str.to_datetime(fmt)
                - pl.col("sense_updated_at").str.to_datetime(fmt)
            ).dt.total_seconds()
            / 86400.0
        ).alias("lag_days")
    )

    lag_days = merged["lag_days"].to_list()
    stale = sum(1 for d in lag_days if d < 0)
    total = len(lag_days)
    pct_stale = round(100.0 * stale / total, 1) if total > 0 else None
    return {
        "pct_stale": pct_stale,
        "stale": stale,
        "total_with_ts": total,
        "lag_days": lag_days,
    }


def compile_qc_instances(
    labeled: pl.DataFrame, docs: pl.DataFrame, rating: int
) -> dict:
    filtered = labeled.filter(pl.col("rating") == rating)
    if filtered.is_empty():
        return {"rating": rating, "instances": []}

    needed_ids = filtered["doc_id"].unique().to_list()
    docs_subset = docs.filter(pl.col("doc_id").is_in(needed_ids))
    docs_map = dict(
        zip(
            docs_subset["doc_id"].to_list(), docs_subset["text"].to_list(), strict=False
        )
    )

    instances = []
    for row in filtered.iter_rows(named=True):
        text = docs_map.get(row["doc_id"], "")
        if not text:
            continue
        context = _extract_context(
            text, row["byte_offset"], row["form"], 60, bold_form=True
        )
        instances.append(
            {
                "form": row["form"],
                "sense_key": row["sense_key"],
                "context": context,
            }
        )

    return {"rating": rating, "instances": instances}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile QC data for viewer")
    parser.add_argument("--mode", required=True, choices=["stats", "instances", "lag"])
    parser.add_argument("--labeled-db", required=True)
    parser.add_argument(
        "--senses-db", help="Path to senses.db (required for instances mode)"
    )
    parser.add_argument(
        "--docs", help="Path to docs.parquet (required for instances mode)"
    )
    parser.add_argument(
        "--rating", type=int, choices=[0, 1], help="Rating to compile (instances mode)"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    occ_store = OccurrenceStore(Path(args.labeled_db))
    labeled = occ_store.to_polars()

    if args.mode == "stats":
        result = compile_qc_stats(labeled)
    elif args.mode == "lag":
        if args.senses_db is None:
            parser.error("--senses-db is required for lag mode")
        result = compile_qc_lag(labeled, SenseStore(Path(args.senses_db)))
    else:
        if args.docs is None or args.rating is None or args.senses_db is None:
            parser.error(
                "--docs, --senses-db, and --rating are required for instances mode"
            )
        alfs = Alfs(entries=SenseStore(Path(args.senses_db)).all_entries())
        labeled = _translate_uuids(labeled, alfs)
        docs = pl.read_parquet(args.docs)
        result = compile_qc_instances(labeled, docs, args.rating)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    if args.mode == "stats":
        print(f"Wrote QC stats → {args.output}")
    elif args.mode == "lag":
        print(f"Wrote lag data ({result['pct_stale']}% stale) → {args.output}")
    else:
        n = len(result["instances"])
        print(f"Wrote {n} rating-{args.rating} instances → {args.output}")


if __name__ == "__main__":
    main()
