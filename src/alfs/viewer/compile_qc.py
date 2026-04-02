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
from alfs.data_models.blocklist import Blocklist
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore


def _translate_uuids(labeled: pl.DataFrame, alfs: Alfs) -> pl.DataFrame:
    """Replace UUID sense_keys with positional keys (e.g. "1", "2")."""
    uuid_to_pos: dict[str, str] = {}
    for _form, alf in alfs.entries.items():
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


def compile_qc_coverage(
    labeled: pl.DataFrame,
    alfs: Alfs,
    corpus_counts: dict[str, int],
    blocklist_forms: set[str],
    n_buckets: int = 20,
    min_bucket_count: int = 50,
    top_corpus_forms: dict[str, int] | None = None,
) -> dict:
    import math

    # Case-insensitive: a form is covered if any case variant has at least one sense.
    has_def: dict[str, bool] = {}
    for form, alf in alfs.entries.items():
        lower = form.lower()
        if bool(alf.senses):
            has_def[lower] = True
        elif lower not in has_def:
            has_def[lower] = False

    # Aggregate labeled counts by lowercase form so "The" and "the" share stats.
    counts_df = (
        labeled.with_columns(pl.col("form").str.to_lowercase().alias("form_lower"))
        .group_by("form_lower")
        .agg(
            pl.len().alias("n_labeled"),
            (pl.col("rating") == 2).sum().alias("n_excellent"),
        )
    )
    n_labeled: dict[str, int] = dict(
        zip(
            counts_df["form_lower"].to_list(),
            counts_df["n_labeled"].to_list(),
            strict=False,
        )
    )
    n_excellent: dict[str, int] = dict(
        zip(
            counts_df["form_lower"].to_list(),
            counts_df["n_excellent"].to_list(),
            strict=False,
        )
    )

    total_labeled = labeled.height
    total_excellent = int((labeled["rating"] == 2).sum())
    global_excellent_rate = (
        total_excellent / total_labeled if total_labeled > 0 else 0.0
    )

    # Total corpus tokens: use _total if present (all corpus), else fall back to tracked
    # sum
    total_corpus = corpus_counts.get(
        "_total", sum(v for k, v in corpus_counts.items() if not k.startswith("_"))
    )

    covered_corpus = sum(
        v
        for k, v in corpus_counts.items()
        if not k.startswith("_") and has_def.get(k.lower(), False)
    )
    pct_instances_covered = (
        covered_corpus / total_corpus * 100.0 if total_corpus > 0 else 0.0
    )

    K = 5
    smoothed_num = 0.0
    for form, count in corpus_counts.items():
        if form.startswith("_"):
            continue
        if not has_def.get(form.lower(), False):
            continue
        nl = n_labeled.get(form.lower(), 0)
        ne = n_excellent.get(form.lower(), 0)
        smoothed_rate = (ne + K * global_excellent_rate) / (nl + K)
        smoothed_num += smoothed_rate * count
    pct_senses_covered_est = (
        smoothed_num / total_corpus * 100.0 if total_corpus > 0 else 0.0
    )

    # For the chart: use top_corpus_forms (all-corpus forms) if available,
    # else fall back to senses.db-filtered corpus_counts.
    # top_corpus_forms includes untracked words so coverage gaps are visible.
    chart_source = (
        top_corpus_forms
        if top_corpus_forms is not None
        else {k: v for k, v in corpus_counts.items() if not k.startswith("_")}
    )
    sorted_forms_all = sorted(
        [(f, c) for f, c in chart_source.items() if any(ch.isalnum() for ch in f)],
        key=lambda x: x[1],
        reverse=True,
    )

    N = len(sorted_forms_all)
    bucket_size = max(1, math.ceil(N / n_buckets))

    bucket_counts_covered: list[int] = []
    bucket_counts_uncovered: list[int] = []
    for i in range(n_buckets):
        chunk = sorted_forms_all[i * bucket_size : (i + 1) * bucket_size]
        if not chunk:
            break
        cov = sum(c for f, c in chunk if has_def.get(f.lower(), False))
        uncov = sum(c for f, c in chunk if not has_def.get(f.lower(), False))
        if cov + uncov < min_bucket_count:
            break  # right-truncate: drop the sparse tail
        bucket_counts_covered.append(cov)
        bucket_counts_uncovered.append(uncov)

    n_shown = len(bucket_counts_covered)

    # Search full sorted list for first uncovered rank
    first_uncovered_rank: int | None = None
    first_uncovered_form: str | None = None
    for rank, (form, _) in enumerate(sorted_forms_all, start=1):
        lower = form.lower()
        if (
            not has_def.get(lower, False)
            and lower not in blocklist_forms
            and form not in blocklist_forms
        ):
            first_uncovered_rank = rank
            first_uncovered_form = form
            break

    # x position for vertical line: only set if rank falls within the shown buckets
    first_uncovered_bucket_x: float | None = None
    if first_uncovered_rank is not None:
        b = (first_uncovered_rank - 1) // bucket_size  # 0-based bucket index
        if b < n_shown:
            first_uncovered_bucket_x = b + 0.5  # left edge of bar at x = b+1

    return {
        "pct_instances_covered": round(pct_instances_covered, 2),
        "pct_senses_covered_est": round(pct_senses_covered_est, 2),
        "first_uncovered_rank": first_uncovered_rank,
        "first_uncovered_form": first_uncovered_form,
        "bucket_counts_covered": bucket_counts_covered,
        "bucket_counts_uncovered": bucket_counts_uncovered,
        "first_uncovered_bucket_x": first_uncovered_bucket_x,
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
    parser.add_argument(
        "--mode", required=True, choices=["stats", "instances", "lag", "coverage"]
    )
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
    parser.add_argument(
        "--corpus-counts",
        help="Path to corpus_counts.json (required for coverage mode)",
    )
    parser.add_argument(
        "--top-corpus-forms",
        help="Path to top_corpus_forms.json (optional, for coverage chart)",
    )
    parser.add_argument(
        "--blocklist", help="Path to blocklist.yaml (optional, for coverage mode)"
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
    elif args.mode == "coverage":
        if args.corpus_counts is None or args.senses_db is None:
            parser.error(
                "--corpus-counts and --senses-db are required for coverage mode"
            )
        corpus_counts: dict[str, int] = json.loads(Path(args.corpus_counts).read_text())
        top_corpus_forms: dict[str, int] | None = (
            json.loads(Path(args.top_corpus_forms).read_text())
            if args.top_corpus_forms
            else None
        )
        alfs = Alfs(entries=SenseStore(Path(args.senses_db)).all_entries())
        blocklist_forms: set[str] = (
            set(Blocklist(Path(args.blocklist)).load().keys())
            if args.blocklist
            else set()
        )
        result = compile_qc_coverage(
            labeled,
            alfs,
            corpus_counts,
            blocklist_forms,
            top_corpus_forms=top_corpus_forms,
        )
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
    elif args.mode == "coverage":
        print(
            f"Coverage: {result['pct_instances_covered']}% instances, "
            f"{result['pct_senses_covered_est']}% senses (est) → {args.output}"
        )
    else:
        n = len(result["instances"])
        print(f"Wrote {n} rating-{args.rating} instances → {args.output}")


if __name__ == "__main__":
    main()
