"""Flask viewer for ETL corpus data (docs.parquet).

Usage:
    python -m alfs.dataviewer.app --docs path/to/docs.parquet
"""

import argparse
import math
from pathlib import Path

from flask import Flask, abort, render_template, request
import polars as pl

PAGE_SIZE = 50

app = Flask(__name__)

_df: pl.DataFrame | None = None


def get_df() -> pl.DataFrame:
    global _df
    assert _df is not None
    return _df


@app.route("/")
def index():
    df = get_df()
    source_filter = request.args.get("source", "").strip()
    page = request.args.get("page", 1, type=int)

    filtered = df.filter(pl.col("source") == source_filter) if source_filter else df

    total = len(filtered)
    total_pages = max(1, math.ceil(total / PAGE_SIZE))
    page = max(1, min(page, total_pages))
    start = (page - 1) * PAGE_SIZE
    rows = filtered.slice(start, PAGE_SIZE).to_dicts()

    # Summary stats
    sources = (
        df.group_by("source").agg(pl.len().alias("count")).sort("source").to_dicts()
    )
    null_counts = {
        col: df[col].null_count() for col in ["year", "title", "author", "source_url"]
    }

    all_sources = sorted(df["source"].drop_nulls().unique().to_list())

    return render_template(
        "index.html",
        rows=rows,
        page=page,
        total_pages=total_pages,
        total=total,
        total_docs=len(df),
        sources=sources,
        null_counts=null_counts,
        all_sources=all_sources,
        source_filter=source_filter,
    )


@app.route("/doc/<doc_id>")
def doc(doc_id: str):
    df = get_df()
    matches = df.filter(pl.col("doc_id") == doc_id)
    if len(matches) == 0:
        abort(404)
    row = matches.to_dicts()[0]
    return render_template("doc.html", row=row)


def main() -> None:
    global _df
    parser = argparse.ArgumentParser(description="ETL data viewer")
    parser.add_argument("--docs", required=True, type=Path, help="Path to docs.parquet")
    args = parser.parse_args()
    _df = pl.read_parquet(args.docs)
    app.run(host="localhost", port=5003, debug=False)


if __name__ == "__main__":
    main()
