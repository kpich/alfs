"""Pairwise conditional sense JSD between sources.

For each source pair, computes the weighted mean of
JSD(sense dist | form, src_A || sense dist | form, src_B)
over forms present in both sources with sufficient labeled instances.

This answers: for words that appear in multiple sources, are they being
disambiguated the same way? Low = senses are stable across sources.
High = sense labels are source-contaminated.

Usage:
    python -m alfs.plots.source_distance_matrices \\
        --labeled-db labeled.db \\
        --docs docs.parquet \\
        --output source_distance_matrices.png \\
        [--min-per-source 3]
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.spatial.distance import jensenshannon

from alfs.data_models.occurrence_store import OccurrenceStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-db", required=True)
    parser.add_argument("--docs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-per-source", type=int, default=3)
    args = parser.parse_args()

    labeled = OccurrenceStore(Path(args.labeled_db)).to_polars()
    docs = pl.read_parquet(args.docs, columns=["doc_id", "source"]).drop_nulls("source")
    df = labeled.join(docs, on="doc_id", how="inner").select(
        ["form", "source", "sense_key"]
    )

    sources = sorted(df["source"].unique().to_list())

    form_sense_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in (
        df.group_by(["form", "source", "sense_key"])
        .agg(pl.len().alias("cnt"))
        .iter_rows(named=True)
    ):
        form_sense_counts[row["form"]][row["source"]][row["sense_key"]] = row["cnt"]

    n = len(sources)
    mat = np.zeros((n, n))
    n_forms_mat = np.zeros((n, n), dtype=int)
    for i, src_a in enumerate(sources):
        for j, src_b in enumerate(sources):
            if i >= j:
                continue
            jsds, weights = [], []
            for _form, src_dict in form_sense_counts.items():
                if src_a not in src_dict or src_b not in src_dict:
                    continue
                ca, cb = src_dict[src_a], src_dict[src_b]
                if (
                    sum(ca.values()) < args.min_per_source
                    or sum(cb.values()) < args.min_per_source
                ):
                    continue
                senses = sorted(set(ca) | set(cb))
                va = np.array([ca.get(s, 0) for s in senses], dtype=float) + 0.1
                vb = np.array([cb.get(s, 0) for s in senses], dtype=float) + 0.1
                jsds.append(float(jensenshannon(va / va.sum(), vb / vb.sum(), base=2)))
                weights.append((sum(ca.values()) + sum(cb.values())) / 2)
            if jsds:
                v = float(np.average(jsds, weights=weights))
                mat[i, j] = mat[j, i] = v
                n_forms_mat[i, j] = n_forms_mat[j, i] = len(jsds)

    fig, ax = plt.subplots(figsize=(5, 4))
    display = np.where(np.eye(n, dtype=bool), np.nan, mat)
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(alpha=0)
    im = ax.imshow(display, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(sources, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(sources, fontsize=9)
    ax.set_title(
        "Conditional sense JSD across sources\n(shared forms only)", fontsize=10
    )

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ax.text(
                j,
                i,
                f"{mat[i,j]:.3f}\n({n_forms_mat[i,j]} forms)",
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.colorbar(im, ax=ax, shrink=0.7, label="JSD (bits)")
    fig.tight_layout()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}  (sources: {sources})")
    for i, src in enumerate(sources):
        print(f"  {src}: {[round(v, 4) for v in mat[i].tolist()]}")


if __name__ == "__main__":
    main()
