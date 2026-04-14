"""Scatter: per-form sense divergence across sources vs. source skew of form usage.

For each form with labeled occurrences in ≥2 sources:
  x = max source proportion: fraction of the form's labeled instances from
      the source that uses it most (range [1/n_sources, 1])
  y = mean pairwise Jensen-Shannon divergence between per-source sense
      distributions for that form

The significance test is a two-tailed t-test on the OLS slope (H0: slope=0),
i.e. no linear relationship between source skew and sense divergence.

Usage:
    python -m alfs.plots.source_sense_divergence \\
        --senses-db senses.db \\
        --labeled-db labeled.db \\
        --docs docs.parquet \\
        --corpus-counts corpus_counts.json \\
        --output source_sense_divergence.png \\
        [--min-per-source 5]
"""

import argparse
import json
from pathlib import Path

from adjustText import adjust_text  # type: ignore[import-untyped]
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.spatial.distance import jensenshannon  # type: ignore[import-untyped]
from scipy.stats import linregress, spearmanr  # type: ignore[import-untyped]

from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base-2) between two unnormalised count vectors."""
    p = p / p.sum()
    q = q / q.sum()
    return float(jensenshannon(p, q, base=2))


def _max_source_proportion(counts: list[int]) -> float:
    """Fraction of instances in the dominant source (max proportion)."""
    arr = np.array(counts, dtype=float)
    return float(arr.max() / arr.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--senses-db", required=True)
    parser.add_argument("--labeled-db", required=True)
    parser.add_argument("--docs", required=True)
    parser.add_argument("--corpus-counts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--min-per-source",
        type=int,
        default=5,
        help="Minimum labeled instances per source for a form to be included",
    )
    parser.add_argument(
        "--min-sense-count",
        type=int,
        default=5,
        help="Min total labeled instances across all sources for a sense to be "
        "included",
    )
    args = parser.parse_args()

    # --- load data ---
    labeled = OccurrenceStore(Path(args.labeled_db)).to_polars()
    docs = pl.read_parquet(args.docs, columns=["doc_id", "source"]).drop_nulls("source")
    corpus_counts: dict[str, int] = {
        k: v
        for k, v in json.loads(Path(args.corpus_counts).read_text()).items()
        if k != "_total"
    }

    # join to get source per occurrence
    df = labeled.join(docs, on="doc_id", how="inner").select(
        ["form", "source", "sense_key"]
    )

    # n_senses per form from senses.db (for colouring)
    entries = SenseStore(Path(args.senses_db)).all_entries()
    n_senses_map = {
        form: len(alf.senses)
        for form, alf in entries.items()
        if alf.spelling_variant_of is None
    }

    sources = sorted(df["source"].unique().to_list())

    # per-form, per-source sense counts
    agg = df.group_by(["form", "source", "sense_key"]).agg(pl.len().alias("cnt"))

    # total labeled instances per (form, source)
    totals = agg.group_by(["form", "source"]).agg(pl.col("cnt").sum().alias("total"))

    # keep only forms that have >= min_per_source instances in >= 2 sources
    qualified_per_source = totals.filter(pl.col("total") >= args.min_per_source)
    source_counts_per_form = qualified_per_source.group_by("form").agg(
        pl.len().alias("n_qualified_sources")
    )
    eligible_forms = source_counts_per_form.filter(pl.col("n_qualified_sources") >= 2)[
        "form"
    ].to_list()

    if not eligible_forms:
        print(
            "No forms meet the min-per-source threshold; try lowering --min-per-source"
        )
        return

    agg_filtered = agg.filter(pl.col("form").is_in(eligible_forms))
    totals_filtered = totals.filter(pl.col("form").is_in(eligible_forms))

    forms_out: list[str] = []
    xs, ys, ns = [], [], []

    for form in eligible_forms:
        form_totals = totals_filtered.filter(pl.col("form") == form).select(
            ["source", "total"]
        )
        qualified_sources = form_totals.filter(pl.col("total") >= args.min_per_source)[
            "source"
        ].to_list()
        if len(qualified_sources) < 2:
            continue

        counts = (
            form_totals.filter(pl.col("source").is_in(qualified_sources))
            .sort("source")["total"]
            .to_list()
        )
        x = _max_source_proportion(counts)

        form_senses = agg_filtered.filter(
            (pl.col("form") == form) & (pl.col("source").is_in(qualified_sources))
        )
        # drop senses with too few total instances (sparse → noisy JSD)
        sense_totals = (
            form_senses.group_by("sense_key")
            .agg(pl.col("cnt").sum().alias("total_cnt"))
            .filter(pl.col("total_cnt") >= args.min_sense_count)
        )
        sense_keys = sorted(sense_totals["sense_key"].to_list())
        if len(sense_keys) < 2:
            continue
        form_senses = form_senses.filter(pl.col("sense_key").is_in(sense_keys))

        def make_sense_vec(fs: pl.DataFrame, sk: list[str]):
            def _vec(src: str) -> np.ndarray:
                rows = fs.filter(pl.col("source") == src)
                vec = np.zeros(len(sk))
                for row in rows.iter_rows(named=True):
                    vec[sk.index(row["sense_key"])] += row["cnt"]
                return vec + 0.1

            return _vec

        sense_vec = make_sense_vec(form_senses, sense_keys)
        jsds = []
        for i, src_a in enumerate(qualified_sources):
            for src_b in qualified_sources[i + 1 :]:
                jsds.append(_jsd(sense_vec(src_a), sense_vec(src_b)))

        forms_out.append(form)
        xs.append(x)
        ys.append(float(np.mean(jsds)))
        ns.append(n_senses_map.get(form, 1))

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    ns_arr = np.array(ns)

    # --- highlight forms: common (top 20% by corpus freq), low X, high Y ---
    freq_threshold = np.percentile(
        [corpus_counts[f] for f in forms_out if f in corpus_counts], 80
    )
    x_median = np.median(xs_arr)
    candidates = [
        (i, f, ys[i])
        for i, f in enumerate(forms_out)
        if corpus_counts.get(f, 0) >= freq_threshold and xs[i] <= x_median
    ]
    candidates.sort(key=lambda t: t[2], reverse=True)
    highlight_idxs = [t[0] for t in candidates[:5]]
    highlight_forms = [t[1] for t in candidates[:5]]

    # --- stats ---
    slope, intercept, _, _, _ = linregress(xs_arr, ys_arr)
    rho, sp_pvalue = spearmanr(xs_arr, ys_arr)
    p_label = f"p={sp_pvalue:.2e}" if sp_pvalue >= 1e-4 else "p<0.0001"

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    sc = ax.scatter(
        xs_arr,
        ys_arr,
        c=ns_arr,
        cmap="Blues",
        norm=LogNorm(vmin=1),
        s=14,
        alpha=0.65,
        linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=15)
    cbar.set_label("Num senses", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # highlighted points — same colour scale, black outline
    ax.scatter(
        xs_arr[highlight_idxs],
        ys_arr[highlight_idxs],
        c=ns_arr[highlight_idxs],
        cmap="Blues",
        norm=LogNorm(vmin=1),
        s=40,
        alpha=0.9,
        linewidths=0.8,
        edgecolors="black",
        zorder=5,
    )

    texts = [
        ax.text(xs_arr[i], ys_arr[i], f, fontsize=7)
        for i, f in zip(highlight_idxs, highlight_forms, strict=False)
    ]
    adjust_text(
        texts,
        ax=ax,
        arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.7},
    )

    x_line = np.linspace(xs_arr.min(), xs_arr.max(), 200)
    ax.plot(x_line, slope * x_line + intercept, color="crimson", linewidth=1.5)

    stats_text = f"slope={slope:.3f}\nSpearman ρ={rho:.3f}\n{p_label}"
    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
    )

    ax.set_xlabel(
        "max source proportion  (fraction of labeled instances from dominant source)"
    )
    ax.set_ylabel("mean pairwise JSD of sense distributions across sources")
    ax.set_title("Per-form: sense divergence across sources vs. source skew")
    fig.tight_layout()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(
        f"Wrote {args.output}  ({len(xs_arr):,} forms, sources: {sources})\n"
        f"Highlighted: {highlight_forms}"
    )


if __name__ == "__main__":
    main()
