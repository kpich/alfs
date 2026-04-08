"""Scatter: per-form sense divergence across sources vs. source skew of form usage.

For each form with labeled occurrences in ≥2 sources:
  x = source skew of the form's labeled instances
      (1 - normalised entropy; 0 = perfectly even, 1 = all in one source)
  y = mean pairwise Jensen-Shannon divergence between per-source sense
      distributions for that form

A strong x→y correlation means sense divergence mirrors form-distribution
skew; scatter / weak correlation means senses vary for reasons beyond domain
vocabulary skew.

Usage:
    python -m alfs.plots.source_sense_divergence \\
        --senses-db senses.db \\
        --labeled-db labeled.db \\
        --docs docs.parquet \\
        --output source_sense_divergence.png \\
        [--min-per-source 5]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.spatial.distance import jensenshannon
from scipy.stats import linregress

from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base-2) between two unnormalised count vectors."""
    p = p / p.sum()
    q = q / q.sum()
    return float(jensenshannon(p, q, base=2))


def _source_skew(counts: list[int]) -> float:
    """1 - normalised Shannon entropy of source counts.

    Returns 0 when counts are perfectly even, 1 when all mass is in one source.
    Returns 0 for a single-source form (treated as maximally concentrated but
    those are filtered out before this is called).
    """
    arr = np.array(counts, dtype=float)
    arr = arr / arr.sum()
    n = len(arr)
    if n <= 1:
        return 1.0
    h = -np.sum(arr * np.log(arr + 1e-12))
    return float(1.0 - h / np.log(n))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--senses-db", required=True)
    parser.add_argument("--labeled-db", required=True)
    parser.add_argument("--docs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--min-per-source",
        type=int,
        default=5,
        help="Minimum labeled instances per source for a form to be included",
    )
    args = parser.parse_args()

    # --- load data ---
    labeled = OccurrenceStore(Path(args.labeled_db)).to_polars()
    docs = pl.read_parquet(args.docs, columns=["doc_id", "source"]).drop_nulls("source")

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

    xs, ys, ns = [], [], []

    for form in eligible_forms:
        form_totals = totals_filtered.filter(pl.col("form") == form).select(
            ["source", "total"]
        )
        # only sources meeting threshold for this form
        qualified_sources = form_totals.filter(pl.col("total") >= args.min_per_source)[
            "source"
        ].to_list()
        if len(qualified_sources) < 2:
            continue

        # source skew using all qualified sources
        counts = (
            form_totals.filter(pl.col("source").is_in(qualified_sources))
            .sort("source")["total"]
            .to_list()
        )
        x = _source_skew(counts)

        # sense distributions per source
        form_senses = agg_filtered.filter(
            (pl.col("form") == form) & (pl.col("source").is_in(qualified_sources))
        )
        sense_keys = sorted(form_senses["sense_key"].unique().to_list())

        def make_sense_vec(fs: pl.DataFrame, sk: list[str]):
            def _vec(src: str) -> np.ndarray:
                rows = fs.filter(pl.col("source") == src)
                vec = np.zeros(len(sk))
                for row in rows.iter_rows(named=True):
                    vec[sk.index(row["sense_key"])] += row["cnt"]
                # add small smoothing so JSD is defined even for unseen senses
                return vec + 0.1

            return _vec

        sense_vec = make_sense_vec(form_senses, sense_keys)
        jsds = []
        for i, src_a in enumerate(qualified_sources):
            for src_b in qualified_sources[i + 1 :]:
                jsds.append(_jsd(sense_vec(src_a), sense_vec(src_b)))

        xs.append(x)
        ys.append(float(np.mean(jsds)))
        ns.append(n_senses_map.get(form, 1))

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    ns_arr = np.array(ns)

    slope, intercept, r_value, p_value, _ = linregress(xs_arr, ys_arr)
    r2 = r_value**2
    p_str = f"{p_value:.2e}" if p_value >= 1e-4 else "p<0.0001"

    fig, ax = plt.subplots(figsize=(8, 5))

    sc = ax.scatter(
        xs_arr,
        ys_arr,
        c=np.log1p(ns_arr),
        cmap="viridis",
        s=14,
        alpha=0.55,
        linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log(1 + n_senses)")

    x_line = np.linspace(xs_arr.min(), xs_arr.max(), 200)
    ax.plot(x_line, slope * x_line + intercept, color="crimson", linewidth=1.5)

    stats_text = (
        f"slope={slope:.3f}  r²={r2:.3f}  p={p_str}  n={len(xs_arr):,}\n"
        f"min_per_source={args.min_per_source}  sources={sources}"
    )
    ax.text(
        0.02,
        0.97,
        stats_text,
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
    )

    ax.set_xlabel("source skew of form  (0 = even, 1 = concentrated in one source)")
    ax.set_ylabel("mean pairwise JSD of sense distributions across sources")
    ax.set_title("Per-form: sense divergence across sources vs. source skew")
    fig.tight_layout()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}  ({len(xs_arr):,} forms, sources: {sources})")


if __name__ == "__main__":
    main()
