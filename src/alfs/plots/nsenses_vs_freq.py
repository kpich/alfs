"""Scatter plot: number of senses vs. log corpus frequency.

Usage:
    python -m alfs.plots.nsenses_vs_freq \
        --senses-db senses.db \
        --corpus-counts corpus_counts.json \
        --output nsenses_vs_freq.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress  # type: ignore[import-untyped]

from alfs.data_models.sense_store import SenseStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--senses-db", required=True)
    parser.add_argument("--corpus-counts", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    entries = SenseStore(Path(args.senses_db)).all_entries()
    corpus_counts: dict[str, int] = json.loads(Path(args.corpus_counts).read_text())

    forms = [
        form
        for form, alf in entries.items()
        if alf.spelling_variant_of is None
        and len(alf.senses) >= 1
        and corpus_counts.get(form, 0) > 0
    ]
    x = np.array([np.log10(corpus_counts[f]) for f in forms])
    y = np.array([len(entries[f].senses) for f in forms])

    slope, intercept, r_value, p_value, _ = linregress(x, y)
    r2 = r_value**2
    p_str = f"{p_value:.2e}" if p_value >= 1e-4 else "p<0.0001"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, s=8, alpha=0.35, linewidths=0)

    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, slope * x_line + intercept, color="crimson", linewidth=1.5)

    stats_text = f"slope={slope:.3f}  r²={r2:.3f}  p={p_str}  n={len(forms):,}"
    ax.text(
        0.02,
        0.97,
        stats_text,
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
    )

    ax.set_xlabel("log₁₀(corpus count)")
    ax.set_ylabel("number of senses")
    ax.set_title("Number of Senses vs. Log Corpus Frequency")
    fig.tight_layout()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
