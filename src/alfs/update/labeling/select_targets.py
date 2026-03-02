"""Select top-N word forms by priority score.

Usage:
    python -m alfs.update.labeling.select_targets \\
        --seg-data-dir by_prefix/ --top-n 10 --output-dir targets/ \\
        [--senses-db senses.db] [--labeled-db labeled.db] [--seed 42]
"""

import argparse
from pathlib import Path
import re
from urllib.parse import quote

import numpy as np
import polars as pl

from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore
from alfs.data_models.update_target import UpdateTarget

# Only consider forms that contain at least one letter (skip punctuation, whitespace,
# etc.)
_WORD_RE = re.compile(r"[a-zA-Z]")


def select_top_n(
    total_counts: pl.DataFrame,  # columns: form (str), total (int)
    n_labeled_counts: pl.DataFrame,  # columns: form (str), n_labeled (int)
    top_n: int,
    rng: np.random.Generator,
    redirect_forms: set[str] | frozenset[str] = frozenset(),
) -> list[str]:
    """Return up to top_n forms sampled proportionally to unlabeled instance count.

    Forms with more unlabeled instances have a higher probability of being selected.
    Forms with zero unlabeled instances are excluded.
    """
    candidates = (
        total_counts.join(n_labeled_counts, on="form", how="left")
        .with_columns(pl.col("n_labeled").fill_null(0))
        .with_columns((pl.col("total") - pl.col("n_labeled")).alias("unlabeled"))
        .filter(
            pl.col("form").map_elements(
                lambda f: bool(_WORD_RE.search(f)), return_dtype=pl.Boolean
            )
        )
    )

    if redirect_forms:
        candidates = candidates.filter(~pl.col("form").is_in(list(redirect_forms)))

    forms = candidates["form"].to_list()
    unlabeled_arr = candidates["unlabeled"].to_numpy().astype(np.float64)
    weights = np.maximum(0.0, unlabeled_arr)
    total_weight = weights.sum()
    if total_weight == 0:
        return []
    probs = weights / total_weight
    n = min(top_n, int((weights > 0).sum()))
    chosen_indices = rng.choice(len(forms), size=n, replace=False, p=probs)
    return [forms[i] for i in chosen_indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select top-N forms by priority score")
    parser.add_argument(
        "--seg-data-dir", required=True, help="Path to by_prefix/ directory"
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--senses-db", default=None, help="Path to senses.db")
    parser.add_argument("--labeled-db", default=None, help="Path to labeled.db")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    args = parser.parse_args()

    seg_dir = Path(args.seg_data_dir)
    parquet_files = list(seg_dir.glob("*/occurrences.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No occurrences.parquet files found in {seg_dir}. " "Run `make seg` first."
        )

    total_counts = (
        pl.concat([pl.scan_parquet(str(f)).select("form") for f in parquet_files])
        .group_by("form")
        .agg(pl.len().alias("total"))
        .collect()
    )

    if args.labeled_db and Path(args.labeled_db).exists():
        n_labeled_counts = (
            OccurrenceStore(Path(args.labeled_db))
            .count_by_form()
            .rename({"n_total": "n_labeled"})
            .select(["form", "n_labeled"])
        )
    else:
        n_labeled_counts = pl.DataFrame(
            {"form": [], "n_labeled": []},
            schema={"form": pl.String, "n_labeled": pl.Int64},
        )

    redirect_forms: set[str] = set()
    if args.senses_db and Path(args.senses_db).exists():
        store = SenseStore(Path(args.senses_db))
        entries = store.all_entries()
        redirect_forms = {f for f, alf in entries.items() if alf.redirect is not None}

    rng = np.random.default_rng(args.seed)
    forms = select_top_n(
        total_counts, n_labeled_counts, args.top_n, rng, redirect_forms
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for form in forms:
        target = UpdateTarget(form=form)
        safe = quote(form, safe="")
        out_path = out_dir / f"{safe}.json"
        out_path.write_text(target.model_dump_json())
        print(f"  {form}")

    print(f"Wrote {len(forms)} targets to {out_dir}")


if __name__ == "__main__":
    main()
