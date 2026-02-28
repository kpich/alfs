"""Select top-N word forms by priority score.

Usage:
    python -m alfs.update.select_targets \\
        --seg-data-dir by_prefix/ --top-n 10 --output-dir targets/ \\
        [--labeled labeled.parquet] [--seed 42]
"""

import argparse
from pathlib import Path
import re
from urllib.parse import quote

import numpy as np
import polars as pl

from alfs.data_models.alf import Alfs
from alfs.data_models.update_target import UpdateTarget

# Only consider forms that contain at least one letter (skip punctuation, whitespace,
# etc.)
_WORD_RE = re.compile(r"[a-zA-Z]")


def select_top_n(
    occurrences_df: pl.DataFrame,  # columns: form (str)
    labeled_df: pl.DataFrame,  # columns: form, doc_id, byte_offset, rating
    top_n: int,
    rng: np.random.Generator,
    redirect_forms: set[str] | frozenset[str] = frozenset(),
) -> list[str]:
    """Return up to top_n forms sorted by priority score, descending.

    Score = binomial(unlabeled, bad_rate) where
    bad_rate = (n_bad + 1) / (n_labeled + 2) with Beta(1,1) prior.
    n_bad counts occurrences with rating in {0, 1} (NONE or POOR sense coverage).
    n_good counts occurrences with rating in {2, 3}.
    """
    total_counts = (
        occurrences_df.select("form").group_by("form").agg(pl.len().alias("total"))
    )

    n_bad = (
        labeled_df.filter(pl.col("rating").is_in([0, 1]))
        .select(["form", "doc_id", "byte_offset"])
        .unique()
        .group_by("form")
        .agg(pl.len().alias("n_bad"))
    )

    n_good = (
        labeled_df.filter(pl.col("rating").is_in([2, 3]))
        .select(["form", "doc_id", "byte_offset"])
        .unique()
        .group_by("form")
        .agg(pl.len().alias("n_good"))
    )

    candidates = (
        total_counts.join(n_bad, on="form", how="left")
        .join(n_good, on="form", how="left")
        .with_columns(
            pl.col("n_bad").fill_null(0),
            pl.col("n_good").fill_null(0),
        )
        .with_columns((pl.col("n_bad") + pl.col("n_good")).alias("n_labeled"))
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
    n_bad_arr = candidates["n_bad"].to_numpy().astype(np.int64)
    n_labeled_arr = candidates["n_labeled"].to_numpy().astype(np.int64)
    unlabeled_arr = candidates["unlabeled"].to_numpy().astype(np.int64)

    bad_rate = (n_bad_arr + 1) / (n_labeled_arr + 2)
    scores = rng.binomial(np.maximum(0, unlabeled_arr), bad_rate)

    order = np.argsort(scores)[::-1]
    sorted_forms = [forms[i] for i in order]

    return sorted_forms[:top_n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select top-N forms by priority score")
    parser.add_argument(
        "--seg-data-dir", required=True, help="Path to by_prefix/ directory"
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--labeled", default=None, help="Path to labeled.parquet")
    parser.add_argument("--alfs", default=None, help="Path to alfs.json")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    args = parser.parse_args()

    seg_dir = Path(args.seg_data_dir)
    parquet_files = list(seg_dir.glob("*/occurrences.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No occurrences.parquet files found in {seg_dir}. " "Run `make seg` first."
        )

    occurrences_df = pl.concat(
        [pl.scan_parquet(str(f)).select("form") for f in parquet_files]
    ).collect()

    if args.labeled and Path(args.labeled).exists():
        labeled_df = pl.read_parquet(args.labeled)
    else:
        labeled_df = pl.DataFrame(
            {"form": [], "doc_id": [], "byte_offset": [], "rating": []},
            schema={
                "form": pl.String,
                "doc_id": pl.String,
                "byte_offset": pl.Int64,
                "rating": pl.Int64,
            },
        )

    redirect_forms: set[str] = set()
    if args.alfs and Path(args.alfs).exists():
        alfs = Alfs.model_validate_json(Path(args.alfs).read_text())
        redirect_forms = {
            f for f, alf in alfs.entries.items() if alf.redirect is not None
        }

    rng = np.random.default_rng(args.seed)
    forms = select_top_n(occurrences_df, labeled_df, args.top_n, rng, redirect_forms)

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
