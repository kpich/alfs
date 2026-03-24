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
    total_counts: pl.DataFrame,
    n_covered_counts: pl.DataFrame,
    top_n: int,
    rng: np.random.Generator,
    min_count: int,
    redirect_forms: set[str] | frozenset[str] = frozenset(),
    smoothing_alpha: float = 0.0,
) -> list[str]:
    """Return up to top_n forms sampled proportionally to sqrt(need_work count).

    Weights are sqrt-transformed to compress Zipfian dynamic range while still
    oversampling the head: a form with 10k needy occurrences is ~10× more
    likely than one with 100, not 100×.
    smoothing_alpha adds a pseudocount so no form has weight 0 (Laplace smoothing).
    """
    candidates = (
        total_counts.join(n_covered_counts, on="form", how="left")
        .with_columns(pl.col("n_covered").fill_null(0))
        .with_columns(
            (pl.col("total") - pl.col("n_covered") + smoothing_alpha).alias("need_work")
        )
        .filter(pl.col("total") >= min_count)
        .filter(
            pl.col("form").map_elements(
                lambda f: bool(_WORD_RE.search(f)), return_dtype=pl.Boolean
            )
        )
    )

    if redirect_forms:
        candidates = candidates.filter(~pl.col("form").is_in(list(redirect_forms)))

    forms = candidates["form"].to_list()
    need_work_arr = candidates["need_work"].to_numpy().astype(np.float64)
    weights = np.sqrt(np.maximum(0.0, need_work_arr))
    total_weight = weights.sum()
    if total_weight == 0:
        return []
    probs = weights / total_weight
    n = min(top_n, int((weights > 0).sum()))
    chosen_indices = rng.choice(len(forms), size=n, replace=False, p=probs)
    return [forms[i] for i in chosen_indices]


def run(
    seg_data_dir: str | Path,
    top_n: int,
    output_dir: str | Path,
    senses_db: str | Path | None = None,
    labeled_db: str | Path | None = None,
    seed: int | None = None,
    min_count: int = 5,
    smoothing_alpha: float = 0.0,
    use_excellent_threshold: bool = False,
) -> list[Path]:
    seg_dir = Path(seg_data_dir)
    parquet_files = list(seg_dir.glob("*/occurrences.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No occurrences.parquet files found in {seg_dir}. Run `make seg` first."
        )

    total_counts = (
        pl.concat([pl.scan_parquet(str(f)).select("form") for f in parquet_files])
        .group_by("form")
        .agg(pl.len().alias("total"))
        .collect()
    )

    if labeled_db and Path(labeled_db).exists():
        cbf = OccurrenceStore(Path(labeled_db)).count_by_form()
        col = "n_excellent" if use_excellent_threshold else "n_total"
        n_covered_counts = cbf.rename({col: "n_covered"}).select(["form", "n_covered"])
    else:
        n_covered_counts = pl.DataFrame(
            {"form": [], "n_covered": []},
            schema={"form": pl.String, "n_covered": pl.Int64},
        )

    redirect_forms: set[str] = set()
    if senses_db and Path(senses_db).exists():
        store = SenseStore(Path(senses_db))
        entries = store.all_entries()
        redirect_forms = {f for f, alf in entries.items() if alf.redirect is not None}

    rng = np.random.default_rng(seed)
    forms = select_top_n(
        total_counts,
        n_covered_counts,
        top_n,
        rng,
        min_count,
        redirect_forms,
        smoothing_alpha=smoothing_alpha,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result: list[Path] = []
    for form in forms:
        target = UpdateTarget(form=form)
        safe = quote(form, safe="")
        out_path = out_dir / f"{safe}.json"
        out_path.write_text(target.model_dump_json())
        print(f"  {form}")
        result.append(out_path)

    print(f"Wrote {len(forms)} targets to {out_dir}")
    return result


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
    parser.add_argument(
        "--min-count", type=int, default=5, help="Minimum raw corpus occurrence count"
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.0,
        help="Laplace smoothing pseudocount",
    )
    parser.add_argument(
        "--use-excellent-threshold",
        action="store_true",
        default=False,
        help="Only count rating=3 occurrences as fully served (for induction)",
    )
    args = parser.parse_args()

    run(
        args.seg_data_dir,
        args.top_n,
        args.output_dir,
        args.senses_db,
        args.labeled_db,
        args.seed,
        min_count=args.min_count,
        smoothing_alpha=args.smoothing_alpha,
        use_excellent_threshold=args.use_excellent_threshold,
    )


if __name__ == "__main__":
    main()
