"""Select top-N word forms by unlabeled occurrence count.

Usage:
    python -m alfs.update.select_targets \\
        --seg-data-dir by_prefix/ --top-n 10 --output-dir targets/ \\
        [--labeled labeled.parquet]
"""

import argparse
from pathlib import Path
import re
from urllib.parse import quote

import polars as pl

from alfs.data_models.update_target import UpdateTarget

# Only consider forms that contain at least one letter (skip punctuation, whitespace,
# etc.)
_WORD_RE = re.compile(r"[a-zA-Z]")


def select_top_n(
    occurrences_df: pl.DataFrame,  # columns: form (str)
    labeled_df: pl.DataFrame,  # columns: form, doc_id, byte_offset, rating
    top_n: int,
) -> list[str]:
    """Return up to top_n forms sorted by unlabeled-occurrence count, descending."""
    total_counts = (
        occurrences_df.select("form").group_by("form").agg(pl.len().alias("total"))
    )

    labeled_counts = (
        labeled_df.filter(pl.col("rating") != 0)  # rating=0 → treat as unlabeled
        .select(["form", "doc_id", "byte_offset"])
        .unique()
        .group_by("form")
        .agg(pl.len().alias("labeled"))
    )

    result = (
        total_counts.join(labeled_counts, on="form", how="left")
        .with_columns(pl.col("labeled").fill_null(0))
        .with_columns((pl.col("total") - pl.col("labeled")).alias("unlabeled"))
        .filter(
            pl.col("form").map_elements(
                lambda f: bool(_WORD_RE.search(f)), return_dtype=pl.Boolean
            )
        )
        .sort("unlabeled", descending=True)
        .head(top_n)
    )

    return result["form"].to_list()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select top-N forms by unlabeled occurrence count"
    )
    parser.add_argument(
        "--seg-data-dir", required=True, help="Path to by_prefix/ directory"
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--labeled", default=None, help="Path to labeled.parquet")
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

    forms = select_top_n(occurrences_df, labeled_df, args.top_n)

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
