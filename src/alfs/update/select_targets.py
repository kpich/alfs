"""Select top-N word forms by unlabeled occurrence count.

Usage:
    python -m alfs.update.select_targets \\
        --seg-data-dir by_prefix/ --top-n 10 --output-dir targets/ \
        [--labeled-dir update_data/]
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select top-N forms by unlabeled occurrence count"
    )
    parser.add_argument(
        "--seg-data-dir", required=True, help="Path to by_prefix/ directory"
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--labeled-dir", default=None, help="Directory with *_labeled.parquet files"
    )
    args = parser.parse_args()

    seg_dir = Path(args.seg_data_dir)
    parquet_files = list(seg_dir.glob("*/occurrences.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No occurrences.parquet files found in {seg_dir}")

    total_counts = (
        pl.concat([pl.scan_parquet(str(f)).select("form") for f in parquet_files])
        .collect()
        .group_by("form")
        .agg(pl.len().alias("total"))
    )

    if args.labeled_dir:
        labeled_files = list(Path(args.labeled_dir).glob("**/*_labeled.parquet"))
        if labeled_files:
            labeled_counts = (
                pl.concat([pl.read_parquet(str(f)) for f in labeled_files])
                .group_by("form")
                .agg(pl.len().alias("labeled"))
            )
        else:
            labeled_counts = pl.DataFrame(
                {"form": [], "labeled": []},
                schema={"form": pl.String, "labeled": pl.UInt32},
            )
    else:
        labeled_counts = pl.DataFrame(
            {"form": [], "labeled": []},
            schema={"form": pl.String, "labeled": pl.UInt32},
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
        .head(args.top_n)
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in result.iter_rows(named=True):
        form = row["form"]
        target = UpdateTarget(form=form)
        safe = quote(form, safe="")
        out_path = out_dir / f"{safe}.json"
        out_path.write_text(target.model_dump_json())
        print(f"  {form}: {row['unlabeled']} unlabeled occurrences")

    print(f"Wrote {len(result)} targets to {out_dir}")


if __name__ == "__main__":
    main()
