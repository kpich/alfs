"""Generate one UpdateTarget JSON file per entry in alfs.json.

Usage:
    python -m alfs.update.generate_relabel_targets \\
        --alfs alfs.json --output-dir targets/ [--labeled labeled.parquet]

If --labeled is provided, only forms with existing rows in labeled.parquet
are emitted (for relabeling runs).
"""

import argparse
from pathlib import Path
from urllib.parse import quote

import polars as pl

from alfs.data_models.alf import Alfs
from alfs.data_models.update_target import UpdateTarget


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one target JSON file per form in alfs.json"
    )
    parser.add_argument("--alfs", required=True, help="Path to alfs.json")
    parser.add_argument(
        "--output-dir", required=True, help="Directory for target JSON files"
    )
    parser.add_argument(
        "--labeled",
        default=None,
        help="If set, only emit targets for forms in labeled.parquet",
    )
    args = parser.parse_args()

    alfs = Alfs.model_validate_json(Path(args.alfs).read_text())

    if args.labeled:
        labeled_forms = set(pl.read_parquet(args.labeled)["form"].to_list())
        forms = [f for f in alfs.entries if f in labeled_forms]
    else:
        forms = list(alfs.entries)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for form in forms:
        target = UpdateTarget(form=form)
        safe = quote(form, safe="")
        (out_dir / f"{safe}.json").write_text(target.model_dump_json())
        print(f"  {form}")

    print(f"Wrote {len(forms)} targets to {out_dir}")


if __name__ == "__main__":
    main()
