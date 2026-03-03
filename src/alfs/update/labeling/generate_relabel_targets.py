"""Generate one UpdateTarget JSON file per entry in senses.db.

Usage:
    python -m alfs.update.labeling.generate_relabel_targets \\
        --senses-db senses.db --output-dir targets/ [--labeled-db labeled.db]

If --labeled-db is provided, only forms with existing rows in labeled.db
are emitted (for relabeling runs).
"""

import argparse
from pathlib import Path
import random
from urllib.parse import quote

from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore
from alfs.data_models.update_target import UpdateTarget


def generate_targets(
    senses_db: Path,
    output_dir: Path,
    labeled_db: Path | None = None,
    nwords: int | None = None,
) -> list[Path]:
    """Write one target JSON per form; return list of written paths."""
    store = SenseStore(senses_db)
    all_forms = store.all_forms()

    if labeled_db and labeled_db.exists():
        occ_store = OccurrenceStore(labeled_db)
        labeled_df = occ_store.to_polars()
        labeled_forms = set(labeled_df["form"].to_list())
        forms = [f for f in all_forms if f in labeled_forms]
    else:
        forms = all_forms

    if nwords is not None:
        forms = random.sample(forms, min(nwords, len(forms)))

    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for form in forms:
        target = UpdateTarget(form=form)
        safe = quote(form, safe="")
        path = output_dir / f"{safe}.json"
        path.write_text(target.model_dump_json())
        written.append(path)
        print(f"  {form}")

    print(f"Wrote {len(forms)} targets to {output_dir}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one target JSON file per form in senses.db"
    )
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument(
        "--output-dir", required=True, help="Directory for target JSON files"
    )
    parser.add_argument(
        "--labeled-db",
        default=None,
        help="If set, only emit targets for forms in labeled.db",
    )
    parser.add_argument(
        "--nwords",
        type=int,
        default=None,
        help="If set, randomly sample this many forms instead of using all",
    )
    args = parser.parse_args()

    generate_targets(
        Path(args.senses_db),
        Path(args.output_dir),
        Path(args.labeled_db) if args.labeled_db else None,
        args.nwords,
    )


if __name__ == "__main__":
    main()
