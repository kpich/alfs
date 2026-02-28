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

    store = SenseStore(Path(args.senses_db))
    all_forms = store.all_forms()

    if args.labeled_db and Path(args.labeled_db).exists():
        occ_store = OccurrenceStore(Path(args.labeled_db))
        labeled_df = occ_store.to_polars()
        labeled_forms = set(labeled_df["form"].to_list())
        forms = [f for f in all_forms if f in labeled_forms]
    else:
        forms = all_forms

    if args.nwords is not None:
        forms = random.sample(forms, min(args.nwords, len(forms)))

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
