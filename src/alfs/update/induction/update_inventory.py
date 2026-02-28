"""Augment senses.db with new senses for a single form (never overwrites, only appends).

Usage:
    python -m alfs.update.induction.update_inventory \\
        --senses-file {form}_senses.json --senses-db senses.db
"""

import argparse
from pathlib import Path

from alfs.data_models.alf import Alf
from alfs.data_models.sense_store import SenseStore


def merge_entry(existing: Alf, new: Alf) -> Alf:
    """Return existing Alf with any genuinely new senses from new appended."""
    existing_defs = {s.definition.strip().lower() for s in existing.senses}
    new_senses = [
        s for s in new.senses if s.definition.strip().lower() not in existing_defs
    ]
    if not new_senses:
        return existing
    return Alf(
        form=existing.form,
        senses=list(existing.senses) + new_senses,
        redirect=existing.redirect,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment senses.db with new senses for a single form"
    )
    parser.add_argument(
        "--senses-file",
        required=True,
        help="Path to {form}_senses.json from INDUCE_SENSES",
    )
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    args = parser.parse_args()

    alf = Alf.model_validate_json(Path(args.senses_file).read_text())
    if not alf.senses:
        print(f"No new senses for '{alf.form}' (empty senses file)")
        return

    store = SenseStore(Path(args.senses_db))

    def merge(existing: Alf | None) -> Alf:
        if existing is None:
            print(f"  Added new entry for '{alf.form}' ({len(alf.senses)} senses)")
            return alf
        merged = merge_entry(existing, alf)
        new_count = len(merged.senses) - len(existing.senses)
        if new_count:
            print(f"  Appended {new_count} new senses for '{alf.form}'")
        else:
            print(f"  No new senses for '{alf.form}' (all duplicates)")
        return merged

    store.update(alf.form, merge)


if __name__ == "__main__":
    main()
