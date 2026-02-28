"""Augment alfs.json with new senses (never overwrites, only appends).

Usage:
    python -m alfs.update.update_inventory \\
        --alfs-data alfs.json --senses-dir senses/ --output alfs.json
"""

import argparse
from pathlib import Path

from alfs.data_models.alf import Alf, Alfs


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
        description="Augment alfs inventory with new senses"
    )
    parser.add_argument("--alfs-data", required=True, help="Path to alfs.json")
    parser.add_argument(
        "--senses-dir", required=True, help="Directory with *.json sense files"
    )
    parser.add_argument(
        "--output", required=True, help="Output path for updated alfs.json"
    )
    args = parser.parse_args()

    alfs_path = Path(args.alfs_data)
    if alfs_path.exists():
        alfs = Alfs.model_validate_json(alfs_path.read_text())
    else:
        alfs = Alfs()

    entries = dict(alfs.entries)

    for sense_file in sorted(Path(args.senses_dir).glob("*.json")):
        alf = Alf.model_validate_json(sense_file.read_text())
        form = alf.form

        if form not in entries:
            entries[form] = alf
            print(f"  Added new entry for '{form}' ({len(alf.senses)} senses)")
        else:
            existing = entries[form]
            merged = merge_entry(existing, alf)
            new_count = len(merged.senses) - len(existing.senses)
            if new_count:
                entries[form] = merged
                print(f"  Appended {new_count} new senses for '{form}'")
            else:
                print(f"  No new senses for '{form}' (all duplicates)")
            # TODO: when UpdateTarget.sense is non-None, refine that specific sense
            # instead

    updated = Alfs(entries=entries)
    Path(args.output).write_text(updated.model_dump_json(indent=2))
    print(f"Wrote updated inventory to {args.output}")


if __name__ == "__main__":
    main()
