"""Clear stale senses from redirect entries.

Redirect entries should never have definitions. This script finds any entry
where ``alf.redirect`` is set but ``alf.senses`` is non-empty and clears the
senses list.

Usage:
    python -m alfs.update.refinement.cleanup --senses-db ../alfs_data/senses.db
"""

import argparse
from pathlib import Path

from alfs.data_models.alf import Alf
from alfs.data_models.sense_store import SenseStore


def _clear_senses(existing: Alf | None) -> Alf:
    assert existing is not None
    return existing.model_copy(update={"senses": []})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clear stale senses from redirect entries"
    )
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    args = parser.parse_args()

    store = SenseStore(Path(args.senses_db))
    fixed = 0
    for form, alf in store.all_entries().items():
        if alf.redirect and alf.senses:
            store.update(form, _clear_senses)
            print(f"  cleared senses on redirect: {form!r} → {alf.redirect!r}")
            fixed += 1

    print(f"Fixed {fixed} entries.")


if __name__ == "__main__":
    main()
