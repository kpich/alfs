"""Find British/American spelling variant pairs and write CC task files.

Usage:
    python -m alfs.update.refinement.spelling_variant \\
        --senses-db ../alfs_data/senses.db \\
        --cc-tasks-dir ../cc_tasks
"""

import argparse
from pathlib import Path
import re
import uuid

from alfs.cc.models import CCSpellingVariantTask, SpellingVariantPair
from alfs.data_models.sense_store import SenseStore

# Each entry: (pattern_on_variant, replacement_to_get_preferred)
# Variants are British; preferred is American.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"our$"), "or"),  # colour -> color, honour -> honor
    (re.compile(r"ise$"), "ize"),  # realise -> realize
    (re.compile(r"yse$"), "yze"),  # analyse -> analyze
    (re.compile(r"re$"), "er"),  # theatre -> theater, centre -> center
    (re.compile(r"ll([^l]*)$"), r"l\1"),  # travelling -> traveling (medial -ll-)
]

_BATCH_SIZE = 50


def _find_candidates(forms: set[str]) -> list[SpellingVariantPair]:
    pairs: list[SpellingVariantPair] = []
    seen: set[tuple[str, str]] = set()
    for form in sorted(forms):
        for pattern, replacement in _PATTERNS:
            preferred = pattern.sub(replacement, form)
            if preferred == form:
                continue
            if preferred not in forms:
                continue
            key = (form, preferred)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(
                SpellingVariantPair(variant_form=form, preferred_form=preferred)
            )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find British/American spelling variant pairs"
    )
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument(
        "--cc-tasks-dir", required=True, help="Path to CC tasks directory"
    )
    args = parser.parse_args()

    sense_store = SenseStore(Path(args.senses_db))
    all_entries = sense_store.all_entries()

    # Only consider forms that are not already redirected/variant
    eligible = {
        form
        for form, alf in all_entries.items()
        if alf.redirect is None and alf.spelling_variant_of is None and alf.senses
    }

    candidates = _find_candidates(eligible)
    if not candidates:
        print("No spelling variant candidates found.")
        return

    pending_dir = Path(args.cc_tasks_dir) / "pending" / "spelling_variant"
    pending_dir.mkdir(parents=True, exist_ok=True)

    batches = [
        candidates[i : i + _BATCH_SIZE] for i in range(0, len(candidates), _BATCH_SIZE)
    ]
    for batch in batches:
        task = CCSpellingVariantTask(id=str(uuid.uuid4()), candidates=batch)
        task_path = pending_dir / f"{task.id}.json"
        task_path.write_text(task.model_dump_json())
        print(f"  wrote CC task with {len(batch)} candidate pair(s)")

    print(f"Done. {len(candidates)} total candidates in {len(batches)} task file(s).")


if __name__ == "__main__":
    main()
