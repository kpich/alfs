"""Propose morphological/derivational redirect links, queued for human approval.

Usage:
    python -m alfs.update.refinement.morph_redirect \\
        --senses-db ../alfs_data/senses.db \\
        --changes-db ../alfs_data/changes.db \\
        [--n 50] [--batch-size 10] [--model gemma2:9b] [--seed 42]
"""

import argparse
from datetime import datetime
from pathlib import Path
import random
import uuid

from alfs.data_models.change_store import Change, ChangeStatus, ChangeStore, ChangeType
from alfs.data_models.sense_store import SenseStore
from alfs.update import llm
from alfs.update.refinement import prompts

_SCREEN_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "form": {"type": "string"},
                    "base": {"type": "string"},
                },
                "required": ["form", "base"],
            },
        }
    },
    "required": ["candidates"],
}

_ANALYZE_SCHEMA = {
    "type": "object",
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "derived_sense_idx": {"type": "integer"},
                    "base_sense_idx": {"type": "integer"},
                    "relation": {"type": "string"},
                    "proposed_definition": {"type": "string"},
                },
                "required": [
                    "derived_sense_idx",
                    "base_sense_idx",
                    "relation",
                    "proposed_definition",
                ],
            },
        }
    },
    "required": ["relations"],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propose morphological redirect links queued for human approval"
    )
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument("--changes-db", required=True, help="Path to changes.db")
    parser.add_argument("--n", type=int, default=50, help="Number of forms to sample")
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Forms per screening batch"
    )
    parser.add_argument("--model", default="gemma2:9b")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    sense_store = SenseStore(Path(args.senses_db))
    change_store = ChangeStore(Path(args.changes_db))

    all_entries = sense_store.all_entries()

    # Eligible = non-redirect, has at least one sense
    eligible = {
        form: alf
        for form, alf in all_entries.items()
        if alf.redirect is None and alf.senses
    }
    eligible_set = set(eligible.keys())

    # Build set of already-pending (form, derived_sense_idx) pairs to skip duplicates
    pending_pairs: set[tuple[str, int]] = set()
    for change in change_store.all_pending():
        if change.type == ChangeType.morph_redirect:
            pending_pairs.add((change.form, change.data["derived_sense_idx"]))

    # Sample eligible forms
    rng = random.Random(args.seed)
    sample_forms = rng.sample(list(eligible.keys()), min(args.n, len(eligible)))

    # Chunk into batches
    batches = [
        sample_forms[i : i + args.batch_size]
        for i in range(0, len(sample_forms), args.batch_size)
    ]

    total_queued = 0

    for batch in batches:
        # Screen batch for morphological candidates
        screen_data = llm.chat_json(
            args.model,
            prompts.morph_screen_prompt(batch),
            format=_SCREEN_SCHEMA,
        )
        candidates = screen_data.get("candidates", [])

        # Filter: base must also be in the eligible inventory
        valid_pairs = [
            (c["form"], c["base"])
            for c in candidates
            if c["form"] in eligible_set and c["base"] in eligible_set
        ]

        for derived_form, base_form in valid_pairs:
            derived_alf = eligible[derived_form]
            base_alf = eligible[base_form]

            analyze_data = llm.chat_json(
                args.model,
                prompts.morph_analyze_prompt(
                    derived_form, base_form, derived_alf, base_alf
                ),
                format=_ANALYZE_SCHEMA,
            )
            relations = analyze_data.get("relations", [])

            for rel in relations:
                derived_idx = rel["derived_sense_idx"]
                base_idx = rel["base_sense_idx"]
                relation = rel["relation"]
                proposed_def = rel["proposed_definition"]

                # Skip already-pending pairs
                if (derived_form, derived_idx) in pending_pairs:
                    continue

                # Validate indices are in range
                if derived_idx < 0 or derived_idx >= len(derived_alf.senses):
                    print(
                        f"  skipped: {derived_form!r} sense {derived_idx} out of range"
                    )
                    continue
                if base_idx < 0 or base_idx >= len(base_alf.senses):
                    print(f"  skipped: {base_form!r} sense {base_idx} out of range")
                    continue

                derived_sense = derived_alf.senses[derived_idx]
                before = derived_sense.model_dump(exclude_none=True)
                after = {
                    **before,
                    "definition": proposed_def,
                    "morph_base": base_form,
                    "morph_relation": relation,
                }

                change = Change(
                    id=str(uuid.uuid4()),
                    type=ChangeType.morph_redirect,
                    form=derived_form,
                    data={
                        "derived_sense_idx": derived_idx,
                        "base_form": base_form,
                        "base_sense_idx": base_idx,
                        "relation": relation,
                        "before": before,
                        "after": after,
                    },
                    status=ChangeStatus.pending,
                    created_at=datetime.utcnow(),
                )
                change_store.add(change)
                pending_pairs.add((derived_form, derived_idx))
                total_queued += 1
                print(
                    f"  queued: {derived_form!r} sense {derived_idx}"
                    f" → {base_form!r} sense {base_idx} ({relation})"
                )

    noun = "change" if total_queued == 1 else "changes"
    print(f"Queued {total_queued} morph redirect {noun}.")


if __name__ == "__main__":
    main()
