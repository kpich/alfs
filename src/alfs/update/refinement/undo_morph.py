"""Detect and undo incorrect morphological links, via clerk queue.

Usage:
    python -m alfs.update.refinement.undo_morph \\
        --senses-db ../alfs_data/senses.db \\
        --queue-dir ../clerk_queue \\
        [--n 10] [--model gemma2:9b] [--seed 42]
"""

import argparse
from datetime import datetime
from pathlib import Path
import random
import uuid

from alfs.clerk.queue import enqueue
from alfs.clerk.request import RewriteRequest
from alfs.data_models.alf import Sense
from alfs.data_models.sense_store import SenseStore
from alfs.update import llm
from alfs.update.refinement import prompts

_SCREEN_SCHEMA = {
    "type": "object",
    "properties": {
        "bad_links": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_num": {"type": "integer"},
                    "proposed_definition": {"type": "string"},
                },
                "required": ["item_num", "proposed_definition"],
            },
        }
    },
    "required": ["bad_links"],
}

_CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "is_valid": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["is_valid", "reason"],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and undo incorrect morphological links via clerk queue"
    )
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument(
        "--queue-dir", required=True, help="Path to clerk queue directory"
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Number of morph-linked senses to sample"
    )
    parser.add_argument("--model", default="gemma2:9b")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    sense_store = SenseStore(Path(args.senses_db))
    queue_dir = Path(args.queue_dir)

    all_entries = sense_store.all_entries()

    # Collect all (form, sense_idx, sense) triples with a morph link
    morph_triples: list[tuple[str, int, Sense]] = []
    for form, alf in all_entries.items():
        for idx, sense in enumerate(alf.senses):
            if sense.morph_base is not None:
                morph_triples.append((form, idx, sense))

    if not morph_triples:
        print("No morph-linked senses found.")
        return

    rng = random.Random(args.seed)
    sample = rng.sample(morph_triples, min(args.n, len(morph_triples)))

    print(f"Screening {len(sample)} morph-linked senses...")

    screen_data = llm.chat_json(
        args.model,
        prompts.undo_morph_screen_prompt(sample),
        format=_SCREEN_SCHEMA,
    )
    bad_links = screen_data.get("bad_links", [])

    total_queued = 0

    for item in bad_links:
        item_num = item.get("item_num")
        proposed_def = item.get("proposed_definition", "")

        # Validate item_num in range (1-based)
        if not isinstance(item_num, int) or item_num < 1 or item_num > len(sample):
            print(f"  skipped: item_num {item_num!r} out of range")
            continue

        form, sense_idx, sense = sample[item_num - 1]

        # Re-check sense still has morph_base (guard against stale data)
        current_alf = all_entries.get(form)
        if current_alf is None or sense_idx >= len(current_alf.senses):
            print(f"  skipped: {form!r} sense {sense_idx} no longer exists")
            continue
        current_sense = current_alf.senses[sense_idx]
        if current_sense.morph_base is None:
            print(f"  skipped: {form!r} sense {sense_idx} no longer has morph_base")
            continue

        morph_base = sense.morph_base or ""
        morph_relation = sense.morph_relation or ""

        # Critic call
        verdict = llm.chat_json(
            args.model,
            prompts.undo_morph_critic_prompt(
                form,
                sense_idx,
                morph_base,
                morph_relation,
                sense.definition,
                proposed_def,
            ),
            format=_CRITIC_SCHEMA,
        )
        if not verdict.get("is_valid", False):
            print(
                f"  skipped: critic rejected undo of {form!r} sense {sense_idx}"
                f" ({verdict.get('reason', '')})"
            )
            continue

        # Build updated senses list with morph fields cleared
        before_senses = list(current_alf.senses)
        after_sense = current_sense.model_copy(
            update={
                "definition": proposed_def,
                "morph_base": None,
                "morph_relation": None,
            }
        )
        after_senses = list(before_senses)
        after_senses[sense_idx] = after_sense

        request = RewriteRequest(
            id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
            form=form,
            before=before_senses,
            after=after_senses,
        )
        enqueue(request, queue_dir)
        total_queued += 1
        print(
            f"  queued: undo morph link on {form!r} sense {sense_idx}"
            f" (was {morph_relation!r} of {morph_base!r})"
        )

    noun = "change" if total_queued == 1 else "changes"
    print(f"Queued {total_queued} undo morph {noun}.")


if __name__ == "__main__":
    main()
