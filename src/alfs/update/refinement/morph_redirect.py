"""Propose morphological/derivational redirect links, via clerk queue.

Usage:
    python -m alfs.update.refinement.morph_redirect \\
        --senses-db ../alfs_data/senses.db \\
        --queue-dir ../clerk_queue \\
        [--n 50] [--batch-size 10] [--model gemma2:9b] [--seed 42]
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import random
import uuid

from alfs.clerk.queue import enqueue
from alfs.clerk.request import MorphRedirectRequest
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

_MORPH_CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "is_valid": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["is_valid", "reason"],
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
        description="Propose morphological redirect links via clerk queue"
    )
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument(
        "--queue-dir", required=True, help="Path to clerk queue directory"
    )
    parser.add_argument("--n", type=int, default=50, help="Number of forms to sample")
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Forms per screening batch"
    )
    parser.add_argument("--model", default="qwen2.5:32b")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--cc-tasks-dir",
        default=None,
        help="Path to CC tasks directory (writes task file instead of calling LLM)",
    )
    args = parser.parse_args()

    cc_tasks_dir = args.cc_tasks_dir or os.environ.get("CC_TASKS_DIR")

    sense_store = SenseStore(Path(args.senses_db))
    queue_dir = Path(args.queue_dir)

    all_entries = sense_store.all_entries()

    # Eligible = non-redirect, has at least one sense
    eligible = {
        form: alf
        for form, alf in all_entries.items()
        if alf.redirect is None and alf.senses
    }
    eligible_set = set(eligible.keys())

    # Sample eligible forms
    rng = random.Random(args.seed)
    sample_forms = rng.sample(list(eligible.keys()), min(args.n, len(eligible)))

    # Chunk into batches
    batches = [
        sample_forms[i : i + args.batch_size]
        for i in range(0, len(sample_forms), args.batch_size)
    ]

    if cc_tasks_dir:
        from alfs.cc.models import CCMorphRedirectTask, FormInfo, SenseInfo

        pending_dir = Path(cc_tasks_dir) / "pending" / "morph_redirect"
        pending_dir.mkdir(parents=True, exist_ok=True)
        inventory_forms = list(eligible_set)
        for batch in batches:
            forms_info = [
                FormInfo(
                    form=f,
                    senses=[
                        SenseInfo(
                            id=s.id,
                            definition=s.definition,
                            subsenses=list(s.subsenses) if s.subsenses else None,
                            pos=s.pos.value if s.pos else None,
                        )
                        for s in eligible[f].senses
                    ],
                )
                for f in batch
            ]
            task = CCMorphRedirectTask(
                id=str(uuid.uuid4()),
                forms=forms_info,
                inventory_forms=inventory_forms,
            )
            task_path = pending_dir / f"{task.id}.json"
            task_path.write_text(task.model_dump_json())
            print(f"  wrote CC task for batch of {len(batch)} forms")
        print("Done.")
        return

    total_queued = 0

    for batch in batches:
        # Screen batch for morphological candidates
        screen_data = llm.chat_json(
            args.model,
            prompts.morph_screen_prompt(batch),
            format=_SCREEN_SCHEMA,
        )
        candidates = screen_data.get("candidates", [])

        # Filter: base must also be in the eligible inventory, no self-references
        valid_pairs = [
            (c["form"], c["base"])
            for c in candidates
            if c["form"] in eligible_set
            and c["base"] in eligible_set
            and c["form"] != c["base"]
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
                after_sense = derived_sense.model_copy(
                    update={
                        "definition": proposed_def,
                        "morph_base": base_form,
                        "morph_relation": relation,
                        "updated_by_model": args.model,
                    }
                )

                verdict = llm.chat_json(
                    args.model,
                    prompts.morph_critic_prompt(
                        derived_form, base_form, relation, proposed_def
                    ),
                    format=_MORPH_CRITIC_SCHEMA,
                )
                if not verdict.get("is_valid", True):
                    print(
                        f"  skipped: critic rejected {derived_form!r} ← {base_form!r}"
                        f" ({verdict.get('reason', '')})"
                    )
                    continue

                request = MorphRedirectRequest(
                    id=str(uuid.uuid4()),
                    created_at=datetime.utcnow(),
                    form=derived_form,
                    derived_sense_idx=derived_idx,
                    base_form=base_form,
                    base_sense_idx=base_idx,
                    relation=relation,
                    before=derived_sense,
                    after=after_sense,
                )
                enqueue(request, queue_dir)
                total_queued += 1
                print(
                    f"  queued: {derived_form!r} sense {derived_idx}"
                    f" → {base_form!r} sense {base_idx} ({relation})"
                )

    noun = "change" if total_queued == 1 else "changes"
    print(f"Queued {total_queued} morph redirect {noun}.")


if __name__ == "__main__":
    main()
