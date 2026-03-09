"""LLM-assisted sense rewrite suggestions, auto-applied via clerk queue.

Usage:
    python -m alfs.update.refinement.rewrite \\
        --senses-db ../alfs_data/senses.db \\
        --queue-dir ../clerk_queue \\
        [--n 5] [--model gemma2:9b]
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import random
import uuid

from alfs.clerk.queue import enqueue
from alfs.clerk.request import RewriteRequest
from alfs.data_models.alf import Sense
from alfs.data_models.sense_store import SenseStore
from alfs.update import llm
from alfs.update.refinement import prompts

_CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "is_improvement": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["is_improvement", "reason"],
}

_REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "senses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "definition": {"type": "string"},
                    "subsenses": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["definition", "subsenses"],
            },
        }
    },
    "required": ["senses"],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-assisted sense rewrite suggestions via clerk queue"
    )
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument(
        "--queue-dir", required=True, help="Path to clerk queue directory"
    )
    parser.add_argument("--n", type=int, default=5, help="Number of forms to rewrite")
    parser.add_argument("--model", default="qwen2.5:32b")
    parser.add_argument(
        "--cc-tasks-dir",
        default=None,
        help="Path to CC tasks directory (writes task file instead of calling LLM)",
    )
    args = parser.parse_args()

    cc_tasks_dir = args.cc_tasks_dir or os.environ.get("CC_TASKS_DIR")

    store = SenseStore(Path(args.senses_db))
    queue_dir = Path(args.queue_dir)

    eligible = [
        (f, a) for f, a in store.all_entries().items() if not a.redirect and a.senses
    ]
    selected = random.sample(eligible, min(args.n, len(eligible)))

    if cc_tasks_dir:
        from alfs.cc.models import CCRewriteTask, SenseInfo

        pending_dir = Path(cc_tasks_dir) / "pending" / "rewrite"
        pending_dir.mkdir(parents=True, exist_ok=True)
        for form, alf in selected:
            task = CCRewriteTask(
                id=str(uuid.uuid4()),
                form=form,
                senses=[
                    SenseInfo(
                        id=s.id,
                        definition=s.definition,
                        subsenses=list(s.subsenses) if s.subsenses else None,
                        pos=s.pos.value if s.pos else None,
                    )
                    for s in alf.senses
                ],
            )
            task_path = pending_dir / f"{task.id}.json"
            task_path.write_text(task.model_dump_json())
            print(f"  wrote CC task for {form!r}")
        print("Done.")
        return

    for form, alf in selected:
        data = llm.chat_json(
            args.model,
            prompts.rewrite_prompt(form, list(alf.senses)),
            format=_REWRITE_SCHEMA,
        )
        returned = data["senses"]
        if len(returned) != len(alf.senses):
            print(
                f"  skipped {form!r}: LLM returned {len(returned)} senses,"
                f" expected {len(alf.senses)}"
            )
            continue
        after = [
            Sense(
                id=alf.senses[i].id,
                definition=s["definition"],
                subsenses=s.get("subsenses") or None,
                pos=alf.senses[i].pos,
                updated_by_model=args.model,
            )
            for i, s in enumerate(returned)
        ]
        verdict = llm.chat_json(
            args.model,
            prompts.critic_prompt(form, list(alf.senses), after),
            format=_CRITIC_SCHEMA,
        )
        if not verdict.get("is_improvement", True):
            print(f"  skipped {form!r} (critic: {verdict.get('reason', '')})")
            continue

        request = RewriteRequest(
            id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
            form=form,
            before=list(alf.senses),
            after=after,
        )
        enqueue(request, queue_dir)
        print(f"  queued rewrite for: {form!r}")

    print("Done.")


if __name__ == "__main__":
    main()
