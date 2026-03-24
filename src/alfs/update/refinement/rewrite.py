"""LLM-assisted sense rewrite suggestions, auto-applied via clerk queue.

Usage:
    python -m alfs.update.refinement.rewrite \\
        --senses-db ../alfs_data/senses.db \\
        --queue-dir ../clerk_queue \\
        [--n 5] [--model gemma2:9b]
"""

import argparse
from datetime import UTC, datetime
import os
from pathlib import Path
import random
import uuid

from alfs.clerk.queue import enqueue
from alfs.clerk.request import RewriteRequest
from alfs.data_models.alf import Sense, morph_base_form
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
        "rewrites": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sense_num": {"type": "integer"},
                    "definition": {"type": "string"},
                },
                "required": ["sense_num", "definition"],
            },
        }
    },
    "required": ["rewrites"],
}


def run(
    senses_db: str | Path,
    queue_dir: str | Path,
    n: int = 5,
    model: str = "qwen2.5:32b",
    cc_tasks_dir: str | Path | None = None,
) -> None:
    store = SenseStore(Path(senses_db))
    queue_dir = Path(queue_dir)

    eligible = [
        (f, a) for f, a in store.all_entries().items() if not a.redirect and a.senses
    ]
    selected = random.sample(eligible, min(n, len(eligible)))

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
        base_name = morph_base_form(alf)
        base_senses: list[Sense] | None = None
        if base_name is not None:
            base_alf = store.read(base_name)
            if base_alf is not None and base_alf.senses:
                base_senses = list(base_alf.senses)
        data = llm.chat_json(
            model,
            prompts.rewrite_prompt(form, list(alf.senses), base_name, base_senses),
            format=_REWRITE_SCHEMA,
        )
        rewrites = data["rewrites"]
        if not rewrites:
            print(f"  {form!r}: no changes proposed")
            continue

        changed: list[tuple[Sense, Sense]] = []
        for item in rewrites:
            idx = item["sense_num"] - 1
            if idx < 0 or idx >= len(alf.senses):
                print(f"  skipped {form!r}: sense_num {item['sense_num']} out of range")
                continue
            orig = alf.senses[idx]
            revised = Sense(
                id=orig.id,
                definition=item["definition"],
                pos=orig.pos,
                updated_by_model=model,
            )
            if orig != revised:
                changed.append((orig, revised))

        if not changed:
            print(f"  {form!r}: no effective changes")
            continue

        verdict = llm.chat_json(
            model,
            prompts.critic_prompt(form, list(alf.senses), changed),
            format=_CRITIC_SCHEMA,
        )
        if not verdict.get("is_improvement", True):
            print(f"  skipped {form!r} (critic: {verdict.get('reason', '')})")
            continue

        for orig, revised in changed:
            enqueue(
                RewriteRequest(
                    id=str(uuid.uuid4()),
                    created_at=datetime.now(UTC),
                    form=form,
                    before=orig,
                    after=revised,
                    requesting_model=model,
                ),
                queue_dir,
            )
        print(f"  queued rewrite for: {form!r}")

    print("Done.")


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

    run(args.senses_db, args.queue_dir, args.n, args.model, cc_tasks_dir)


if __name__ == "__main__":
    main()
