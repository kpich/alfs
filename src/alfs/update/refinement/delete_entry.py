"""LLM-driven entry deletion for mistokenized/artifact word forms.

Usage:
    python -m alfs.update.refinement.delete_entry \\
        --senses-db ../alfs_data/senses.db \\
        --labeled-db ../alfs_data/labeled.db \\
        --queue-dir ../clerk_queue \\
        [--n 10] [--model qwen2.5:32b]
"""

import argparse
import os
from pathlib import Path
import uuid

import polars as pl

from alfs.clerk.queue import enqueue
from alfs.clerk.request import DeleteEntryRequest
from alfs.corpus import fetch_instances
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore
from alfs.update import llm
from alfs.update.refinement import prompts

_DELETE_SCHEMA = {
    "type": "object",
    "properties": {
        "should_delete": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["should_delete", "reason"],
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
        description="LLM-driven entry deletion for artifact word forms"
    )
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument("--labeled-db", required=True, help="Path to labeled.db")
    parser.add_argument("--docs", required=True, help="Path to docs.parquet")
    parser.add_argument(
        "--queue-dir", required=True, help="Path to clerk queue directory"
    )
    parser.add_argument("--n", type=int, default=10, help="Number of forms to evaluate")
    parser.add_argument("--model", default="qwen2.5:32b")
    parser.add_argument(
        "--cc-tasks-dir",
        default=None,
        help="Path to CC tasks directory (writes task file instead of calling LLM)",
    )
    args = parser.parse_args()

    cc_tasks_dir = args.cc_tasks_dir or os.environ.get("CC_TASKS_DIR")

    store = SenseStore(Path(args.senses_db))
    occ_store = OccurrenceStore(Path(args.labeled_db))
    docs_df = pl.read_parquet(args.docs)
    labeled_df = occ_store.to_polars()
    queue_dir = Path(args.queue_dir)

    counts_df = occ_store.count_by_form()
    all_entries = store.all_entries()

    eligible = [(f, a) for f, a in all_entries.items() if not a.redirect and a.senses]

    if counts_df.height > 0:
        count_map: dict[str, int] = dict(
            zip(
                counts_df["form"].to_list(),
                counts_df["n_total"].to_list(),
                strict=False,
            )
        )
    else:
        count_map = {}

    eligible.sort(key=lambda fa: count_map.get(fa[0], 0))
    selected = eligible[: args.n]

    if cc_tasks_dir:
        from alfs.cc.models import CCDeleteEntryTask, SenseInfo

        pending_dir = Path(cc_tasks_dir) / "pending" / "delete_entry"
        pending_dir.mkdir(parents=True, exist_ok=True)
        for form, alf in selected:
            examples = [
                fetch_instances(
                    form,
                    alf.senses[i].id,
                    labeled_df,
                    docs_df,
                    min_rating=2,
                    context_chars=100,
                    max_instances=3,
                )
                for i in range(len(alf.senses))
            ]
            task = CCDeleteEntryTask(
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
                examples=examples,
            )
            task_path = pending_dir / f"{task.id}.json"
            task_path.write_text(task.model_dump_json())
            print(f"  wrote CC task for {form!r}")
        print("Done.")
        return

    for form, alf in selected:
        examples = [
            fetch_instances(
                form,
                alf.senses[i].id,
                labeled_df,
                docs_df,
                min_rating=2,
                context_chars=100,
                max_instances=3,
            )
            for i in range(len(alf.senses))
        ]

        data = llm.chat_json(
            args.model,
            prompts.delete_entry_prompt(form, list(alf.senses), examples),
            format=_DELETE_SCHEMA,
        )

        should_delete = data.get("should_delete", False)
        reason = data.get("reason", "")

        if not should_delete:
            print(f"  `{form}`: keeping entry")
            continue

        critic_data = llm.chat_json(
            args.model,
            prompts.delete_entry_critic_prompt(
                form, list(alf.senses), examples, reason
            ),
            format=_CRITIC_SCHEMA,
        )

        if not critic_data.get("is_valid", False):
            print(
                f"  `{form}`: critic rejected deletion — "
                f"{critic_data.get('reason', '')}"
            )
            continue

        request = DeleteEntryRequest(form=form, reason=reason)
        enqueue(request, queue_dir)
        print(f"  queued delete for {form!r} — {reason}")

    print("Done.")


if __name__ == "__main__":
    main()
