"""LLM-driven redundant-sense deletion, auto-applied via clerk queue.

Usage:
    python -m alfs.update.refinement.trim_sense \\
        --senses-db ../alfs_data/senses.db \\
        --labeled-db ../alfs_data/labeled.db \\
        --docs ../text_data/latest/docs.parquet \\
        --queue-dir ../clerk_queue \\
        [--n 50] [--model gemma2:9b]
"""

import argparse
from datetime import datetime
from pathlib import Path
import random
import uuid

import polars as pl

from alfs.clerk.queue import enqueue
from alfs.clerk.request import TrimSenseRequest
from alfs.corpus import fetch_instances
from alfs.data_models.alf import sense_key
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore
from alfs.update import llm
from alfs.update.refinement import prompts

_TRIM_SCHEMA = {
    "type": "object",
    "properties": {
        "sense_num": {"type": ["integer", "null"]},
        "reason": {"type": "string"},
    },
    "required": ["sense_num", "reason"],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-driven redundant-sense deletion via clerk queue"
    )
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument("--labeled-db", required=True, help="Path to labeled.db")
    parser.add_argument("--docs", required=True, help="Path to docs.parquet")
    parser.add_argument(
        "--queue-dir", required=True, help="Path to clerk queue directory"
    )
    parser.add_argument("--n", type=int, default=50, help="Number of forms to evaluate")
    parser.add_argument("--model", default="gemma2:9b")
    args = parser.parse_args()

    store = SenseStore(Path(args.senses_db))
    occ_store = OccurrenceStore(Path(args.labeled_db))
    docs_df = pl.read_parquet(args.docs)
    labeled_df = occ_store.to_polars()
    queue_dir = Path(args.queue_dir)

    eligible = [
        (f, a)
        for f, a in store.all_entries().items()
        if not a.redirect and len(a.senses) >= 2
    ]
    selected = random.sample(eligible, min(args.n, len(eligible)))

    for form, alf in selected:
        examples = [
            fetch_instances(
                form,
                sense_key(i),
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
            prompts.trim_sense_prompt(form, list(alf.senses), examples),
            format=_TRIM_SCHEMA,
        )

        sense_num = data.get("sense_num")
        reason = data.get("reason", "")

        if sense_num is None:
            print(f"  `{form}`: all senses distinct")
            continue

        if not isinstance(sense_num, int) or not (1 <= sense_num <= len(alf.senses)):
            print(
                f"  skipped {form!r}: invalid sense_num {sense_num!r}"
                f" (have {len(alf.senses)} senses)"
            )
            continue

        deleted_idx = sense_num - 1
        remaining = [s for i, s in enumerate(alf.senses) if i != deleted_idx]

        request = TrimSenseRequest(
            id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
            form=form,
            before=list(alf.senses),
            after=remaining,
            deleted_idx=deleted_idx,
            reason=reason,
        )
        enqueue(request, queue_dir)
        print(f"  queued trim for {form!r}: sense {sense_num} — {reason}")

    print("Done.")


if __name__ == "__main__":
    main()
