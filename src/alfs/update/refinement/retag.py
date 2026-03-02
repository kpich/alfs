"""LLM-assisted POS re-evaluation for already-tagged senses, via clerk queue.

Usage:
    python -m alfs.update.refinement.retag \\
        --senses-db ../alfs_data/senses.db \\
        --labeled-db ../alfs_data/labeled.db \\
        --docs ../text_data/latest/docs.parquet \\
        --queue-dir ../clerk_queue \\
        [--n 10] [--model gemma2:9b]
"""

import argparse
from datetime import datetime
from pathlib import Path
import random
import uuid

import polars as pl

from alfs.clerk.queue import enqueue
from alfs.clerk.request import PosTagRequest
from alfs.corpus import fetch_instances
from alfs.data_models.alf import sense_key
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.pos import PartOfSpeech
from alfs.data_models.sense_store import SenseStore
from alfs.update import llm
from alfs.update.refinement import prompts

_POS_VALUES = [p.value for p in PartOfSpeech]

_POS_SCHEMA = {
    "type": "object",
    "properties": {
        "pos": {"type": "string", "enum": _POS_VALUES},
    },
    "required": ["pos"],
}

_POS_CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "is_valid": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["is_valid", "reason"],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-assisted POS re-evaluation via clerk queue"
    )
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument("--labeled-db", required=True, help="Path to labeled.db")
    parser.add_argument("--docs", required=True, help="Path to docs.parquet")
    parser.add_argument(
        "--queue-dir", required=True, help="Path to clerk queue directory"
    )
    parser.add_argument("--n", type=int, default=10, help="Number of forms to retag")
    parser.add_argument("--model", default="gemma2:9b")
    args = parser.parse_args()

    sense_store = SenseStore(Path(args.senses_db))
    occ_store = OccurrenceStore(Path(args.labeled_db))
    queue_dir = Path(args.queue_dir)
    labeled_df = occ_store.to_polars()
    docs_df = pl.read_parquet(args.docs)

    eligible = [
        (f, a)
        for f, a in sense_store.all_entries().items()
        if not a.redirect and any(s.pos is not None for s in a.senses)
    ]
    selected = random.sample(eligible, min(args.n, len(eligible)))

    queued = 0
    for form, alf in selected:
        new_senses = []
        changed_descriptions = []

        for top_idx, sense in enumerate(alf.senses):
            if sense.pos is None:
                new_senses.append(sense)
                continue

            sk = sense_key(top_idx)
            instances = fetch_instances(form, sk, labeled_df, docs_df)
            prompt = prompts.postag_prompt(form, sense.definition, instances)
            data = llm.chat_json(args.model, prompt, format=_POS_SCHEMA)
            new_pos = PartOfSpeech(data["pos"])

            if new_pos != sense.pos:
                critic = llm.chat_json(
                    args.model,
                    prompts.postag_critic_prompt(
                        form, sense.definition, new_pos.value, instances
                    ),
                    format=_POS_CRITIC_SCHEMA,
                )
                if not critic.get("is_valid", True):
                    print(
                        f"  {form!r} sense {top_idx + 1}: critic rejected"
                        f" {sense.pos.value}→{new_pos.value}"
                        f" ({critic.get('reason', '')})"
                    )
                    new_senses.append(sense)
                    continue
                changed_descriptions.append(
                    f"  sense {top_idx + 1}: {sense.pos.value}→{new_pos.value}"
                )
                new_senses.append(sense.model_copy(update={"pos": new_pos}))
            else:
                new_senses.append(sense)

        if changed_descriptions:
            request = PosTagRequest(
                id=str(uuid.uuid4()),
                created_at=datetime.utcnow(),
                form=form,
                before=list(alf.senses),
                after=new_senses,
            )
            enqueue(request, queue_dir)
            print(f"  {form!r}: queued pos_tag change")
            for desc in changed_descriptions:
                print(desc)
            queued += 1
        else:
            print(f"  {form!r}: no changes")

    print(f"Queued {queued} pos_tag changes.")


if __name__ == "__main__":
    main()
