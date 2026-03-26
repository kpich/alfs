"""LLM-assisted POS tagging and re-evaluation for senses, via clerk queue.

Usage:
    python -m alfs.update.refinement.retag \\
        --senses-db ../alfs_data/senses.db \\
        --labeled-db ../alfs_data/labeled.db \\
        --docs ../text_data/latest/docs.parquet \\
        --queue-dir ../clerk_queue \\
        [--n 10] [--model gemma2:9b]
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path
import random
import uuid

import polars as pl

from alfs.clerk.queue import enqueue
from alfs.clerk.request import PosTagRequest
from alfs.corpus import fetch_instances
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.pos import PartOfSpeech
from alfs.data_models.sense_store import SenseStore
from alfs.update import llm
from alfs.update.refinement import prompts
from alfs.update.refinement.schemas import CRITIC_SCHEMA as _POS_CRITIC_SCHEMA
from alfs.update.refinement.schemas import POS_SCHEMA as _POS_SCHEMA


def run(
    senses_db: str | Path,
    labeled_db: str | Path,
    docs: str | Path,
    queue_dir: str | Path,
    n: int = 10,
    model: str = "qwen2.5:32b",
) -> int:
    sense_store = SenseStore(Path(senses_db))
    occ_store = OccurrenceStore(Path(labeled_db))
    queue_dir = Path(queue_dir)
    labeled_df = occ_store.to_polars()
    docs_df = pl.read_parquet(docs)

    eligible = [
        (f, a)
        for f, a in sense_store.all_entries().items()
        if not a.redirect and a.senses
    ]
    selected = random.sample(eligible, min(n, len(eligible)))

    queued = 0
    for form, alf in selected:
        new_senses = []
        changed_descriptions = []

        for top_idx, sense in enumerate(alf.senses):
            instances = fetch_instances(form, sense.id, labeled_df, docs_df)
            prompt = prompts.postag_prompt(form, sense.definition, instances)
            data = llm.chat_json(model, prompt, format=_POS_SCHEMA)
            new_pos = PartOfSpeech(data["pos"])

            old_pos_str = sense.pos.value if sense.pos else "None"
            if new_pos != sense.pos:
                critic = llm.chat_json(
                    model,
                    prompts.postag_critic_prompt(
                        form, sense.definition, new_pos.value, instances
                    ),
                    format=_POS_CRITIC_SCHEMA,
                )
                if not critic.get("is_valid", True):
                    print(
                        f"  {form!r} sense {top_idx + 1}: critic rejected"
                        f" {old_pos_str}→{new_pos.value}"
                        f" ({critic.get('reason', '')})"
                    )
                    new_senses.append(sense)
                    continue
                changed_descriptions.append(
                    f"  sense {top_idx + 1}: {old_pos_str}→{new_pos.value}"
                )
                new_senses.append(
                    sense.model_copy(update={"pos": new_pos, "updated_by_model": model})
                )
            else:
                new_senses.append(sense)

        for before_sense, after_sense in zip(alf.senses, new_senses, strict=False):
            if before_sense == after_sense:
                continue
            enqueue(
                PosTagRequest(
                    id=str(uuid.uuid4()),
                    created_at=datetime.now(UTC),
                    form=form,
                    before=before_sense,
                    after=after_sense,
                    requesting_model=model,
                ),
                queue_dir,
            )
        if changed_descriptions:
            print(f"  {form!r}: queued pos_tag changes")
            for desc in changed_descriptions:
                print(desc)
            queued += 1
        else:
            print(f"  {form!r}: no changes")

    print(f"Queued {queued} pos_tag changes.")
    return queued


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
    parser.add_argument("--model", default="qwen2.5:32b")
    args = parser.parse_args()
    run(args.senses_db, args.labeled_db, args.docs, args.queue_dir, args.n, args.model)


if __name__ == "__main__":
    main()
