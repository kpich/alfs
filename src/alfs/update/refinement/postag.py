"""LLM-assisted POS tagging for dictionary senses.

Usage:
    python -m alfs.update.refinement.postag \\
        --senses-db ../alfs_data/senses.db \\
        --labeled-db ../alfs_data/labeled.db \\
        --docs ../text_data/latest/docs.parquet \\
        --queue-dir ../clerk_queue \\
        [--model gemma2:9b]
"""

import argparse
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
import uuid

import polars as pl

from alfs.clerk.queue import enqueue
from alfs.clerk.request import UpdatePosRequest
from alfs.corpus import fetch_instances
from alfs.data_models.alf import Alf
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


def _make_tagger(
    form: str,
    labeled_df: pl.DataFrame,
    docs_df: pl.DataFrame,
    model: str,
) -> Callable[[Alf | None], Alf]:
    def tag_form(existing: Alf | None) -> Alf:
        if existing is None or existing.redirect:
            return existing or Alf(form=form)

        new_senses = []
        for top_idx, sense in enumerate(existing.senses):
            if sense.pos is not None:
                new_senses.append(sense)
                continue

            instances = fetch_instances(form, sense.id, labeled_df, docs_df)
            prompt = prompts.postag_prompt(form, sense.definition, instances)
            data = llm.chat_json(model, prompt, format=_POS_SCHEMA)
            pos_str = data["pos"]
            critic = llm.chat_json(
                model,
                prompts.postag_critic_prompt(
                    form, sense.definition, pos_str, instances
                ),
                format=_POS_CRITIC_SCHEMA,
            )
            if not critic.get("is_valid", True):
                print(
                    f"  {form!r} sense {top_idx + 1}: critic rejected"
                    f" pos={pos_str!r} ({critic.get('reason', '')})"
                )
                new_senses.append(sense)
                continue
            pos = PartOfSpeech(pos_str)
            new_senses.append(
                sense.model_copy(update={"pos": pos, "updated_by_model": model})
            )

        tagged = sum(1 for s in new_senses if s.pos is not None)
        print(f"  {form!r}: {tagged}/{len(new_senses)} senses tagged")
        return existing.model_copy(update={"senses": new_senses})

    return tag_form


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-assisted POS tagging for senses")
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument("--labeled-db", required=True, help="Path to labeled.db")
    parser.add_argument("--docs", required=True, help="Path to docs.parquet")
    parser.add_argument(
        "--queue-dir", required=True, help="Path to clerk queue directory"
    )
    parser.add_argument("--model", default="qwen2.5:32b")
    args = parser.parse_args()

    sense_store = SenseStore(Path(args.senses_db))
    occ_store = OccurrenceStore(Path(args.labeled_db))
    queue_dir = Path(args.queue_dir)
    labeled_df = occ_store.to_polars()
    docs_df = pl.read_parquet(args.docs)

    for form in sense_store.all_forms():
        existing = sense_store.read(form)
        if existing is None or existing.redirect:
            continue
        tagger = _make_tagger(form, labeled_df, docs_df, args.model)
        updated = tagger(existing)
        if list(updated.senses) == list(existing.senses):
            continue
        request = UpdatePosRequest(
            id=str(uuid.uuid4()),
            created_at=datetime.now(UTC),
            form=form,
            before=list(existing.senses),
            after=list(updated.senses),
            requesting_model=args.model,
        )
        enqueue(request, queue_dir)

    print("Done tagging POS.")


if __name__ == "__main__":
    main()
