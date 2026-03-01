"""LLM-assisted POS re-evaluation for already-tagged senses, queued for human approval.

Usage:
    python -m alfs.update.refinement.retag \\
        --senses-db ../alfs_data/senses.db \\
        --labeled-db ../alfs_data/labeled.db \\
        --docs ../text_data/latest/docs.parquet \\
        --changes-db ../alfs_data/changes.db \\
        [--n 10] [--model gemma2:9b]
"""

import argparse
from datetime import datetime
from pathlib import Path
import random
import uuid

import polars as pl

from alfs.corpus import fetch_instances
from alfs.data_models.alf import sense_key
from alfs.data_models.change_store import Change, ChangeStatus, ChangeStore, ChangeType
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-assisted POS re-evaluation queued for human approval"
    )
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument("--labeled-db", required=True, help="Path to labeled.db")
    parser.add_argument("--docs", required=True, help="Path to docs.parquet")
    parser.add_argument("--changes-db", required=True, help="Path to changes.db")
    parser.add_argument("--n", type=int, default=10, help="Number of forms to retag")
    parser.add_argument("--model", default="gemma2:9b")
    args = parser.parse_args()

    sense_store = SenseStore(Path(args.senses_db))
    occ_store = OccurrenceStore(Path(args.labeled_db))
    change_store = ChangeStore(Path(args.changes_db))
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
                changed_descriptions.append(
                    f"  sense {top_idx + 1}: {sense.pos.value}→{new_pos.value}"
                )
                new_senses.append(sense.model_copy(update={"pos": new_pos}))
            else:
                new_senses.append(sense)

        if changed_descriptions:
            change = Change(
                id=str(uuid.uuid4()),
                type=ChangeType.pos_tag,
                form=form,
                data={
                    "before": [s.model_dump() for s in alf.senses],
                    "after": [s.model_dump() for s in new_senses],
                },
                status=ChangeStatus.pending,
                created_at=datetime.utcnow(),
            )
            change_store.add(change)
            print(f"  {form!r}: queued pos_tag change")
            for desc in changed_descriptions:
                print(desc)
            queued += 1
        else:
            print(f"  {form!r}: no changes")

    print(f"Queued {queued} pos_tag changes.")


if __name__ == "__main__":
    main()
