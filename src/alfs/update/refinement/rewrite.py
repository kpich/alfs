"""LLM-assisted sense rewrite suggestions, queued for human approval.

Usage:
    python -m alfs.update.refinement.rewrite \\
        --senses-db ../alfs_data/senses.db \\
        --changes-db ../alfs_data/changes.db \\
        [--n 5] [--model gemma2:9b]
"""

import argparse
from datetime import datetime
from pathlib import Path
import random
import uuid

from alfs.data_models.alf import Sense
from alfs.data_models.change_store import Change, ChangeStatus, ChangeStore, ChangeType
from alfs.data_models.sense_store import SenseStore
from alfs.update import llm
from alfs.update.refinement import prompts

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
        description="LLM-assisted sense rewrite suggestions queued for human approval"
    )
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument("--changes-db", required=True, help="Path to changes.db")
    parser.add_argument("--n", type=int, default=5, help="Number of forms to rewrite")
    parser.add_argument("--model", default="gemma2:9b")
    args = parser.parse_args()

    store = SenseStore(Path(args.senses_db))
    change_store = ChangeStore(Path(args.changes_db))

    eligible = [
        (f, a) for f, a in store.all_entries().items() if not a.redirect and a.senses
    ]
    selected = random.sample(eligible, min(args.n, len(eligible)))

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
                definition=s["definition"],
                subsenses=s.get("subsenses", []),
                pos=alf.senses[i].pos,
            )
            for i, s in enumerate(returned)
        ]
        change = Change(
            id=str(uuid.uuid4()),
            type=ChangeType.rewrite,
            form=form,
            data={
                "before": [s.model_dump() for s in alf.senses],
                "after": [s.model_dump() for s in after],
            },
            status=ChangeStatus.pending,
            created_at=datetime.utcnow(),
        )
        change_store.add(change)
        print(f"  queued rewrite for: {form!r}")

    print(f"Queued {len(selected)} rewrites.")


if __name__ == "__main__":
    main()
