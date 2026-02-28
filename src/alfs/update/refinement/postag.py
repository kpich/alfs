"""LLM-assisted POS tagging for dictionary senses.

Usage:
    python -m alfs.update.refinement.postag \\
        --senses-db ../alfs_data/senses.db \\
        --labeled-db ../alfs_data/labeled.db \\
        --docs ../text_data/latest/docs.parquet \\
        [--model gemma2:9b]
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import polars as pl

from alfs.corpus import fetch_instances
from alfs.data_models.alf import Alf, sense_key
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

            sk = sense_key(top_idx)
            instances = fetch_instances(form, sk, labeled_df, docs_df)
            prompt = prompts.postag_prompt(form, sense.definition, instances)
            data = llm.chat_json(model, prompt, format=_POS_SCHEMA)
            pos = PartOfSpeech(data["pos"])
            new_senses.append(sense.model_copy(update={"pos": pos}))

        tagged = sum(1 for s in new_senses if s.pos is not None)
        print(f"  {form!r}: {tagged}/{len(new_senses)} senses tagged")
        return existing.model_copy(update={"senses": new_senses})

    return tag_form


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-assisted POS tagging for senses")
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument("--labeled-db", required=True, help="Path to labeled.db")
    parser.add_argument("--docs", required=True, help="Path to docs.parquet")
    parser.add_argument("--model", default="gemma2:9b")
    args = parser.parse_args()

    sense_store = SenseStore(Path(args.senses_db))
    occ_store = OccurrenceStore(Path(args.labeled_db))
    labeled_df = occ_store.to_polars()
    docs_df = pl.read_parquet(args.docs)

    for form in sense_store.all_forms():
        sense_store.update(form, _make_tagger(form, labeled_df, docs_df, args.model))

    print("Done tagging POS.")


if __name__ == "__main__":
    main()
