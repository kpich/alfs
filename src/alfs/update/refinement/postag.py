"""LLM-assisted POS tagging for dictionary senses.

Usage:
    python -m alfs.update.refinement.postag \\
        --alfs ../alfs_data/alfs.json \\
        --labeled ../alfs_data/labeled.parquet \\
        --docs ../text_data/latest/docs.parquet \\
        --output ../alfs_data/alfs.json \\
        [--model gemma2:9b]
"""

import argparse
from pathlib import Path

import polars as pl

from alfs.corpus import fetch_instances
from alfs.data_models.alf import Alf, Alfs, sense_key
from alfs.data_models.pos import PartOfSpeech
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
    parser = argparse.ArgumentParser(description="LLM-assisted POS tagging for senses")
    parser.add_argument("--alfs", required=True, help="Path to alfs.json")
    parser.add_argument("--labeled", required=True, help="Path to labeled.parquet")
    parser.add_argument("--docs", required=True, help="Path to docs.parquet")
    parser.add_argument("--output", required=True, help="Output path for alfs.json")
    parser.add_argument("--model", default="gemma2:9b")
    args = parser.parse_args()

    alfs = Alfs.model_validate_json(Path(args.alfs).read_text())
    labeled_df = pl.read_parquet(args.labeled)
    docs_df = pl.read_parquet(args.docs)

    new_entries: dict[str, Alf] = {}
    for form, alf in alfs.entries.items():
        if alf.redirect:
            new_entries[form] = alf
            continue

        new_senses = []
        for top_idx, sense in enumerate(alf.senses):
            if sense.pos is not None:
                new_senses.append(sense)
                continue

            sk = sense_key(top_idx)
            instances = fetch_instances(form, sk, labeled_df, docs_df)
            prompt = prompts.postag_prompt(form, sense.definition, instances)
            data = llm.chat_json(args.model, prompt, format=_POS_SCHEMA)
            pos = PartOfSpeech(data["pos"])
            new_senses.append(sense.model_copy(update={"pos": pos}))

        tagged = sum(1 for s in new_senses if s.pos is not None)
        print(f"  {form!r}: {tagged}/{len(new_senses)} senses tagged")
        new_entries[form] = alf.model_copy(update={"senses": new_senses})

    new_alfs = alfs.model_copy(update={"entries": new_entries})
    Path(args.output).write_text(new_alfs.model_dump_json(indent=2))
    print(f"Wrote updated dictionary to {args.output}")


if __name__ == "__main__":
    main()
