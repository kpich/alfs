"""Induce senses for a word form using Ollama.

Usage:
    python -m alfs.update.induction.induce_senses \\
        --target target.json --seg-data-dir by_prefix/ --docs docs.parquet \\
        --output senses.json --model qwen2.5:32b --context-chars 150 --max-samples 20 \\
        [--senses-db senses.db] [--labeled-db labeled.db] [--cc-tasks-dir DIR]
"""

import argparse
import os
from pathlib import Path
import random
import uuid

import polars as pl

from alfs.data_models.alf import Alf, Sense, morph_base_form
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.pos import PartOfSpeech
from alfs.data_models.sense_store import SenseStore
from alfs.data_models.update_target import UpdateTarget
from alfs.seg.aggregate_occurrences import prefix as form_prefix
from alfs.update import llm
from alfs.update.induction import prompts

_SENSE_SCHEMA = {
    "type": "object",
    "properties": {
        "all_covered": {"type": "boolean"},
        "senses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "definition": {"type": "string"},
                    "examples": {"type": "array", "items": {"type": "integer"}},
                    "pos": {"type": "string", "enum": [p.value for p in PartOfSpeech]},
                },
                "required": ["definition", "examples", "pos"],
            },
        },
    },
    "required": ["senses"],
}

_CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "is_valid": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["is_valid", "reason"],
}


def extract_context(text: str, byte_offset: int, form: str, context_chars: int) -> str:
    char_offset = len(text.encode()[:byte_offset].decode())
    start = max(0, char_offset - context_chars)
    end = char_offset + len(form) + context_chars
    return text[start:end]


def run(
    target_file: str | Path,
    seg_data_dir: str | Path,
    docs: str | Path,
    output: str | Path,
    model: str = "qwen2.5:32b",
    context_chars: int = 150,
    max_samples: int = 20,
    senses_db: str | Path | None = None,
    labeled_db: str | Path | None = None,
    cc_tasks_dir: str | Path | None = None,
) -> None:
    target = UpdateTarget.model_validate_json(Path(target_file).read_text())
    form = target.form

    existing_defs: list[str] = []
    if senses_db and Path(senses_db).exists():
        store = SenseStore(Path(senses_db))
        entry = store.read(form)
        if entry:
            resolved = entry
            if entry.redirect:
                canonical = store.read(entry.redirect)
                if canonical:
                    resolved = canonical
            existing_defs = [s.definition for s in resolved.senses]
            base_name = morph_base_form(resolved)
            if base_name is not None:
                base_entry = store.read(base_name)
                if base_entry is not None:
                    existing_defs.extend(s.definition for s in base_entry.senses)

    occ_path = Path(seg_data_dir) / form_prefix(form) / "occurrences.parquet"
    if not occ_path.exists():
        alf = Alf(form=form, senses=[])
        Path(output).write_text(alf.model_dump_json())
        print(f"No occurrences parquet for '{form}' ({occ_path}); skipping.")
        return
    df = pl.read_parquet(str(occ_path)).filter(pl.col("form") == form)
    all_occurrences = list(df.select(["doc_id", "byte_offset"]).iter_rows(named=True))

    well_labeled: set[tuple[str, int]] = set()
    if labeled_db and Path(labeled_db).exists():
        occ_store = OccurrenceStore(Path(labeled_db))
        ldf = occ_store.query_form(form).filter(pl.col("rating").is_in([3]))
        for row in ldf.select(["doc_id", "byte_offset"]).iter_rows():
            well_labeled.add((row[0], row[1]))
    all_occurrences = [
        o
        for o in all_occurrences
        if (o["doc_id"], o["byte_offset"]) not in well_labeled
    ]

    random.seed(42)
    samples = random.sample(all_occurrences, min(max_samples, len(all_occurrences)))

    needed_doc_ids = list({occ["doc_id"] for occ in samples})
    docs_df = (
        pl.scan_parquet(str(docs))
        .filter(pl.col("doc_id").is_in(needed_doc_ids))
        .collect(engine="streaming")
    )
    docs_map = dict(
        zip(docs_df["doc_id"].to_list(), docs_df["text"].to_list(), strict=False)
    )

    contexts = []
    for occ in samples:
        text = docs_map.get(occ["doc_id"], "")
        if not text:
            continue
        ctx = extract_context(text, occ["byte_offset"], form, context_chars)
        contexts.append(ctx)

    if not contexts:
        alf = Alf(form=form, senses=[])
        Path(output).write_text(alf.model_dump_json())
        # TODO: investigate why doc_ids from occurrences parquet are absent from
        # docs.parquet for this form — this likely indicates a data pipeline bug
        print(f"No contexts found for '{form}'; skipping.")
        return

    if cc_tasks_dir:
        from alfs.cc.models import CCInductionTask

        task = CCInductionTask(
            id=str(uuid.uuid4()),
            form=form,
            contexts=contexts,
            existing_defs=existing_defs,
        )
        pending_dir = Path(cc_tasks_dir) / "pending" / "induction"
        pending_dir.mkdir(parents=True, exist_ok=True)
        task_path = pending_dir / f"{task.id}.json"
        task_path.write_text(task.model_dump_json())
        # Write empty output so downstream pipeline doesn't fail
        alf = Alf(form=form, senses=[])
        Path(output).write_text(alf.model_dump_json())
        print(f"Wrote CC task for '{form}' to {task_path}")
        return

    prompt = prompts.induction_prompt(form, contexts, existing_defs)
    data = llm.chat_json(model, prompt, format=_SENSE_SCHEMA)

    if data.get("all_covered", False) and existing_defs:
        alf = Alf(form=form, senses=[])
        Path(output).write_text(alf.model_dump_json())
        print(f"Existing senses cover all contexts for '{form}'; no new sense added.")
        return

    proposed = data.get("senses", [])
    accepted: list[Sense] = []
    for item in proposed:
        pos_str = item.get("pos")
        try:
            pos = PartOfSpeech(pos_str) if pos_str else None
        except ValueError:
            pos = None
        sense = Sense(
            definition=item["definition"],
            pos=pos,
            updated_by_model=model,
        )
        verdict = llm.chat_json(
            model,
            prompts.induction_critic_prompt(
                form,
                sense.definition,
                existing_defs + [s.definition for s in accepted],
            ),
            format=_CRITIC_SCHEMA,
        )
        if verdict.get("is_valid", True):
            accepted.append(sense)
        else:
            print(f"  critic rejected '{form}': {verdict.get('reason', '')}")

    alf = Alf(form=form, senses=accepted)
    Path(output).write_text(alf.model_dump_json())
    print(f"Wrote {len(accepted)} sense(s) for '{form}' to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Induce senses for a word form")
    parser.add_argument("--target", required=True, help="Path to target.json")
    parser.add_argument(
        "--seg-data-dir", required=True, help="Path to by_prefix/ directory"
    )
    parser.add_argument("--docs", required=True, help="Path to docs.parquet")
    parser.add_argument("--output", required=True, help="Output path for senses.json")
    parser.add_argument("--model", default="qwen2.5:32b")
    parser.add_argument("--context-chars", type=int, default=150)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument(
        "--senses-db", default=None, help="Path to senses.db (optional)"
    )
    parser.add_argument(
        "--labeled-db", default=None, help="Path to labeled.db (optional)"
    )
    parser.add_argument(
        "--cc-tasks-dir",
        default=None,
        help="Path to CC tasks directory (writes task file instead of calling LLM)",
    )
    args = parser.parse_args()

    cc_tasks_dir = args.cc_tasks_dir or os.environ.get("CC_TASKS_DIR")

    run(
        args.target,
        args.seg_data_dir,
        args.docs,
        args.output,
        args.model,
        args.context_chars,
        args.max_samples,
        args.senses_db,
        args.labeled_db,
        cc_tasks_dir,
    )


if __name__ == "__main__":
    main()
