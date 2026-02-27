"""Induce senses for a word form using Ollama.

Usage:
    python -m alfs.update.induce_senses \\
        --target target.json --seg-data-dir by_prefix/ --docs docs.parquet \\
        --output senses.json --model llama3.1:8b --context-chars 150 --max-samples 20 \\
        [--alfs alfs.json]
"""

import argparse
from pathlib import Path
import random

import polars as pl

from alfs.data_models.alf import Alf, Alfs, Sense
from alfs.data_models.update_target import UpdateTarget
from alfs.update import llm, prompts


def extract_context(text: str, byte_offset: int, form: str, context_chars: int) -> str:
    char_offset = len(text.encode()[:byte_offset].decode())
    start = max(0, char_offset - context_chars)
    end = char_offset + len(form) + context_chars
    return text[start:end]


def main() -> None:
    parser = argparse.ArgumentParser(description="Induce senses for a word form")
    parser.add_argument("--target", required=True, help="Path to target.json")
    parser.add_argument(
        "--seg-data-dir", required=True, help="Path to by_prefix/ directory"
    )
    parser.add_argument("--docs", required=True, help="Path to docs.parquet")
    parser.add_argument("--output", required=True, help="Output path for senses.json")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--context-chars", type=int, default=150)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--alfs", default=None, help="Path to alfs.json (optional)")
    args = parser.parse_args()

    target = UpdateTarget.model_validate_json(Path(args.target).read_text())
    form = target.form

    existing_defs: list[str] = []
    if args.alfs and Path(args.alfs).exists():
        alfs = Alfs.model_validate_json(Path(args.alfs).read_text())
        entry = alfs.entries.get(form)
        if entry:
            existing_defs = [s.definition for s in entry.senses]

    docs_df = pl.read_parquet(args.docs)
    docs = dict(
        zip(docs_df["doc_id"].to_list(), docs_df["text"].to_list(), strict=False)
    )

    prefix = form[0].lower() if form and form[0].lower().isalpha() else "other"
    occ_path = Path(args.seg_data_dir) / prefix / "occurrences.parquet"
    df = pl.read_parquet(str(occ_path)).filter(pl.col("form") == form)
    occurrences = df.select(["doc_id", "byte_offset"]).iter_rows(named=True)
    all_occurrences = list(occurrences)

    random.seed(42)
    samples = random.sample(
        all_occurrences, min(args.max_samples, len(all_occurrences))
    )

    contexts = []
    for occ in samples:
        text = docs.get(occ["doc_id"], "")
        if not text:
            continue
        ctx = extract_context(text, occ["byte_offset"], form, args.context_chars)
        contexts.append(ctx)

    if not contexts:
        raise ValueError(f"No contexts found for form '{form}'")

    prompt = prompts.induction_prompt(form, contexts, existing_defs)
    data = llm.chat_json(args.model, prompt)

    # Extract the sense object from various response shapes
    if "sense" in data:
        s = data["sense"]
    elif "senses" in data and data["senses"]:
        s = data["senses"][0]
    else:
        s = data

    # Extract definition — handle string sense or dicts with alternate key names
    if isinstance(s, str):
        definition = s
        subsenses: list = []
    elif "definition" in s:
        definition = s["definition"]
        subsenses = s.get("subsenses", [])
    else:
        for key in ("meaning", "description", "def", "gloss"):
            if key in s:
                definition = s[key]
                subsenses = s.get("subsenses", [])
                break
        else:
            raise ValueError(f"Could not find definition in LLM response: {data!r}")

    sense = Sense(
        definition=definition,
        subsenses=[
            sub if isinstance(sub, str) else sub.get("definition", str(sub))
            for sub in subsenses
        ],
    )
    alf = Alf(form=form, senses=[sense])

    Path(args.output).write_text(alf.model_dump_json())
    print(f"Wrote 1 sense for '{form}' to {args.output}")


if __name__ == "__main__":
    main()
