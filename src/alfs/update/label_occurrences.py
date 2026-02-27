"""Label occurrences with sense keys and ratings using Ollama.

Usage:
    python -m alfs.update.label_occurrences \\
        --target target.json --seg-data-dir by_prefix/ --docs docs.parquet \\
        --alfs alfs.json --output {form}_labeled.parquet \\
        --model llama3.1:8b --context-chars 150 [--labeled labeled.parquet]
"""

import argparse
from pathlib import Path

import polars as pl

from alfs.data_models.alf import Alfs, sense_key
from alfs.data_models.annotated_occurrence import AnnotatedOccurrence, OccurrenceRating
from alfs.data_models.update_target import UpdateTarget
from alfs.update import llm, prompts


def extract_context(text: str, byte_offset: int, form: str, context_chars: int) -> str:
    char_offset = len(text.encode()[:byte_offset].decode())
    start = max(0, char_offset - context_chars)
    end = char_offset + len(form) + context_chars
    return text[start:end]


def build_sense_menu(alfs: Alfs, form: str) -> str:
    alf = alfs.entries[form]
    lines = []
    for i, sense in enumerate(alf.senses):
        lines.append(f"{i + 1}. {sense.definition}")
        for j, sub in enumerate(sense.subsenses):
            sub_key = sense_key(i, j)
            lines.append(f"   {sub_key}. {sub}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label occurrences with sense keys and ratings"
    )
    parser.add_argument("--target", required=True)
    parser.add_argument("--seg-data-dir", required=True)
    parser.add_argument("--docs", required=True)
    parser.add_argument("--alfs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--context-chars", type=int, default=150)
    parser.add_argument(
        "--max-occurrences", type=int, default=100
    )  # TODO: artificially low for dev
    parser.add_argument("--labeled", default=None, help="Path to labeled.parquet")
    args = parser.parse_args()

    target = UpdateTarget.model_validate_json(Path(args.target).read_text())
    form = target.form

    alfs = Alfs.model_validate_json(Path(args.alfs).read_text())
    if form not in alfs.entries:
        raise ValueError(f"No entry for '{form}' in alfs.json")

    docs_df = pl.read_parquet(args.docs)
    docs = dict(
        zip(docs_df["doc_id"].to_list(), docs_df["text"].to_list(), strict=False)
    )

    prefix = form[0].lower() if form and form[0].lower().isalpha() else "other"
    occ_path = Path(args.seg_data_dir) / prefix / "occurrences.parquet"
    df = pl.read_parquet(str(occ_path)).filter(pl.col("form") == form)

    labeled_pairs: set[tuple[str, int]] = set()
    if args.labeled and Path(args.labeled).exists():
        ldf = pl.read_parquet(args.labeled).filter(pl.col("form") == form)
        for row in ldf.select(["doc_id", "byte_offset"]).iter_rows():
            labeled_pairs.add((row[0], row[1]))

    sense_menu = build_sense_menu(alfs, form)

    # TODO: batch occurrences into a single prompt instead of one LLM call
    # per occurrence
    results: list[dict] = []
    for occ in df.to_dicts():
        if len(results) >= args.max_occurrences:
            break

        doc_id = occ["doc_id"]
        byte_offset = occ["byte_offset"]

        if (doc_id, byte_offset) in labeled_pairs:
            continue

        text = docs.get(doc_id, "")
        if not text:
            continue

        context = extract_context(text, byte_offset, form, args.context_chars)
        prompt = prompts.labeling_prompt(form, context, sense_menu)
        data = llm.chat_json(args.model, prompt)
        ann = AnnotatedOccurrence(
            doc_id=doc_id,
            byte_offset=byte_offset,
            sense_key=data["sense_key"],
            rating=OccurrenceRating(data["rating"]),
        )
        results.append({"form": form, **ann.model_dump()})

    if results:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(results).write_parquet(str(out_path))
        print(f"Labeled {len(results)} occurrences for '{form}' → {args.output}")
    else:
        print(f"No new occurrences to label for '{form}'")


if __name__ == "__main__":
    main()
