"""Label occurrences with sense keys and ratings using Ollama.

Usage:
    python -m alfs.update.labeling.label_occurrences \\
        --target target.json --seg-data-dir by_prefix/ --docs docs.parquet \\
        --senses-db senses.db --labeled-db labeled.db \\
        --model gemma2:9b --context-chars 150
"""

import argparse
from pathlib import Path

import polars as pl

from alfs.data_models.alf import Alf, sense_key
from alfs.data_models.annotated_occurrence import AnnotatedOccurrence, OccurrenceRating
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore
from alfs.data_models.update_target import UpdateTarget
from alfs.seg.aggregate_occurrences import prefix as form_prefix
from alfs.update import llm
from alfs.update.labeling import prompts

_LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "sense_key": {"type": "string"},
        "rating": {"type": "integer"},
    },
    "required": ["sense_key", "rating"],
}


def extract_context(text: str, byte_offset: int, form: str, context_chars: int) -> str:
    char_offset = len(text.encode()[:byte_offset].decode())
    start = max(0, char_offset - context_chars)
    end = char_offset + len(form) + context_chars
    return text[start:end]


def build_sense_menu(store: SenseStore, form: str) -> tuple[str, dict[str, str]]:
    """Return (menu_text, display_to_uuid_map).

    display_to_uuid_map maps display keys like "1", "2a" to UUID-based sense keys
    that are written to labeled.db.
    """
    alf = store.read(form)
    if alf is None:
        raise ValueError(f"No entry for '{form}' in senses.db")
    menu_form = alf.redirect if alf.redirect is not None else form
    target_alf: Alf | None
    if menu_form == form:
        target_alf = alf
    else:
        target_alf = store.read(menu_form)
    if target_alf is None:
        raise ValueError(
            f"Redirect target '{menu_form}' for '{form}' not found in senses.db"
        )
    lines = []
    key_map: dict[str, str] = {}
    for i, sense in enumerate(target_alf.senses):
        display = str(i + 1)
        key_map[display] = sense.id
        pos_tag = f" [{sense.pos.value}]" if sense.pos else ""
        lines.append(f"{i + 1}.{pos_tag} {sense.definition}")
        for j, sub in enumerate(sense.subsenses or []):
            sub_key = sense_key(i, j)
            key_map[sub_key] = sense.id + chr(ord("a") + j)
            lines.append(f"   {sub_key}. {sub}")
    return "\n".join(lines), key_map


def run(
    target_file: str | Path,
    seg_data_dir: str | Path,
    docs: str | Path,
    senses_db: str | Path,
    labeled_db: str | Path,
    model: str = "gemma2:9b",
    context_chars: int = 150,
    max_occurrences: int = 100,
) -> None:
    target = UpdateTarget.model_validate_json(Path(target_file).read_text())
    form = target.form

    sense_store = SenseStore(Path(senses_db))
    occ_store = OccurrenceStore(Path(labeled_db))

    entry = sense_store.read(form)
    if entry is None or (not entry.senses and entry.redirect is None):
        print(f"No senses for '{form}' in senses.db; skipping labeling.")
        return

    sense_menu, key_map = build_sense_menu(sense_store, form)

    occ_path = Path(seg_data_dir) / form_prefix(form) / "occurrences.parquet"
    if not occ_path.exists():
        print(f"No occurrences parquet for '{form}' ({occ_path}); skipping labeling.")
        return
    df = pl.read_parquet(str(occ_path)).filter(pl.col("form") == form)

    labeled_pairs: set[tuple[str, int]] = set()
    existing = occ_store.query_form(form).filter(pl.col("rating").is_in([2, 3]))
    for row in existing.select(["doc_id", "byte_offset"]).iter_rows():
        labeled_pairs.add((row[0], row[1]))

    to_process = [
        occ
        for occ in df.to_dicts()
        if (occ["doc_id"], occ["byte_offset"]) not in labeled_pairs
    ][:max_occurrences]

    needed_doc_ids = list({occ["doc_id"] for occ in to_process})
    docs_df = (
        pl.scan_parquet(str(docs))
        .filter(pl.col("doc_id").is_in(needed_doc_ids))
        .collect(engine="streaming")
    )
    docs_map = dict(
        zip(docs_df["doc_id"].to_list(), docs_df["text"].to_list(), strict=False)
    )

    # TODO: batch occurrences into a single prompt instead of one LLM call
    # per occurrence
    upsert_rows: list[tuple[str, str, int, str, int]] = []
    for occ in to_process:
        doc_id = occ["doc_id"]
        byte_offset = occ["byte_offset"]

        text = docs_map.get(doc_id, "")
        if not text:
            continue

        context = extract_context(text, byte_offset, form, context_chars)
        prompt = prompts.labeling_prompt(form, context, sense_menu)
        data = llm.chat_json(model, prompt, format=_LABEL_SCHEMA)
        display_key = data["sense_key"]
        uuid_key = key_map.get(display_key, display_key)
        ann = AnnotatedOccurrence(
            doc_id=doc_id,
            byte_offset=byte_offset,
            sense_key=uuid_key,
            rating=OccurrenceRating(data["rating"]),
        )
        upsert_rows.append(
            (form, ann.doc_id, ann.byte_offset, ann.sense_key, ann.rating.value)
        )

    if upsert_rows:
        occ_store.upsert_many(upsert_rows)
        print(f"Labeled {len(upsert_rows)} occurrences for '{form}' → labeled.db")
    else:
        print(f"No new occurrences to label for '{form}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label occurrences with sense keys and ratings"
    )
    parser.add_argument("--target", required=True)
    parser.add_argument("--seg-data-dir", required=True)
    parser.add_argument("--docs", required=True)
    parser.add_argument("--senses-db", required=True, help="Path to senses.db")
    parser.add_argument("--labeled-db", required=True, help="Path to labeled.db")
    parser.add_argument("--model", default="gemma2:9b")
    parser.add_argument("--context-chars", type=int, default=150)
    parser.add_argument(
        "--max-occurrences", type=int, default=100
    )  # TODO: artificially low for dev
    args = parser.parse_args()

    run(
        args.target,
        args.seg_data_dir,
        args.docs,
        args.senses_db,
        args.labeled_db,
        args.model,
        args.context_chars,
        args.max_occurrences,
    )


if __name__ == "__main__":
    main()
