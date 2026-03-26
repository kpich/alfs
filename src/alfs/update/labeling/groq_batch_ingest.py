"""Ingest Groq batch output into labeled.db.

Usage:
    python -m alfs.update.labeling.groq_batch_ingest \\
        --batch-output groq_batch/batch_output.jsonl \\
        --metadata groq_batch/batch_metadata.jsonl \\
        --senses-db senses.db --labeled-db labeled.db
"""

import argparse
import json
from pathlib import Path

from alfs.data_models.annotated_occurrence import OccurrenceRating
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore
from alfs.update.labeling.label_occurrences import build_sense_menu


def parse_response(content: str) -> dict[str, object] | None:
    """Parse LLM response JSON into {sense_key, rating}. Returns None on error."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None
    if "sense_key" not in data or "rating" not in data:
        return None
    return data


def ingest(
    batch_output: str | Path,
    metadata_path: str | Path,
    senses_db: str | Path,
    labeled_db: str | Path,
) -> int:
    """Parse Groq batch output JSONL and upsert results into labeled.db.

    Returns the number of rows inserted.
    """
    meta: dict[str, dict[str, object]] = {}
    with Path(metadata_path).open() as f:
        for line in f:
            row = json.loads(line.strip())
            meta[row["custom_id"]] = row

    sense_store = SenseStore(Path(senses_db))
    occ_store = OccurrenceStore(Path(labeled_db))

    key_map_cache: dict[str, dict[str, str]] = {}
    upsert_rows: list[tuple[str, str, int, str, int]] = []
    skipped = 0

    with Path(batch_output).open() as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                print(f"Line {line_no}: malformed JSON, skipping")
                skipped += 1
                continue

            custom_id = str(result.get("custom_id", ""))
            if custom_id not in meta:
                print(f"Line {line_no}: unknown custom_id {custom_id!r}, skipping")
                skipped += 1
                continue

            item = meta[custom_id]
            form = str(item["form"])
            doc_id = str(item["doc_id"])
            byte_offset = int(item["byte_offset"])  # type: ignore[call-overload]

            try:
                content = result["response"]["body"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                print(
                    f"Line {line_no}: unexpected response structure "
                    f"for {custom_id!r}, skipping"
                )
                skipped += 1
                continue

            parsed = parse_response(str(content))
            if parsed is None:
                print(
                    f"Line {line_no}: could not parse response for {custom_id!r}, "
                    "skipping"
                )
                skipped += 1
                continue

            display_key = str(parsed["sense_key"])

            try:
                rating = OccurrenceRating(int(parsed["rating"]))  # type: ignore[call-overload]
            except (ValueError, TypeError):
                print(
                    f"Line {line_no}: invalid rating {parsed['rating']!r} "
                    f"for {custom_id!r}, skipping"
                )
                skipped += 1
                continue

            if display_key == "0":
                uuid_key = "0"
            else:
                if form not in key_map_cache:
                    try:
                        _, key_map = build_sense_menu(sense_store, form)
                    except ValueError:
                        print(f"Line {line_no}: no sense menu for {form!r}, skipping")
                        skipped += 1
                        continue
                    key_map_cache[form] = key_map
                key_map = key_map_cache[form]
                if display_key not in key_map:
                    print(
                        f"Line {line_no}: sense key {display_key!r} not in menu "
                        f"for {form!r}, skipping"
                    )
                    skipped += 1
                    continue
                uuid_key = key_map[display_key]

            upsert_rows.append((form, doc_id, byte_offset, uuid_key, rating.value))

    if upsert_rows:
        occ_store.upsert_many(upsert_rows)

    print(f"Ingested {len(upsert_rows)} rows, skipped {skipped}")
    return len(upsert_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Groq batch output into labeled.db"
    )
    parser.add_argument("--batch-output", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--senses-db", required=True)
    parser.add_argument("--labeled-db", required=True)
    args = parser.parse_args()

    ingest(
        batch_output=args.batch_output,
        metadata_path=args.metadata,
        senses_db=args.senses_db,
        labeled_db=args.labeled_db,
    )


if __name__ == "__main__":
    main()
