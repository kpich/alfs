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
import shutil

from alfs.data_models.annotated_occurrence import OccurrenceRating
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore
from alfs.update.labeling.label_occurrences import build_sense_menu


def parse_response(content: str) -> dict[str, object] | None:
    """Parse LLM response JSON into {sense_key, rating, synonyms}. Returns None on
    error."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None
    if "sense_key" not in data or "rating" not in data:
        return None
    raw_synonyms = data.get("synonyms")
    synonyms: list[str] | None = (
        [str(s) for s in raw_synonyms] if isinstance(raw_synonyms, list) else None
    )
    return {
        "sense_key": data["sense_key"],
        "rating": data["rating"],
        "synonyms": synonyms,
    }


def _find_metadata(batch_output: Path, batch_dir: Path) -> Path:
    """Scan batch_metadata_*.jsonl files in batch_dir to find one whose custom_id
    range contains the first custom_id in batch_output.

    Raises ValueError if no match or ambiguous.
    """
    # Read the first custom_id from the output file (may not be '0' if some
    # requests were dropped by Groq due to validation errors).
    first_id: str | None = None
    with batch_output.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                first_id = str(obj.get("custom_id", ""))
                break
            except json.JSONDecodeError:
                continue
    if first_id is None:
        raise ValueError(f"Could not read any custom_id from {batch_output}")

    try:
        first_id_int = int(first_id)
    except ValueError:
        raise ValueError(
            f"custom_id {first_id!r} in {batch_output} is not an integer; "
            f"cannot auto-discover metadata. Use --metadata explicitly."
        ) from None

    matches: list[Path] = []
    for meta_file in sorted(batch_dir.glob("batch_metadata_*.jsonl")):
        # Read first and last line to get the custom_id range for this chunk.
        try:
            with meta_file.open() as f:
                first_line = f.readline().strip()
                last_line = first_line
                for last_line in f:
                    last_line = last_line.strip()
            if not first_line:
                continue
            lo = int(json.loads(first_line)["custom_id"])
            hi = int(json.loads(last_line)["custom_id"])
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
        if lo <= first_id_int <= hi:
            matches.append(meta_file)

    if not matches:
        raise ValueError(
            f"No batch_metadata_*.jsonl in {batch_dir} contains "
            f"custom_id {first_id!r}. "
            f"Make sure the metadata file is in --batch-dir."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple metadata files match custom_id {first_id!r}: {matches}. "
            f"Use --metadata to specify explicitly."
        )
    return matches[0]


def ingest(
    batch_output: str | Path,
    senses_db: str | Path,
    labeled_db: str | Path,
    metadata_path: str | Path | None = None,
    log_dir: str | Path | None = None,
    batch_dir: str | Path | None = None,
    archive_dir: str | Path | None = None,
) -> int:
    """Parse Groq batch output JSONL and upsert results into labeled.db.

    If metadata_path is None, batch_dir must be set and the matching metadata
    file is auto-discovered by custom_id.

    If archive_dir is set, moves batch_output, metadata, and matching batch_input
    (if present) into archive_dir after successful ingest.

    Returns the number of rows inserted.
    """
    batch_output = Path(batch_output)
    if metadata_path is None:
        if batch_dir is None:
            raise ValueError("Either --metadata or --batch-dir must be provided")
        metadata_path = _find_metadata(batch_output, Path(batch_dir))
        print(f"Auto-discovered metadata: {metadata_path}")
    else:
        metadata_path = Path(metadata_path)

    meta: dict[str, dict[str, object]] = {}
    with metadata_path.open() as f:
        for line in f:
            row = json.loads(line.strip())
            meta[row["custom_id"]] = row

    occ_store = OccurrenceStore(Path(labeled_db))
    sense_store = SenseStore(Path(senses_db))  # only needed for forms without key_map

    upsert_rows: list[
        tuple[str, tuple[str, str, int, str, int, list[str] | None]]
    ] = []  # (model, row)
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
                # Prefer key_map stored in metadata (snapshot at prepare time).
                # Fall back to rebuilding from current sense store for old metadata
                # files that predate this field.
                stored = item.get("key_map")
                if stored is not None:
                    assert isinstance(stored, dict)
                    key_map: dict[str, str] = {
                        str(k): str(v) for k, v in stored.items()
                    }
                else:
                    try:
                        _, key_map = build_sense_menu(sense_store, form)
                    except ValueError:
                        print(f"Line {line_no}: no sense menu for {form!r}, skipping")
                        skipped += 1
                        continue
                if display_key not in key_map:
                    print(
                        f"Line {line_no}: sense key {display_key!r} not in menu "
                        f"for {form!r}, skipping"
                    )
                    skipped += 1
                    continue
                uuid_key = key_map[display_key]

            synonyms = parsed.get("synonyms")
            model_name = str(item.get("model", "unknown"))
            upsert_rows.append(
                (
                    model_name,
                    (form, doc_id, byte_offset, uuid_key, rating.value, synonyms),
                )  # type: ignore[arg-type]
            )

    if upsert_rows:
        rows_by_model: dict[
            str, list[tuple[str, str, int, str, int, list[str] | None]]
        ] = {}
        for model_name, row in upsert_rows:
            rows_by_model.setdefault(model_name, []).append(row)
        for model_name, model_rows in rows_by_model.items():
            occ_store.upsert_many(model_rows, model=model_name)
            if log_dir is not None:
                from alfs.data_models.instance_log import append_upserts

                append_upserts(Path(log_dir), model_rows, model=model_name)

    print(f"Ingested {len(upsert_rows)} rows, skipped {skipped}")

    if archive_dir is not None:
        archive_path = Path(archive_dir)
        archive_path.mkdir(parents=True, exist_ok=True)
        shutil.move(str(batch_output), archive_path / batch_output.name)
        shutil.move(str(metadata_path), archive_path / metadata_path.name)
        # Also move matching batch_input if it exists alongside the metadata
        batch_input_path = metadata_path.parent / metadata_path.name.replace(
            "batch_metadata_", "batch_input_"
        )
        if batch_input_path.exists():
            shutil.move(str(batch_input_path), archive_path / batch_input_path.name)
        print(f"Archived files to {archive_path}")

    return len(upsert_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Groq batch output into labeled.db"
    )
    parser.add_argument("--batch-output", required=True)
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to metadata JSONL (auto-discovered from --batch-dir if omitted)",
    )
    parser.add_argument(
        "--batch-dir",
        default=None,
        help=(
            "Directory to scan for matching metadata"
            " (used when --metadata is omitted)"
        ),
    )
    parser.add_argument("--senses-db", required=True)
    parser.add_argument("--labeled-db", required=True)
    parser.add_argument(
        "--log-dir", default=None, help="Directory for instance-tagging change log"
    )
    parser.add_argument(
        "--archive-dir", default=None, help="Move ingested files here after success"
    )
    args = parser.parse_args()

    ingest(
        batch_output=args.batch_output,
        senses_db=args.senses_db,
        labeled_db=args.labeled_db,
        metadata_path=args.metadata,
        log_dir=args.log_dir,
        batch_dir=args.batch_dir,
        archive_dir=args.archive_dir,
    )


if __name__ == "__main__":
    main()
