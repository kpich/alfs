"""Ingest Groq critic batch output into labeled.db.

For each batch response, sets last_critic_date on all reviewed instances and
downgrades rating=0 on instances the critic flagged as incorrectly labeled.

Usage:
    python -m alfs.update.labeling.critic_batch_ingest \\
        --batch-output critic_batch/critic_output.jsonl \\
        --labeled-db labeled.db \\
        [--metadata critic_batch/critic_metadata_*.jsonl] \\
        [--batch-dir critic_batch/] \\
        [--archive-dir critic_batch_archive/]
"""

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil


def parse_critic_response(content: str) -> list[int] | None:
    """Parse critic response into a list of 1-based bad indices.

    Returns None on parse failure; returns [] if the critic found no bad instances.
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None
    bad_indices = data.get("bad_indices")
    if not isinstance(bad_indices, list):
        return None
    result: list[int] = []
    for item in bad_indices:
        if isinstance(item, int):
            result.append(item)
        elif isinstance(item, float) and item == int(item):
            result.append(int(item))
    return result


def _find_metadata(batch_output: Path, batch_dir: Path) -> Path:
    """Scan critic_metadata_*.jsonl files in batch_dir to find one whose
    custom_id range contains the first custom_id in batch_output."""
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
            f"use --metadata explicitly."
        ) from None

    matches: list[Path] = []
    for meta_file in sorted(batch_dir.glob("critic_metadata_*.jsonl")):
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
            f"No critic_metadata_*.jsonl in {batch_dir} contains "
            f"custom_id {first_id!r}. Use --metadata to specify explicitly."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple metadata files match custom_id {first_id!r}: {matches}. "
            f"Use --metadata to specify explicitly."
        )
    return matches[0]


def ingest(
    batch_output: str | Path,
    labeled_db: str | Path,
    metadata_path: str | Path | None = None,
    batch_dir: str | Path | None = None,
    archive_dir: str | Path | None = None,
) -> tuple[int, int]:
    """Parse critic batch output JSONL and update labeled.db.

    Returns (n_reviewed, n_downgraded).
    """
    from alfs.data_models.occurrence_store import OccurrenceStore

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
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # (form, doc_id, byte_offset) triples grouped by critic model name
    reviewed_by_model: dict[str, list[tuple[str, str, int]]] = {}
    bad_by_model: dict[str, list[tuple[str, str, int]]] = {}
    skipped = 0

    with batch_output.open() as f:
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
            instances: list[dict[str, object]] = item["instances"]  # type: ignore[assignment]
            model_name = str(item.get("model", "unknown"))

            try:
                content = result["response"]["body"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                print(
                    f"Line {line_no}: unexpected response structure "
                    f"for {custom_id!r}, skipping"
                )
                skipped += 1
                continue

            bad_indices = parse_critic_response(str(content))
            if bad_indices is None:
                print(
                    f"Line {line_no}: could not parse critic response "
                    f"for {custom_id!r}, skipping"
                )
                skipped += 1
                continue

            # Validate indices are in range
            bad_indices_valid = [i for i in bad_indices if 1 <= i <= len(instances)]
            if len(bad_indices_valid) != len(bad_indices):
                out_of_range = [
                    i for i in bad_indices if not (1 <= i <= len(instances))
                ]
                print(
                    f"Line {line_no}: ignoring out-of-range bad_indices "
                    f"{out_of_range} for {custom_id!r} ({len(instances)} instances)"
                )

            bad_set = set(bad_indices_valid)
            reviewed_by_model.setdefault(model_name, [])
            bad_by_model.setdefault(model_name, [])
            for idx, inst in enumerate(instances, 1):
                triple = (form, str(inst["doc_id"]), int(inst["byte_offset"]))  # type: ignore[call-overload]
                reviewed_by_model[model_name].append(triple)
                if idx in bad_set:
                    bad_by_model[model_name].append(triple)

    n_reviewed = sum(len(v) for v in reviewed_by_model.values())
    n_bad = sum(len(v) for v in bad_by_model.values())
    for model_name, reviewed in reviewed_by_model.items():
        occ_store.mark_critic_reviewed(
            reviewed, timestamp, model_name, bad_by_model.get(model_name)
        )

    print(
        f"Reviewed {n_reviewed} instances, "
        f"downgraded {n_bad} to rating=0"
        + (f", skipped {skipped} responses" if skipped else "")
    )

    if archive_dir is not None:
        archive_path = Path(archive_dir)
        archive_path.mkdir(parents=True, exist_ok=True)
        shutil.move(str(batch_output), archive_path / batch_output.name)
        shutil.move(str(metadata_path), archive_path / metadata_path.name)
        critic_input_path = metadata_path.parent / metadata_path.name.replace(
            "critic_metadata_", "critic_input_"
        )
        if critic_input_path.exists():
            shutil.move(str(critic_input_path), archive_path / critic_input_path.name)
        print(f"Archived files to {archive_path}")

    return n_reviewed, n_bad


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Groq critic batch output into labeled.db"
    )
    parser.add_argument("--batch-output", required=True)
    parser.add_argument(
        "--metadata",
        default=None,
        help=(
            "Path to critic metadata JSONL "
            "(auto-discovered from --batch-dir if omitted)"
        ),
    )
    parser.add_argument(
        "--batch-dir",
        default=None,
        help=(
            "Directory to scan for matching metadata "
            "(used when --metadata is omitted)"
        ),
    )
    parser.add_argument("--labeled-db", required=True)
    parser.add_argument(
        "--archive-dir", default=None, help="Move ingested files here after success"
    )
    args = parser.parse_args()

    ingest(
        batch_output=args.batch_output,
        labeled_db=args.labeled_db,
        metadata_path=args.metadata,
        batch_dir=args.batch_dir,
        archive_dir=args.archive_dir,
    )


if __name__ == "__main__":
    main()
