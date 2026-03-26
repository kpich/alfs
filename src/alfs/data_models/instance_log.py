"""Append-only change log for instance (occurrence) labeling mutations.

Log files are monthly JSONL: {log_dir}/YYYY-MM.jsonl

Three event types:
  upsert          — occurrence labeled or re-labeled
  delete_by_sense — all occurrences for a sense deleted (TrimSenseRequest /
                    PruneRequest)
  delete_by_form  — all occurrences for a form deleted (DeleteEntryRequest)
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
import json
from pathlib import Path
import threading

_lock = threading.Lock()


def _month_path(log_dir: Path, dt: datetime) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{dt:%Y-%m}.jsonl"


def _write(log_dir: Path, dt: datetime, lines: list[str]) -> None:
    path = _month_path(log_dir, dt)
    with _lock, path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def append_upserts(
    log_dir: Path,
    rows: Iterable[tuple[str, str, int, str, int, list[str] | None]],
    model: str,
    at: datetime | None = None,
) -> None:
    """Append one upsert event per row to the monthly JSONL log.

    Each row is (form, doc_id, byte_offset, sense_key, rating, synonyms).
    A single timestamp is used for the entire batch.
    """
    now = at if at is not None else datetime.now(UTC)
    at_str = now.isoformat()
    lines = [
        json.dumps(
            {
                "type": "upsert",
                "form": form,
                "doc_id": doc_id,
                "byte_offset": byte_offset,
                "sense_key": sense_key,
                "rating": rating,
                "synonyms": synonyms,
                "model": model,
                "at": at_str,
            },
            ensure_ascii=False,
        )
        for form, doc_id, byte_offset, sense_key, rating, synonyms in rows
    ]
    if lines:
        _write(log_dir, now, lines)


def append_delete_by_sense(
    log_dir: Path,
    form: str,
    sense_id: str,
    request_id: str,
    at: datetime | None = None,
) -> None:
    """Append a delete_by_sense event (all occurrences for one sense deleted)."""
    now = at if at is not None else datetime.now(UTC)
    line = json.dumps(
        {
            "type": "delete_by_sense",
            "form": form,
            "sense_id": sense_id,
            "request_id": request_id,
            "at": now.isoformat(),
        },
        ensure_ascii=False,
    )
    _write(log_dir, now, [line])


def append_delete_by_form(
    log_dir: Path,
    form: str,
    request_id: str,
    at: datetime | None = None,
) -> None:
    """Append a delete_by_form event (all occurrences for a form deleted)."""
    now = at if at is not None else datetime.now(UTC)
    line = json.dumps(
        {
            "type": "delete_by_form",
            "form": form,
            "request_id": request_id,
            "at": now.isoformat(),
        },
        ensure_ascii=False,
    )
    _write(log_dir, now, [line])
