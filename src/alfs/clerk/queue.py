"""File-system queue for clerk change requests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
from datetime import UTC, datetime
import logging
from pathlib import Path
import time
import traceback

from pydantic import TypeAdapter

from alfs.clerk.request import (
    ChangeRequest,
    DeleteEntryRequest,
    PruneRequest,
    TrimSenseRequest,
)
from alfs.data_models.instance_log import (
    append_delete_by_form,
    append_delete_by_sense,
)
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore

_adapter: TypeAdapter[ChangeRequest] = TypeAdapter(ChangeRequest)
logger = logging.getLogger(__name__)


def _ensure_dirs(queue_dir: Path) -> None:
    for subdir in ("pending", "processing", "done", "failed"):
        (queue_dir / subdir).mkdir(parents=True, exist_ok=True)


def enqueue(request: ChangeRequest, queue_dir: Path) -> None:
    """Serialize request to JSON, write atomically to pending/."""
    _ensure_dirs(queue_dir)
    filename = f"{request.id}.json"  # type: ignore[union-attr]
    pending_dir = queue_dir / "pending"
    tmp_path = pending_dir / f".{filename}.tmp"
    pending_path = pending_dir / filename
    data = _adapter.dump_json(request)
    tmp_path.write_bytes(data)
    tmp_path.rename(pending_path)
    logger.debug("Enqueued %s %s", request.type, request.id)  # type: ignore[union-attr]


def _claim_file(pending_path: Path, processing_dir: Path) -> Path | None:
    """Atomically claim a file from pending/ by renaming to processing/."""
    processing_path = processing_dir / pending_path.name
    try:
        pending_path.rename(processing_path)
        return processing_path
    except (FileNotFoundError, OSError):
        return None


def _process_file(
    processing_path: Path,
    done_dir: Path,
    failed_dir: Path,
    sense_store: SenseStore,
    occ_store: OccurrenceStore | None,
    log_dir: Path | None,
) -> None:
    """Parse and apply one request file; move to done/ or failed/."""
    try:
        request = _adapter.validate_json(processing_path.read_bytes())
        applied = request.apply(sense_store, occ_store)  # type: ignore[union-attr]
        processing_path.rename(done_dir / processing_path.name)
        logger.info(
            "Applied %s %s (form=%r)",
            request.type,  # type: ignore[union-attr]
            request.id,  # type: ignore[union-attr]
            request.form,  # type: ignore[union-attr]
        )
        if applied and log_dir is not None:
            now = datetime.now(UTC)
            if isinstance(request, TrimSenseRequest):
                append_delete_by_sense(
                    log_dir, request.form, request.sense_id, request.id, now
                )
            elif isinstance(request, PruneRequest):
                for sid in request.removed_ids:
                    append_delete_by_sense(log_dir, request.form, sid, request.id, now)
            elif isinstance(request, DeleteEntryRequest):
                append_delete_by_form(log_dir, request.form, request.id, now)
    except Exception:
        tb = traceback.format_exc()
        failed_path = failed_dir / processing_path.name
        processing_path.rename(failed_path)
        (failed_dir / f"{processing_path.stem}.err").write_text(tb)
        logger.error("Failed %s:\n%s", processing_path.name, tb)


def drain(
    queue_dir: Path,
    sense_store: SenseStore,
    occ_store: OccurrenceStore | None,
    workers: int = 4,
    log_dir: Path | None = None,
) -> None:
    """Process all current pending requests in a thread pool, then return."""
    _ensure_dirs(queue_dir)
    pending_dir = queue_dir / "pending"
    processing_dir = queue_dir / "processing"
    done_dir = queue_dir / "done"
    failed_dir = queue_dir / "failed"

    for orphan in sorted(processing_dir.glob("*.json")):
        with contextlib.suppress(OSError):
            orphan.rename(pending_dir / orphan.name)

    pending_files = sorted(pending_dir.glob("*.json"))
    if not pending_files:
        logger.debug("No pending requests.")
        return

    claimed: list[Path] = []
    for pf in pending_files:
        path = _claim_file(pf, processing_dir)
        if path is not None:
            claimed.append(path)

    logger.info("Processing %d requests with %d workers", len(claimed), workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _process_file, cp, done_dir, failed_dir, sense_store, occ_store, log_dir
            )
            for cp in claimed
        ]
        for future in as_completed(futures):
            future.result()


def watch(
    queue_dir: Path,
    sense_store: SenseStore,
    occ_store: OccurrenceStore | None,
    workers: int = 4,
    poll_interval: float = 1.0,
    log_dir: Path | None = None,
) -> None:
    """Continuously poll pending/ and process new requests until interrupted."""
    logger.info("Watching %s (poll_interval=%.1fs)", queue_dir, poll_interval)
    try:
        while True:
            drain(queue_dir, sense_store, occ_store, workers=workers, log_dir=log_dir)
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        logger.info("Interrupted; stopping watcher.")
