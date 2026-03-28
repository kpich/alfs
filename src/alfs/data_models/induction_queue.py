"""Persistent queue of word forms awaiting sense induction.

The queue is stored as a human-editable YAML file. Entries are deduplicated
by form name so running enqueue scripts repeatedly is idempotent.

Usage:
    from alfs.data_models.induction_queue import InductionQueue, InductionQueueEntry
    from alfs.data_models.occurrence import Occurrence

    q = InductionQueue(Path("../alfs_data/induction_queue.yaml"))
    q.add_forms(["running", "thrumming"])
    entries = q.dequeue_all()
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
import yaml

from alfs.data_models.occurrence import Occurrence


class InductionQueueEntry(BaseModel):
    form: str
    occurrences: list[Occurrence] = []  # empty = sample freely during induction


class InductionQueue:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> list[InductionQueueEntry]:
        """Read the queue; returns empty list if file absent."""
        if not self._path.exists():
            return []
        raw = yaml.safe_load(self._path.read_text()) or []
        return [InductionQueueEntry.model_validate(item) for item in raw]

    def save(self, entries: list[InductionQueueEntry]) -> None:
        """Atomically write entries to the queue file."""
        data = [e.model_dump(mode="json") for e in entries]
        tmp = self._path.with_suffix(".yaml.tmp")
        tmp.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False))
        tmp.rename(self._path)

    def add_forms(
        self,
        forms: list[str],
        occs_by_form: dict[str, list[Occurrence]] | None = None,
    ) -> int:
        """Append forms not already in the queue. Returns count newly added."""
        existing = self.load()
        existing_forms = {e.form for e in existing}
        added = 0
        for form in forms:
            if form in existing_forms:
                continue
            occs = (occs_by_form or {}).get(form, [])
            existing.append(InductionQueueEntry(form=form, occurrences=occs))
            existing_forms.add(form)
            added += 1
        if added:
            self.save(existing)
        return added

    def dequeue_all(self) -> list[InductionQueueEntry]:
        """Read all entries and clear the file atomically. Returns entries."""
        entries = self.load()
        if entries:
            self.save([])
        return entries

    def dequeue(self, limit: int) -> list[InductionQueueEntry]:
        """Read up to limit entries, save remainder back, return dequeued entries."""
        entries = self.load()
        batch = entries[:limit]
        if batch:
            self.save(entries[limit:])
        return batch

    def remove_forms(self, forms: set[str]) -> None:
        """Remove specific forms from the queue."""
        entries = self.load()
        filtered = [e for e in entries if e.form not in forms]
        if len(filtered) != len(entries):
            self.save(filtered)
