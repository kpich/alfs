"""Persistent queue of MWE candidates awaiting lexicographer review.

The queue is stored as a human-editable YAML file. Entries are deduplicated
by form name so running enqueue scripts repeatedly is idempotent.

Usage:
    from alfs.data_models.mwe_queue import MWEQueue, MWEQueueEntry

    q = MWEQueue(Path("../alfs_data/mwe_queue.yaml"))
    q.add_candidates([MWEQueueEntry(form="a priori", ...)])
    entries = q.dequeue(10)
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
import yaml

from alfs.data_models.occurrence import Occurrence


class MWEQueueEntry(BaseModel):
    form: str
    components: list[str]
    pmi: float
    corpus_count: int
    occurrences: list[Occurrence] = []


class MWEQueue:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> list[MWEQueueEntry]:
        if not self._path.exists():
            return []
        raw = yaml.safe_load(self._path.read_text()) or []
        return [MWEQueueEntry.model_validate(item) for item in raw]

    def save(self, entries: list[MWEQueueEntry]) -> None:
        data = [e.model_dump(mode="json") for e in entries]
        tmp = self._path.with_suffix(".yaml.tmp")
        tmp.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False))
        tmp.rename(self._path)

    def add_candidates(self, candidates: list[MWEQueueEntry]) -> int:
        """Append candidates not already in the queue. Returns count newly added."""
        existing = self.load()
        existing_forms = {e.form for e in existing}
        added = 0
        for c in candidates:
            if c.form in existing_forms:
                continue
            existing.append(c)
            existing_forms.add(c.form)
            added += 1
        if added:
            self.save(existing)
        return added

    def dequeue(self, limit: int) -> list[MWEQueueEntry]:
        """Read up to limit entries, save remainder back, return dequeued."""
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
