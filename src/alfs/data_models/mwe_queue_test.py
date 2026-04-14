"""Tests for MWE candidate queue."""

from __future__ import annotations

from pathlib import Path

from alfs.data_models.mwe_queue import MWEQueue, MWEQueueEntry
from alfs.data_models.occurrence import Occurrence


def _entry(form: str, components: list[str] | None = None) -> MWEQueueEntry:
    return MWEQueueEntry(
        form=form,
        components=components or form.split(),
        pmi=10.0,
        corpus_count=100,
        occurrences=[Occurrence(doc_id="d1", byte_offset=0)],
    )


def test_roundtrip(tmp_path: Path):
    q = MWEQueue(tmp_path / "mwe_queue.yaml")
    entries = [_entry("a priori"), _entry("take care")]
    q.save(entries)
    loaded = q.load()
    assert len(loaded) == 2
    assert loaded[0].form == "a priori"
    assert loaded[1].form == "take care"


def test_add_candidates_dedup(tmp_path: Path):
    q = MWEQueue(tmp_path / "mwe_queue.yaml")
    assert q.add_candidates([_entry("a priori")]) == 1
    assert q.add_candidates([_entry("a priori"), _entry("take care")]) == 1
    assert len(q.load()) == 2


def test_dequeue(tmp_path: Path):
    q = MWEQueue(tmp_path / "mwe_queue.yaml")
    q.add_candidates([_entry("a priori"), _entry("take care"), _entry("ad hoc")])
    batch = q.dequeue(2)
    assert len(batch) == 2
    assert batch[0].form == "a priori"
    remaining = q.load()
    assert len(remaining) == 1
    assert remaining[0].form == "ad hoc"


def test_remove_forms(tmp_path: Path):
    q = MWEQueue(tmp_path / "mwe_queue.yaml")
    q.add_candidates([_entry("a priori"), _entry("take care")])
    q.remove_forms({"a priori"})
    remaining = q.load()
    assert len(remaining) == 1
    assert remaining[0].form == "take care"


def test_load_nonexistent(tmp_path: Path):
    q = MWEQueue(tmp_path / "missing.yaml")
    assert q.load() == []
