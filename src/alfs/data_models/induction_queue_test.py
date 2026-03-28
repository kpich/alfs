"""Tests for InductionQueue."""

from pathlib import Path

from alfs.data_models.induction_queue import InductionQueue, InductionQueueEntry
from alfs.data_models.occurrence import Occurrence


def test_load_absent_file_returns_empty(tmp_path: Path):
    q = InductionQueue(tmp_path / "queue.yaml")
    assert q.load() == []


def test_add_forms_basic(tmp_path: Path):
    q = InductionQueue(tmp_path / "queue.yaml")
    added = q.add_forms(["cat", "dog"])
    assert added == 2
    entries = q.load()
    assert [e.form for e in entries] == ["cat", "dog"]


def test_add_forms_deduplicates(tmp_path: Path):
    q = InductionQueue(tmp_path / "queue.yaml")
    q.add_forms(["cat"])
    added = q.add_forms(["cat", "dog"])
    assert added == 1  # only dog is new
    entries = q.load()
    assert [e.form for e in entries] == ["cat", "dog"]


def test_add_forms_preserves_existing_refs(tmp_path: Path):
    q = InductionQueue(tmp_path / "queue.yaml")
    refs = [Occurrence(doc_id="doc1", byte_offset=100)]
    q.add_forms(["cat"], occs_by_form={"cat": refs})
    # Adding cat again should not overwrite refs
    q.add_forms(["cat"])
    entries = q.load()
    assert len(entries) == 1
    assert entries[0].occurrences == refs


def test_add_forms_with_occurrence_refs(tmp_path: Path):
    q = InductionQueue(tmp_path / "queue.yaml")
    refs = [Occurrence(doc_id="doc1", byte_offset=42)]
    q.add_forms(["cat"], occs_by_form={"cat": refs})
    entries = q.load()
    assert entries[0].form == "cat"
    assert entries[0].occurrences[0].doc_id == "doc1"
    assert entries[0].occurrences[0].byte_offset == 42


def test_dequeue_all_empties_file(tmp_path: Path):
    q = InductionQueue(tmp_path / "queue.yaml")
    q.add_forms(["cat", "dog"])
    entries = q.dequeue_all()
    assert len(entries) == 2
    assert q.load() == []


def test_dequeue_all_empty_queue(tmp_path: Path):
    q = InductionQueue(tmp_path / "queue.yaml")
    entries = q.dequeue_all()
    assert entries == []


def test_save_load_round_trip(tmp_path: Path):
    q = InductionQueue(tmp_path / "queue.yaml")
    original = [
        InductionQueueEntry(
            form="thrumming",
            occurrences=[Occurrence(doc_id="abc", byte_offset=999)],
        ),
        InductionQueueEntry(form="haberdasher", occurrences=[]),
    ]
    q.save(original)
    loaded = q.load()
    assert len(loaded) == 2
    assert loaded[0].form == "thrumming"
    assert loaded[0].occurrences[0].byte_offset == 999
    assert loaded[1].form == "haberdasher"
    assert loaded[1].occurrences == []


def test_remove_forms(tmp_path: Path):
    q = InductionQueue(tmp_path / "queue.yaml")
    q.add_forms(["cat", "dog", "fish"])
    q.remove_forms({"dog"})
    forms = [e.form for e in q.load()]
    assert forms == ["cat", "fish"]


def test_remove_nonexistent_forms_noop(tmp_path: Path):
    q = InductionQueue(tmp_path / "queue.yaml")
    q.add_forms(["cat"])
    q.remove_forms({"bird"})  # not in queue
    assert [e.form for e in q.load()] == ["cat"]
