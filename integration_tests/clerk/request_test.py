"""Integration tests for clerk request apply() methods."""

from datetime import datetime
from pathlib import Path

from alfs.clerk.request import PruneRequest, RewriteRequest, TrimSenseRequest
from alfs.data_models.alf import Alf, Sense
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore


def _sense_store(tmp_path: Path) -> SenseStore:
    return SenseStore(tmp_path / "senses.db")


def _occ_store(tmp_path: Path) -> OccurrenceStore:
    return OccurrenceStore(tmp_path / "labeled.db")


def _make_request_id() -> str:
    return "test-request-id"


# --- TrimSenseRequest ---


def test_trim_sense_removes_sense_from_senses_db(tmp_path: Path) -> None:
    store = _sense_store(tmp_path)
    sense_a = Sense(definition="sense A")
    sense_b = Sense(definition="sense B")
    store.write(Alf(form="word", senses=[sense_a, sense_b]))

    request = TrimSenseRequest(
        id=_make_request_id(),
        created_at=datetime.utcnow(),
        form="word",
        before=[sense_a, sense_b],
        after=[sense_b],
        sense_id=sense_a.id,
        reason="test",
    )
    request.apply(store, None)

    result = store.read("word")
    assert result is not None
    assert len(result.senses) == 1
    assert result.senses[0].id == sense_b.id


def test_trim_sense_deletes_labeled_rows_for_removed_sense(tmp_path: Path) -> None:
    store = _sense_store(tmp_path)
    occ = _occ_store(tmp_path)
    sense_a = Sense(definition="sense A")
    sense_b = Sense(definition="sense B")
    store.write(Alf(form="word", senses=[sense_a, sense_b]))

    occ.upsert_many(
        [
            ("word", "doc1", 0, sense_a.id, 2),
            ("word", "doc1", 100, sense_b.id, 3),
        ]
    )

    request = TrimSenseRequest(
        id=_make_request_id(),
        created_at=datetime.utcnow(),
        form="word",
        before=[sense_a, sense_b],
        after=[sense_b],
        sense_id=sense_a.id,
        reason="test",
    )
    request.apply(store, occ)

    df = occ.query_form("word")
    assert len(df) == 1
    assert df["sense_key"][0] == sense_b.id


def test_trim_sense_preserves_other_sense_ids(tmp_path: Path) -> None:
    store = _sense_store(tmp_path)
    sense_a = Sense(definition="sense A")
    sense_b = Sense(definition="sense B")
    sense_c = Sense(definition="sense C")
    store.write(Alf(form="word", senses=[sense_a, sense_b, sense_c]))

    request = TrimSenseRequest(
        id=_make_request_id(),
        created_at=datetime.utcnow(),
        form="word",
        before=[sense_a, sense_b, sense_c],
        after=[sense_a, sense_c],
        sense_id=sense_b.id,
        reason="test",
    )
    request.apply(store, None)

    result = store.read("word")
    assert result is not None
    ids = [s.id for s in result.senses]
    assert ids == [sense_a.id, sense_c.id]


# --- PruneRequest ---


def test_prune_removes_multiple_senses_from_senses_db(tmp_path: Path) -> None:
    store = _sense_store(tmp_path)
    sense_a = Sense(definition="sense A")
    sense_b = Sense(definition="sense B")
    sense_c = Sense(definition="sense C")
    store.write(Alf(form="word", senses=[sense_a, sense_b, sense_c]))

    request = PruneRequest(
        id=_make_request_id(),
        created_at=datetime.utcnow(),
        form="word",
        before=[sense_a, sense_b, sense_c],
        after=[sense_a],
        removed_ids=[sense_b.id, sense_c.id],
    )
    request.apply(store, None)

    result = store.read("word")
    assert result is not None
    assert len(result.senses) == 1
    assert result.senses[0].id == sense_a.id


def test_prune_deletes_labeled_rows_for_removed_senses(tmp_path: Path) -> None:
    store = _sense_store(tmp_path)
    occ = _occ_store(tmp_path)
    sense_a = Sense(definition="sense A")
    sense_b = Sense(definition="sense B")
    sense_c = Sense(definition="sense C")
    store.write(Alf(form="word", senses=[sense_a, sense_b, sense_c]))

    occ.upsert_many(
        [
            ("word", "doc1", 0, sense_a.id, 2),
            ("word", "doc1", 100, sense_b.id, 3),
            ("word", "doc1", 200, sense_c.id, 1),
        ]
    )

    request = PruneRequest(
        id=_make_request_id(),
        created_at=datetime.utcnow(),
        form="word",
        before=[sense_a, sense_b, sense_c],
        after=[sense_a],
        removed_ids=[sense_b.id, sense_c.id],
    )
    request.apply(store, occ)

    df = occ.query_form("word")
    assert len(df) == 1
    assert df["sense_key"][0] == sense_a.id


def test_prune_surviving_senses_keep_original_ids(tmp_path: Path) -> None:
    store = _sense_store(tmp_path)
    sense_a = Sense(definition="sense A")
    sense_b = Sense(definition="sense B")
    store.write(Alf(form="word", senses=[sense_a, sense_b]))

    request = PruneRequest(
        id=_make_request_id(),
        created_at=datetime.utcnow(),
        form="word",
        before=[sense_a, sense_b],
        after=[sense_a],
        removed_ids=[sense_b.id],
    )
    request.apply(store, None)

    result = store.read("word")
    assert result is not None
    assert result.senses[0].id == sense_a.id


# --- RewriteRequest ---


def test_rewrite_updates_sense_definitions(tmp_path: Path) -> None:
    store = _sense_store(tmp_path)
    sense_a = Sense(definition="old definition A")
    sense_b = Sense(definition="old definition B")
    store.write(Alf(form="word", senses=[sense_a, sense_b]))

    new_a = sense_a.model_copy(update={"definition": "new definition A"})
    new_b = sense_b.model_copy(update={"definition": "new definition B"})

    request = RewriteRequest(
        id=_make_request_id(),
        created_at=datetime.utcnow(),
        form="word",
        before=[sense_a, sense_b],
        after=[new_a, new_b],
    )
    request.apply(store, None)

    result = store.read("word")
    assert result is not None
    assert result.senses[0].definition == "new definition A"
    assert result.senses[1].definition == "new definition B"


def test_rewrite_preserves_sense_ids(tmp_path: Path) -> None:
    store = _sense_store(tmp_path)
    sense_a = Sense(definition="old definition A")
    sense_b = Sense(definition="old definition B")
    store.write(Alf(form="word", senses=[sense_a, sense_b]))

    new_a = sense_a.model_copy(update={"definition": "new definition A"})
    new_b = sense_b.model_copy(update={"definition": "new definition B"})

    request = RewriteRequest(
        id=_make_request_id(),
        created_at=datetime.utcnow(),
        form="word",
        before=[sense_a, sense_b],
        after=[new_a, new_b],
    )
    request.apply(store, None)

    result = store.read("word")
    assert result is not None
    assert result.senses[0].id == sense_a.id
    assert result.senses[1].id == sense_b.id
