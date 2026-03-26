"""Tests for ChangeRequest.apply() methods."""

from datetime import UTC, datetime
from pathlib import Path
import uuid

import pytest

from alfs.clerk.queue import drain, enqueue
from alfs.clerk.request import (
    AddSensesRequest,
    ClearRedirectSensesRequest,
    DeleteEntryRequest,
    MorphRedirectRequest,
    PosTagRequest,
    PruneRequest,
    RewriteRequest,
    SetRedirectRequest,
    SetSpellingVariantRequest,
    TrimSenseRequest,
    UpdatePosRequest,
)
from alfs.data_models.alf import Alf, Sense
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore


def _req_id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


def _sense(definition: str, model: str | None = None) -> Sense:
    return Sense(definition=definition, updated_by_model=model)


@pytest.fixture
def store(tmp_path: Path) -> SenseStore:
    return SenseStore(tmp_path / "senses.db")


@pytest.fixture
def occ_store(tmp_path: Path) -> OccurrenceStore:
    return OccurrenceStore(tmp_path / "labeled.db")


# --- AddSensesRequest ---


def test_add_senses_creates_new_entry(store: SenseStore) -> None:
    req = AddSensesRequest(
        id=_req_id(),
        created_at=_now(),
        form="run",
        new_senses=[_sense("to move quickly")],
    )
    result = req.apply(store, None)
    assert result is True
    entry = store.read("run")
    assert entry is not None
    assert len(entry.senses) == 1
    assert entry.senses[0].definition == "to move quickly"


def test_add_senses_merges_into_existing(store: SenseStore) -> None:
    store.write(Alf(form="run", senses=[_sense("to move quickly")]))
    req = AddSensesRequest(
        id=_req_id(),
        created_at=_now(),
        form="run",
        new_senses=[_sense("to manage")],
    )
    req.apply(store, None)
    entry = store.read("run")
    assert entry is not None
    assert len(entry.senses) == 2


def test_add_senses_skips_duplicate_definitions(store: SenseStore) -> None:
    store.write(Alf(form="run", senses=[_sense("to move quickly")]))
    req = AddSensesRequest(
        id=_req_id(),
        created_at=_now(),
        form="run",
        new_senses=[_sense("To Move Quickly")],  # same after strip/lower
    )
    req.apply(store, None)
    entry = store.read("run")
    assert entry is not None
    assert len(entry.senses) == 1


# --- RewriteRequest ---


def test_rewrite_replaces_matching_sense(store: SenseStore) -> None:
    before = _sense("old definition")
    store.write(Alf(form="run", senses=[before]))
    after = before.model_copy(update={"definition": "new definition"})
    req = RewriteRequest(
        id=_req_id(),
        created_at=_now(),
        form="run",
        before=before,
        after=after,
    )
    result = req.apply(store, None)
    assert result is True
    entry = store.read("run")
    assert entry is not None
    assert entry.senses[0].definition == "new definition"


def test_rewrite_skips_if_rank_insufficient(store: SenseStore) -> None:
    before = _sense("old definition", model="claude-code")  # high-rank model
    store.write(Alf(form="run", senses=[before]))
    after = before.model_copy(update={"definition": "new definition"})
    req = RewriteRequest(
        id=_req_id(),
        created_at=_now(),
        form="run",
        before=before,
        after=after,
        requesting_model="qwen2.5:32b",  # lower rank
    )
    result = req.apply(store, None)
    assert result is False
    entry = store.read("run")
    assert entry is not None
    assert entry.senses[0].definition == "old definition"


# --- PosTagRequest / UpdatePosRequest ---


def test_pos_tag_replaces_sense(store: SenseStore) -> None:
    from alfs.data_models.pos import PartOfSpeech

    before = _sense("to move quickly")
    store.write(Alf(form="run", senses=[before]))
    after = before.model_copy(update={"pos": PartOfSpeech.verb})
    req = PosTagRequest(
        id=_req_id(),
        created_at=_now(),
        form="run",
        before=before,
        after=after,
    )
    result = req.apply(store, None)
    assert result is True
    entry = store.read("run")
    assert entry is not None
    assert entry.senses[0].pos == PartOfSpeech.verb


def test_update_pos_skips_if_rank_insufficient(store: SenseStore) -> None:
    from alfs.data_models.pos import PartOfSpeech

    before = _sense("to move quickly", model="claude-code")
    store.write(Alf(form="run", senses=[before]))
    after = before.model_copy(update={"pos": PartOfSpeech.verb})
    req = UpdatePosRequest(
        id=_req_id(),
        created_at=_now(),
        form="run",
        before=before,
        after=after,
        requesting_model="qwen2.5:32b",
    )
    result = req.apply(store, None)
    assert result is False


# --- PruneRequest ---


def test_prune_removes_senses(store: SenseStore) -> None:
    s1 = _sense("to move quickly")
    s2 = _sense("to manage")
    store.write(Alf(form="run", senses=[s1, s2]))
    req = PruneRequest(
        id=_req_id(),
        created_at=_now(),
        form="run",
        before=[s1, s2],
        after=[s2],
        removed_ids=[s1.id],
    )
    result = req.apply(store, None)
    assert result is True
    entry = store.read("run")
    assert entry is not None
    assert len(entry.senses) == 1
    assert entry.senses[0].definition == "to manage"


def test_prune_deletes_occurrences(
    store: SenseStore, occ_store: OccurrenceStore
) -> None:
    s1 = _sense("to move quickly")
    s2 = _sense("to manage")
    store.write(Alf(form="run", senses=[s1, s2]))
    occ_store.upsert_many([("run", "doc1", 0, s1.id, 1, None)], model="test")
    occ_store.upsert_many([("run", "doc2", 10, s2.id, 1, None)], model="test")

    req = PruneRequest(
        id=_req_id(),
        created_at=_now(),
        form="run",
        before=[s1, s2],
        after=[s2],
        removed_ids=[s1.id],
    )
    req.apply(store, occ_store)

    remaining = occ_store.query_form("run")
    assert len(remaining) == 1
    assert remaining["sense_key"][0] == s2.id


def test_prune_skips_if_rank_insufficient(store: SenseStore) -> None:
    s1 = _sense("to move quickly", model="claude-code")
    store.write(Alf(form="run", senses=[s1]))
    req = PruneRequest(
        id=_req_id(),
        created_at=_now(),
        form="run",
        before=[s1],
        after=[],
        removed_ids=[s1.id],
        requesting_model="qwen2.5:32b",
    )
    result = req.apply(store, None)
    assert result is False
    entry = store.read("run")
    assert entry is not None
    assert len(entry.senses) == 1


# --- TrimSenseRequest ---


def test_trim_sense_removes_one_sense(store: SenseStore) -> None:
    s1 = _sense("to move quickly")
    s2 = _sense("to manage")
    store.write(Alf(form="run", senses=[s1, s2]))
    req = TrimSenseRequest(
        id=_req_id(),
        created_at=_now(),
        form="run",
        before=[s1, s2],
        after=[s2],
        sense_id=s1.id,
        reason="redundant",
    )
    result = req.apply(store, None)
    assert result is True
    entry = store.read("run")
    assert entry is not None
    assert len(entry.senses) == 1
    assert entry.senses[0].definition == "to manage"


def test_trim_sense_deletes_occurrences(
    store: SenseStore, occ_store: OccurrenceStore
) -> None:
    s1 = _sense("to move quickly")
    store.write(Alf(form="run", senses=[s1]))
    occ_store.upsert_many([("run", "doc1", 0, s1.id, 1, None)], model="test")

    req = TrimSenseRequest(
        id=_req_id(),
        created_at=_now(),
        form="run",
        before=[s1],
        after=[],
        sense_id=s1.id,
        reason="redundant",
    )
    req.apply(store, occ_store)

    assert len(occ_store.query_form("run")) == 0


def test_trim_sense_skips_if_rank_insufficient(store: SenseStore) -> None:
    s1 = _sense("to move quickly", model="claude-code")
    store.write(Alf(form="run", senses=[s1]))
    req = TrimSenseRequest(
        id=_req_id(),
        created_at=_now(),
        form="run",
        before=[s1],
        after=[],
        sense_id=s1.id,
        reason="redundant",
        requesting_model="qwen2.5:32b",
    )
    result = req.apply(store, None)
    assert result is False


# --- MorphRedirectRequest ---


def test_morph_redirect_updates_derived_sense(store: SenseStore) -> None:
    before = _sense("moving quickly")
    store.write(Alf(form="running", senses=[before]))
    after = before.model_copy(update={"morph_base": "run", "morph_relation": "gerund"})
    req = MorphRedirectRequest(
        id=_req_id(),
        created_at=_now(),
        form="running",
        derived_sense_idx=0,
        base_form="run",
        base_sense_idx=0,
        relation="gerund",
        before=before,
        after=after,
        promote_to_parent=False,
    )
    result = req.apply(store, None)
    assert result is True
    entry = store.read("running")
    assert entry is not None
    assert entry.senses[0].morph_base == "run"


def test_morph_redirect_promotes_to_parent(store: SenseStore) -> None:
    before = _sense("moving quickly")
    store.write(Alf(form="running", senses=[before]))
    store.write(Alf(form="run", senses=[_sense("existing run sense")]))
    after = before.model_copy(update={"morph_base": "run", "morph_relation": "gerund"})
    req = MorphRedirectRequest(
        id=_req_id(),
        created_at=_now(),
        form="running",
        derived_sense_idx=0,
        base_form="run",
        base_sense_idx=0,
        relation="gerund",
        before=before,
        after=after,
        promote_to_parent=True,
    )
    req.apply(store, None)
    parent = store.read("run")
    assert parent is not None
    assert len(parent.senses) == 2
    assert any(s.definition == "moving quickly" for s in parent.senses)


def test_morph_redirect_creates_parent_if_missing(store: SenseStore) -> None:
    before = _sense("moving quickly")
    store.write(Alf(form="running", senses=[before]))
    after = before.model_copy(update={"morph_base": "run", "morph_relation": "gerund"})
    req = MorphRedirectRequest(
        id=_req_id(),
        created_at=_now(),
        form="running",
        derived_sense_idx=0,
        base_form="run",
        base_sense_idx=0,
        relation="gerund",
        before=before,
        after=after,
        promote_to_parent=True,
    )
    req.apply(store, None)
    parent = store.read("run")
    assert parent is not None
    assert len(parent.senses) == 1
    assert parent.senses[0].definition == "moving quickly"


# --- SetRedirectRequest ---


def test_set_redirect_on_existing_entry(store: SenseStore) -> None:
    store.write(Alf(form="colour", senses=[_sense("a visual property")]))
    req = SetRedirectRequest(
        id=_req_id(), created_at=_now(), form="colour", redirect_to="color"
    )
    result = req.apply(store, None)
    assert result is True
    entry = store.read("colour")
    assert entry is not None
    assert entry.redirect == "color"


def test_set_redirect_creates_entry_if_missing(store: SenseStore) -> None:
    req = SetRedirectRequest(
        id=_req_id(), created_at=_now(), form="colour", redirect_to="color"
    )
    req.apply(store, None)
    entry = store.read("colour")
    assert entry is not None
    assert entry.redirect == "color"


# --- SetSpellingVariantRequest ---


def test_set_spelling_variant(store: SenseStore) -> None:
    store.write(Alf(form="colour", senses=[_sense("a visual property")]))
    req = SetSpellingVariantRequest(
        id=_req_id(), created_at=_now(), form="colour", preferred_form="color"
    )
    result = req.apply(store, None)
    assert result is True
    entry = store.read("colour")
    assert entry is not None
    assert entry.spelling_variant_of == "color"


# --- ClearRedirectSensesRequest ---


def test_clear_redirect_senses(store: SenseStore) -> None:
    store.write(
        Alf(form="run", senses=[_sense("to move quickly"), _sense("to manage")])
    )
    req = ClearRedirectSensesRequest(id=_req_id(), created_at=_now(), form="run")
    result = req.apply(store, None)
    assert result is True
    entry = store.read("run")
    assert entry is not None
    assert entry.senses == []


# --- DeleteEntryRequest ---


def test_delete_entry_removes_from_store(store: SenseStore) -> None:
    store.write(Alf(form="run", senses=[_sense("to move quickly")]))
    req = DeleteEntryRequest(id=_req_id(), created_at=_now(), form="run", reason="test")
    result = req.apply(store, None)
    assert result is True
    assert store.read("run") is None


def test_delete_entry_removes_occurrences(
    store: SenseStore, occ_store: OccurrenceStore
) -> None:
    s1 = _sense("to move quickly")
    store.write(Alf(form="run", senses=[s1]))
    occ_store.upsert_many([("run", "doc1", 0, s1.id, 1, None)], model="test")
    req = DeleteEntryRequest(id=_req_id(), created_at=_now(), form="run", reason="test")
    req.apply(store, occ_store)
    assert len(occ_store.query_form("run")) == 0


def test_delete_entry_skips_if_rank_insufficient(store: SenseStore) -> None:
    s1 = _sense("to move quickly", model="claude-code")
    store.write(Alf(form="run", senses=[s1]))
    req = DeleteEntryRequest(
        id=_req_id(),
        created_at=_now(),
        form="run",
        reason="test",
        requesting_model="qwen2.5:32b",
    )
    result = req.apply(store, None)
    assert result is False
    assert store.read("run") is not None


def test_delete_entry_missing_form_is_noop(store: SenseStore) -> None:
    req = DeleteEntryRequest(
        id=_req_id(), created_at=_now(), form="nonexistent", reason="test"
    )
    result = req.apply(store, None)
    assert result is True


# --- Queue drain ---


def test_enqueue_and_drain_applies_request(tmp_path: Path) -> None:
    store = SenseStore(tmp_path / "senses.db")
    queue_dir = tmp_path / "queue"
    req = AddSensesRequest(
        id=_req_id(),
        created_at=_now(),
        form="run",
        new_senses=[_sense("to move quickly")],
    )
    enqueue(req, queue_dir)
    assert len(list((queue_dir / "pending").glob("*.json"))) == 1

    drain(queue_dir, store, None)

    assert len(list((queue_dir / "pending").glob("*.json"))) == 0
    assert len(list((queue_dir / "done").glob("*.json"))) == 1
    entry = store.read("run")
    assert entry is not None
    assert entry.senses[0].definition == "to move quickly"


def test_drain_moves_invalid_to_failed(tmp_path: Path) -> None:
    store = SenseStore(tmp_path / "senses.db")
    queue_dir = tmp_path / "queue"
    (queue_dir / "pending").mkdir(parents=True)
    bad_file = queue_dir / "pending" / "bad.json"
    bad_file.write_text("not valid json at all {{{")

    drain(queue_dir, store, None)

    assert not bad_file.exists()
    assert (queue_dir / "failed" / "bad.json").exists()
    assert (queue_dir / "failed" / "bad.err").exists()


def test_drain_processes_multiple_requests(tmp_path: Path) -> None:
    store = SenseStore(tmp_path / "senses.db")
    queue_dir = tmp_path / "queue"
    for form in ("cat", "dog", "fish"):
        enqueue(
            AddSensesRequest(
                id=_req_id(),
                created_at=_now(),
                form=form,
                new_senses=[_sense(f"a {form}")],
            ),
            queue_dir,
        )

    drain(queue_dir, store, None)

    for form in ("cat", "dog", "fish"):
        assert store.read(form) is not None
    assert len(list((queue_dir / "done").glob("*.json"))) == 3
