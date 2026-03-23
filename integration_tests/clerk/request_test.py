"""Integration tests for clerk request apply() methods."""

from datetime import UTC, datetime
from pathlib import Path

from alfs.clerk.request import (
    MorphRedirectRequest,
    PruneRequest,
    RewriteRequest,
    TrimSenseRequest,
)
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
        created_at=datetime.now(UTC),
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
        created_at=datetime.now(UTC),
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
        created_at=datetime.now(UTC),
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
        created_at=datetime.now(UTC),
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
        created_at=datetime.now(UTC),
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
        created_at=datetime.now(UTC),
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
        created_at=datetime.now(UTC),
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
        created_at=datetime.now(UTC),
        form="word",
        before=[sense_a, sense_b],
        after=[new_a, new_b],
    )
    request.apply(store, None)

    result = store.read("word")
    assert result is not None
    assert result.senses[0].id == sense_a.id
    assert result.senses[1].id == sense_b.id


# --- MorphRedirectRequest ---


def _make_morph_request(
    form: str,
    derived_sense_idx: int,
    base_form: str,
    before: Sense,
    after: Sense,
) -> MorphRedirectRequest:
    return MorphRedirectRequest(
        id=_make_request_id(),
        created_at=datetime.now(UTC),
        form=form,
        derived_sense_idx=derived_sense_idx,
        base_form=base_form,
        base_sense_idx=0,
        relation="plural",
        before=before,
        after=after,
    )


def test_morph_redirect_tags_derived_sense(tmp_path: Path) -> None:
    store = _sense_store(tmp_path)
    before = Sense(definition="canine animal")
    after = before.model_copy(
        update={
            "definition": "plural of dog (n.)",
            "morph_base": "dog",
            "morph_relation": "plural",
        }
    )
    store.write(Alf(form="dogs", senses=[before]))
    store.write(Alf(form="dog", senses=[Sense(definition="canine animal")]))

    _make_morph_request("dogs", 0, "dog", before, after).apply(store, None)

    dogs = store.read("dogs")
    assert dogs is not None
    assert dogs.senses[0].morph_base == "dog"
    assert dogs.senses[0].morph_relation == "plural"
    assert dogs.senses[0].definition == "plural of dog (n.)"


def test_morph_redirect_promotes_before_sense_to_parent(tmp_path: Path) -> None:
    store = _sense_store(tmp_path)
    before = Sense(definition="canine animal")
    after = before.model_copy(
        update={
            "definition": "plural of dog (n.)",
            "morph_base": "dog",
            "morph_relation": "plural",
        }
    )
    store.write(Alf(form="dogs", senses=[before]))
    store.write(Alf(form="dog", senses=[]))

    _make_morph_request("dogs", 0, "dog", before, after).apply(store, None)

    dog = store.read("dog")
    assert dog is not None
    assert any(s.definition == "canine animal" for s in dog.senses)


def test_morph_redirect_promotes_when_flagged(tmp_path: Path) -> None:
    store = _sense_store(tmp_path)
    before = Sense(definition="canine animal")
    after = before.model_copy(
        update={
            "definition": "plural of dog (n.)",
            "morph_base": "dog",
            "morph_relation": "plural",
        }
    )
    store.write(Alf(form="dogs", senses=[before]))
    store.write(Alf(form="dog", senses=[]))

    _make_morph_request("dogs", 0, "dog", before, after).apply(store, None)

    dog = store.read("dog")
    assert dog is not None
    assert any(s.definition == "canine animal" for s in dog.senses)


def test_morph_redirect_skips_promotion_when_not_flagged(tmp_path: Path) -> None:
    store = _sense_store(tmp_path)
    existing_parent_sense = Sense(definition="canine animal")
    before = Sense(definition="canine animal")
    after = before.model_copy(
        update={
            "definition": "plural of dog (n.)",
            "morph_base": "dog",
            "morph_relation": "plural",
        }
    )
    store.write(Alf(form="dogs", senses=[before]))
    store.write(Alf(form="dog", senses=[existing_parent_sense]))

    request = MorphRedirectRequest(
        id=_make_request_id(),
        created_at=datetime.now(UTC),
        form="dogs",
        derived_sense_idx=0,
        base_form="dog",
        base_sense_idx=0,
        relation="plural",
        before=before,
        after=after,
        promote_to_parent=False,
    )
    request.apply(store, None)

    dog = store.read("dog")
    assert dog is not None
    assert len(dog.senses) == 1


def test_morph_redirect_child_retains_independent_sense(tmp_path: Path) -> None:
    store = _sense_store(tmp_path)
    morph_sense = Sense(definition="canine animal")
    independent_sense = Sense(definition="plural of 'dog' (informal harassment)")
    after = morph_sense.model_copy(
        update={
            "definition": "plural of dog (n.)",
            "morph_base": "dog",
            "morph_relation": "plural",
        }
    )
    store.write(Alf(form="dogs", senses=[morph_sense, independent_sense]))
    store.write(Alf(form="dog", senses=[Sense(definition="canine animal")]))

    _make_morph_request("dogs", 0, "dog", morph_sense, after).apply(store, None)

    dogs = store.read("dogs")
    assert dogs is not None
    assert len(dogs.senses) == 2
    assert dogs.senses[1].id == independent_sense.id


def test_morph_redirect_independent_sense_not_promoted(tmp_path: Path) -> None:
    store = _sense_store(tmp_path)
    morph_sense = Sense(definition="canine animal")
    independent_sense = Sense(definition="plural of 'dog' (informal harassment)")
    after = morph_sense.model_copy(
        update={
            "definition": "plural of dog (n.)",
            "morph_base": "dog",
            "morph_relation": "plural",
        }
    )
    store.write(Alf(form="dogs", senses=[morph_sense, independent_sense]))
    store.write(Alf(form="dog", senses=[Sense(definition="canine animal")]))

    _make_morph_request("dogs", 0, "dog", morph_sense, after).apply(store, None)

    dog = store.read("dog")
    assert dog is not None
    assert not any(s.definition == independent_sense.definition for s in dog.senses)


def test_morph_redirect_multiple_relations_same_form(tmp_path: Path) -> None:
    store = _sense_store(tmp_path)
    noun_sense = Sense(definition="canine animal", pos=None)
    verb_sense = Sense(definition="to follow or harass persistently")
    store.write(Alf(form="dogs", senses=[noun_sense, verb_sense]))
    store.write(Alf(form="dog", senses=[]))

    after_noun = noun_sense.model_copy(
        update={
            "definition": "plural of dog (n.)",
            "morph_base": "dog",
            "morph_relation": "plural",
        }
    )
    MorphRedirectRequest(
        id=_make_request_id(),
        created_at=datetime.now(UTC),
        form="dogs",
        derived_sense_idx=0,
        base_form="dog",
        base_sense_idx=0,
        relation="plural",
        before=noun_sense,
        after=after_noun,
    ).apply(store, None)

    after_verb = verb_sense.model_copy(
        update={
            "definition": "third person singular of dog (v.)",
            "morph_base": "dog",
            "morph_relation": "3sg_present",
        }
    )
    MorphRedirectRequest(
        id=_make_request_id(),
        created_at=datetime.now(UTC),
        form="dogs",
        derived_sense_idx=1,
        base_form="dog",
        base_sense_idx=0,
        relation="3sg_present",
        before=verb_sense,
        after=after_verb,
    ).apply(store, None)

    dogs = store.read("dogs")
    assert dogs is not None
    assert dogs.senses[0].morph_relation == "plural"
    assert dogs.senses[1].morph_relation == "3sg_present"

    dog = store.read("dog")
    assert dog is not None
    defs = {s.definition for s in dog.senses}
    assert "canine animal" in defs
    assert "to follow or harass persistently" in defs
