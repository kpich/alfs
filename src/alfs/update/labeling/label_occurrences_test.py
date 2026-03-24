import pytest

from alfs.data_models.alf import Alf, Sense
from alfs.data_models.pos import PartOfSpeech
from alfs.data_models.sense_store import SenseStore
from alfs.update.labeling.label_occurrences import build_sense_menu


def _store(tmp_path, *entries: Alf) -> SenseStore:
    store = SenseStore(tmp_path / "senses.db")
    for entry in entries:
        store.write(entry)
    return store


def _alf(form: str, *definitions: str) -> Alf:
    return Alf(form=form, senses=[Sense(definition=d) for d in definitions])


def test_build_sense_menu_simple(tmp_path):
    alf = _alf("run", "to move quickly", "to operate")
    store = _store(tmp_path, alf)
    menu, key_map = build_sense_menu(store, "run")
    assert "1. to move quickly" in menu
    assert "2. to operate" in menu
    assert key_map["1"] == alf.senses[0].id
    assert key_map["2"] == alf.senses[1].id


def test_build_sense_menu_follows_redirect(tmp_path):
    canonical = _alf("run", "to move quickly")
    alias = Alf(form="Run", senses=[], redirect="run")
    store = _store(tmp_path, canonical, alias)
    menu, key_map = build_sense_menu(store, "Run")
    assert "1. to move quickly" in menu
    assert key_map["1"] == canonical.senses[0].id


def test_build_sense_menu_includes_pos(tmp_path):
    store = _store(
        tmp_path,
        Alf(
            form="run",
            senses=[Sense(definition="to move quickly", pos=PartOfSpeech.verb)],
        ),
    )
    menu, _ = build_sense_menu(store, "run")
    assert "1. [verb] to move quickly" in menu


def test_build_sense_menu_broken_redirect_raises(tmp_path):
    alias = Alf(form="Run", senses=[], redirect="nonexistent")
    store = _store(tmp_path, alias)
    with pytest.raises(ValueError, match="nonexistent"):
        build_sense_menu(store, "Run")


def test_build_sense_menu_morph_base_senses_are_selectable(tmp_path):
    dog = _alf("dog", "a domesticated animal", "an unattractive person")
    dogs = Alf(
        form="dogs",
        senses=[
            Sense(definition="plural of dog", morph_base="dog"),
            Sense(definition="harasses persistently", morph_base="dog"),
        ],
    )
    store = _store(tmp_path, dog, dogs)
    menu, key_map = build_sense_menu(store, "dogs")

    # dogs' own senses at 1 and 2
    assert "1. plural of dog" in menu
    assert "2. harasses persistently" in menu
    assert key_map["1"] == dogs.senses[0].id
    assert key_map["2"] == dogs.senses[1].id

    # dog's senses numbered 3 and 4, selectable
    assert "3. a domesticated animal" in menu
    assert "4. an unattractive person" in menu
    assert key_map["3"] == dog.senses[0].id
    assert key_map["4"] == dog.senses[1].id


def test_build_sense_menu_redirect_then_morph_base(tmp_path):
    dog = _alf("dog", "a domesticated animal")
    dogs = Alf(
        form="dogs",
        senses=[Sense(definition="plural of dog", morph_base="dog")],
    )
    Dogs = Alf(form="Dogs", senses=[], redirect="dogs")
    store = _store(tmp_path, dog, dogs, Dogs)
    menu, key_map = build_sense_menu(store, "Dogs")

    # dogs' own sense at 1
    assert "1. plural of dog" in menu
    assert key_map["1"] == dogs.senses[0].id

    # dog's sense at 2, selectable
    assert "2. a domesticated animal" in menu
    assert key_map["2"] == dog.senses[0].id
