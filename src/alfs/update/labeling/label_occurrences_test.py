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
    store = _store(tmp_path, _alf("run", "to move quickly", "to operate"))
    menu = build_sense_menu(store, "run")
    assert "1. to move quickly" in menu
    assert "2. to operate" in menu


def test_build_sense_menu_follows_redirect(tmp_path):
    canonical = _alf("run", "to move quickly")
    alias = Alf(form="Run", senses=[], redirect="run")
    store = _store(tmp_path, canonical, alias)
    menu = build_sense_menu(store, "Run")
    assert "1. to move quickly" in menu


def test_build_sense_menu_includes_pos(tmp_path):
    store = _store(
        tmp_path,
        Alf(
            form="run",
            senses=[Sense(definition="to move quickly", pos=PartOfSpeech.verb)],
        ),
    )
    menu = build_sense_menu(store, "run")
    assert "1. [verb] to move quickly" in menu


def test_build_sense_menu_broken_redirect_raises(tmp_path):
    alias = Alf(form="Run", senses=[], redirect="nonexistent")
    store = _store(tmp_path, alias)
    with pytest.raises(ValueError, match="nonexistent"):
        build_sense_menu(store, "Run")
