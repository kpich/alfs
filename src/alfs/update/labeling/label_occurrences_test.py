import pytest

from alfs.data_models.alf import Alf, Alfs, Sense
from alfs.data_models.pos import PartOfSpeech
from alfs.update.labeling.label_occurrences import build_sense_menu


def _alfs(*entries: Alf) -> Alfs:
    return Alfs(entries={a.form: a for a in entries})


def _alf(form: str, *definitions: str) -> Alf:
    return Alf(form=form, senses=[Sense(definition=d) for d in definitions])


def test_build_sense_menu_simple():
    alfs = _alfs(_alf("run", "to move quickly", "to operate"))
    menu = build_sense_menu(alfs, "run")
    assert "1. to move quickly" in menu
    assert "2. to operate" in menu


def test_build_sense_menu_follows_redirect():
    canonical = _alf("run", "to move quickly")
    alias = Alf(form="Run", senses=[], redirect="run")
    alfs = _alfs(canonical, alias)
    menu = build_sense_menu(alfs, "Run")
    assert "1. to move quickly" in menu


def test_build_sense_menu_includes_pos():
    alfs = _alfs(
        Alf(
            form="run",
            senses=[Sense(definition="to move quickly", pos=PartOfSpeech.verb)],
        )
    )
    menu = build_sense_menu(alfs, "run")
    assert "1. [verb] to move quickly" in menu


def test_build_sense_menu_broken_redirect_raises():
    alias = Alf(form="Run", senses=[], redirect="nonexistent")
    alfs = _alfs(alias)
    with pytest.raises(ValueError, match="nonexistent"):
        build_sense_menu(alfs, "Run")
