from alfs.data_models.alf import Alf, Sense
from alfs.update.update_inventory import merge_entry


def _alf(form: str, *definitions: str) -> Alf:
    return Alf(form=form, senses=[Sense(definition=d) for d in definitions])


def test_merge_adds_new_sense():
    existing = _alf("run", "to move quickly")
    new = _alf("run", "to manage or operate")
    result = merge_entry(existing, new)
    assert len(result.senses) == 2
    assert result.senses[0].definition == "to move quickly"
    assert result.senses[1].definition == "to manage or operate"


def test_merge_skips_exact_duplicate():
    existing = _alf("run", "to move quickly")
    new = _alf("run", "to move quickly")
    result = merge_entry(existing, new)
    assert len(result.senses) == 1


def test_merge_skips_case_and_whitespace_duplicate():
    existing = _alf("run", "to move quickly")
    new = _alf("run", "  To Move Quickly  ")
    result = merge_entry(existing, new)
    assert len(result.senses) == 1


def test_merge_preserves_redirect():
    existing = Alf(form="The", senses=[], redirect="the")
    new = _alf("The", "some new sense")
    result = merge_entry(existing, new)
    assert result.redirect == "the"
