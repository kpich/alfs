from alfs.data_models.alf import Alf, Sense, morph_base_form, parse_sense_key, sense_key


def test_sense_key():
    assert sense_key(0) == "1"
    assert sense_key(2) == "3"


def test_parse_sense_key():
    assert parse_sense_key("1") == 0
    assert parse_sense_key("3") == 2


def test_sense_key_round_trip():
    for top in range(3):
        assert parse_sense_key(sense_key(top)) == top


def test_get_sense_top_level():
    alf = Alf(
        form="run",
        senses=[
            Sense(definition="to move fast"),
            Sense(definition="to manage"),
        ],
    )
    assert alf.get_sense("1") == "to move fast"
    assert alf.get_sense("2") == "to manage"


def test_morph_base_form_no_senses():
    alf = Alf(form="does", senses=[])
    assert morph_base_form(alf) is None


def test_morph_base_form_no_morph_base():
    alf = Alf(form="does", senses=[Sense(definition="third-person singular of do")])
    assert morph_base_form(alf) is None


def test_morph_base_form_all_same():
    alf = Alf(
        form="does",
        senses=[
            Sense(definition="third-person singular of do", morph_base="do"),
            Sense(definition="plural of doe", morph_base="do"),
        ],
    )
    assert morph_base_form(alf) == "do"


def test_morph_base_form_mixed_bases():
    alf = Alf(
        form="x",
        senses=[
            Sense(definition="sense 1", morph_base="a"),
            Sense(definition="sense 2", morph_base="b"),
        ],
    )
    assert morph_base_form(alf) is None


def test_morph_base_form_some_without_base():
    alf = Alf(
        form="does",
        senses=[
            Sense(definition="third-person singular of do", morph_base="do"),
            Sense(definition="standalone sense"),
        ],
    )
    assert morph_base_form(alf) == "do"
