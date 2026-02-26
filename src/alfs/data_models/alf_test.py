from alfs.data_models.alf import Alf, Sense, parse_sense_key, sense_key


def test_sense_key_top_level():
    assert sense_key(0) == "1"
    assert sense_key(2) == "3"


def test_sense_key_subsense():
    assert sense_key(2, 0) == "3a"
    assert sense_key(2, 1) == "3b"


def test_parse_sense_key_top_level():
    assert parse_sense_key("1") == (0, None)
    assert parse_sense_key("3") == (2, None)


def test_parse_sense_key_subsense():
    assert parse_sense_key("3b") == (2, 1)
    assert parse_sense_key("1a") == (0, 0)


def test_sense_key_round_trip():
    for top in range(3):
        assert parse_sense_key(sense_key(top)) == (top, None)
        for sub in range(3):
            assert parse_sense_key(sense_key(top, sub)) == (top, sub)


def test_get_sense_top_level():
    alf = Alf(
        form="run",
        senses=[
            Sense(definition="to move fast", subsenses=["on foot", "in a race"]),
            Sense(definition="to manage"),
        ],
    )
    assert alf.get_sense("1") == "to move fast"
    assert alf.get_sense("2") == "to manage"


def test_get_sense_subsense():
    alf = Alf(
        form="run",
        senses=[
            Sense(definition="to move fast", subsenses=["on foot", "in a race"]),
        ],
    )
    assert alf.get_sense("1a") == "on foot"
    assert alf.get_sense("1b") == "in a race"
