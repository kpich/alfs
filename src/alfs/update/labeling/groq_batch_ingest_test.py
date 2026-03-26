"""Unit tests for groq_batch_ingest.parse_response."""

from alfs.update.labeling.groq_batch_ingest import parse_response


def test_parse_valid() -> None:
    result = parse_response('{"sense_key": "1", "rating": 2}')
    assert result == {"sense_key": "1", "rating": 2}


def test_parse_rating_zero() -> None:
    result = parse_response('{"sense_key": "0", "rating": 0}')
    assert result == {"sense_key": "0", "rating": 0}


def test_parse_malformed_json() -> None:
    assert parse_response("not json") is None


def test_parse_empty_string() -> None:
    assert parse_response("") is None


def test_parse_missing_sense_key() -> None:
    assert parse_response('{"rating": 2}') is None


def test_parse_missing_rating() -> None:
    assert parse_response('{"sense_key": "1"}') is None


def test_parse_extra_fields_ok() -> None:
    result = parse_response(
        '{"sense_key": "2", "rating": 1, "explanation": "fits well"}'
    )
    assert result is not None
    assert result["sense_key"] == "2"
    assert result["rating"] == 1
