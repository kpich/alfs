"""Unit tests for groq_batch_ingest."""

import json
from pathlib import Path

from alfs.data_models.alf import Alf, Sense
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.pos import PartOfSpeech
from alfs.data_models.sense_store import SenseStore
from alfs.update.labeling.groq_batch_ingest import ingest, parse_response


def test_parse_valid() -> None:
    result = parse_response('{"sense_key": "1", "rating": 2}')
    assert result == {"sense_key": "1", "rating": 2, "synonyms": None}


def test_parse_rating_zero_skip() -> None:
    result = parse_response('{"sense_key": "0", "rating": 0}')
    assert result == {"sense_key": "0", "rating": 0, "synonyms": None}


def test_parse_rating_zero_none() -> None:
    result = parse_response('{"sense_key": "_none", "rating": 0}')
    assert result == {"sense_key": "_none", "rating": 0, "synonyms": None}


def test_parse_with_synonyms() -> None:
    result = parse_response(
        '{"sense_key": "1", "rating": 2, "synonyms": ["quick", "fast"]}'
    )
    assert result is not None
    assert result["synonyms"] == ["quick", "fast"]


def test_parse_empty_synonyms() -> None:
    result = parse_response('{"sense_key": "1", "rating": 2, "synonyms": []}')
    assert result is not None
    assert result["synonyms"] == []


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
    assert result["synonyms"] is None


def _write_response_line(custom_id: str, sense_key: str, rating: int) -> str:
    body = {
        "custom_id": custom_id,
        "response": {
            "body": {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "sense_key": sense_key,
                                    "rating": rating,
                                    "synonyms": [],
                                }
                            )
                        }
                    }
                ]
            }
        },
    }
    return json.dumps(body) + "\n"


def test_ingest_passes_through_reserved_sentinels(tmp_path: Path) -> None:
    """LLM responses with sense_key="0" or "_none" must land in labeled.db
    verbatim, without needing a key_map lookup against senses.db."""
    senses_db = tmp_path / "senses.db"
    labeled_db = tmp_path / "labeled.db"
    SenseStore(senses_db).update(
        "cat",
        lambda _: Alf(
            form="cat",
            senses=[Sense(definition="a small feline", pos=PartOfSpeech.noun)],
        ),
    )

    metadata_path = tmp_path / "meta.jsonl"
    with metadata_path.open("w") as f:
        for cid, doc_id in [("0", "d_skip"), ("1", "d_none")]:
            f.write(
                json.dumps(
                    {
                        "custom_id": cid,
                        "form": "cat",
                        "doc_id": doc_id,
                        "byte_offset": 0,
                        "model": "test-model",
                    }
                )
                + "\n"
            )

    batch_output = tmp_path / "out.jsonl"
    with batch_output.open("w") as f:
        f.write(_write_response_line("0", "0", 0))
        f.write(_write_response_line("1", "_none", 0))

    n = ingest(batch_output, senses_db, labeled_db, metadata_path=metadata_path)
    assert n == 2

    df = OccurrenceStore(labeled_db).query_form("cat")
    rows = {r["doc_id"]: r["sense_key"] for r in df.iter_rows(named=True)}
    assert rows == {"d_skip": "0", "d_none": "_none"}
