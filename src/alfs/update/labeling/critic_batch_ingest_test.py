"""Unit tests for critic_batch_ingest."""

import json
from pathlib import Path

import pytest

from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.update.labeling.critic_batch_ingest import ingest, parse_critic_response

# --- parse_critic_response ---


def test_parse_empty_bad_indices() -> None:
    assert parse_critic_response('{"bad_indices": []}') == []


def test_parse_with_bad_indices() -> None:
    assert parse_critic_response('{"bad_indices": [1, 3]}') == [1, 3]


def test_parse_float_indices_that_are_ints() -> None:
    assert parse_critic_response('{"bad_indices": [1.0, 2.0]}') == [1, 2]


def test_parse_malformed_json() -> None:
    assert parse_critic_response("not json") is None


def test_parse_missing_bad_indices_key() -> None:
    assert parse_critic_response('{"other": [1]}') is None


def test_parse_bad_indices_not_list() -> None:
    assert parse_critic_response('{"bad_indices": 1}') is None


def test_parse_extra_fields_ok() -> None:
    result = parse_critic_response('{"bad_indices": [2], "explanation": "wrong sense"}')
    assert result == [2]


# --- ingest ---


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_response_line(custom_id: str, bad_indices: list[int]) -> dict:
    return {
        "custom_id": custom_id,
        "response": {
            "body": {
                "choices": [
                    {"message": {"content": json.dumps({"bad_indices": bad_indices})}}
                ]
            }
        },
    }


@pytest.fixture
def store(tmp_path: Path) -> OccurrenceStore:
    s = OccurrenceStore(tmp_path / "labeled.db")
    s.upsert_many(
        [
            ("dog", "doc1", 0, "uuid-1", 2, None),
            ("dog", "doc1", 10, "uuid-1", 2, None),
            ("dog", "doc1", 20, "uuid-1", 2, None),
        ],
        model="test-model",
    )
    return s


def test_ingest_marks_reviewed_and_downgrades(
    store: OccurrenceStore, tmp_path: Path
) -> None:
    batch_out = tmp_path / "critic_output.jsonl"
    metadata = tmp_path / "critic_metadata_001.jsonl"

    instances = [
        {"doc_id": "doc1", "byte_offset": 0},
        {"doc_id": "doc1", "byte_offset": 10},
        {"doc_id": "doc1", "byte_offset": 20},
    ]
    _write_jsonl(
        metadata,
        [
            {
                "custom_id": "0",
                "form": "dog",
                "sense_uuid": "uuid-1",
                "sense_definition": "a domestic animal",
                "model": "openai/gpt-oss-20b",
                "instances": instances,
            }
        ],
    )
    _write_jsonl(batch_out, [_make_response_line("0", [2])])  # index 2 (1-based) is bad

    n_reviewed, n_bad = ingest(
        batch_output=batch_out,
        labeled_db=store._db_path,
        metadata_path=metadata,
    )

    assert n_reviewed == 3
    assert n_bad == 1

    df = store.query_form("dog").sort("byte_offset")
    assert df["rating"][0] == 2  # byte_offset=0: not flagged
    assert df["rating"][1] == 0  # byte_offset=10: index 2 → downgraded
    assert df["rating"][2] == 2  # byte_offset=20: not flagged
    assert df["last_critic_date"][0] is not None
    assert df["last_critic_date"][1] is not None
    assert df["last_critic_date"][2] is not None


def test_ingest_all_good(store: OccurrenceStore, tmp_path: Path) -> None:
    batch_out = tmp_path / "critic_output.jsonl"
    metadata = tmp_path / "critic_metadata_001.jsonl"

    instances = [
        {"doc_id": "doc1", "byte_offset": 0},
        {"doc_id": "doc1", "byte_offset": 10},
    ]
    _write_jsonl(
        metadata,
        [
            {
                "custom_id": "0",
                "form": "dog",
                "sense_uuid": "uuid-1",
                "sense_definition": "a domestic animal",
                "model": "openai/gpt-oss-20b",
                "instances": instances,
            }
        ],
    )
    _write_jsonl(batch_out, [_make_response_line("0", [])])

    n_reviewed, n_bad = ingest(
        batch_output=batch_out,
        labeled_db=store._db_path,
        metadata_path=metadata,
    )

    assert n_reviewed == 2
    assert n_bad == 0
    df = store.query_form("dog").sort("byte_offset")
    assert df["rating"][0] == 2
    assert df["rating"][1] == 2
    assert df["last_critic_date"][0] is not None


def test_ingest_out_of_range_indices_ignored(
    store: OccurrenceStore, tmp_path: Path
) -> None:
    batch_out = tmp_path / "critic_output.jsonl"
    metadata = tmp_path / "critic_metadata_001.jsonl"

    instances = [{"doc_id": "doc1", "byte_offset": 0}]
    _write_jsonl(
        metadata,
        [
            {
                "custom_id": "0",
                "form": "dog",
                "sense_uuid": "uuid-1",
                "sense_definition": "a domestic animal",
                "model": "openai/gpt-oss-20b",
                "instances": instances,
            }
        ],
    )
    # bad_indices=[99] is out of range (only 1 instance)
    _write_jsonl(batch_out, [_make_response_line("0", [99])])

    n_reviewed, n_bad = ingest(
        batch_output=batch_out,
        labeled_db=store._db_path,
        metadata_path=metadata,
    )

    assert n_reviewed == 1
    assert n_bad == 0  # out-of-range index ignored
    df = store.query_form("dog")
    assert df["rating"][0] == 2  # not downgraded
    assert df["last_critic_date"][0] is not None  # still marked reviewed


def test_ingest_auto_discover_metadata(store: OccurrenceStore, tmp_path: Path) -> None:
    batch_out = tmp_path / "critic_output.jsonl"
    metadata = tmp_path / "critic_metadata_20260406T120000_001.jsonl"

    instances = [{"doc_id": "doc1", "byte_offset": 0}]
    _write_jsonl(
        metadata,
        [
            {
                "custom_id": "0",
                "form": "dog",
                "sense_uuid": "uuid-1",
                "sense_definition": "a domestic animal",
                "model": "openai/gpt-oss-20b",
                "instances": instances,
            }
        ],
    )
    _write_jsonl(batch_out, [_make_response_line("0", [])])

    n_reviewed, _ = ingest(
        batch_output=batch_out,
        labeled_db=store._db_path,
        batch_dir=tmp_path,
    )
    assert n_reviewed == 1
