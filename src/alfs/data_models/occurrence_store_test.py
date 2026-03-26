from pathlib import Path

import polars as pl
import pytest

from alfs.data_models.occurrence_store import OccurrenceStore


@pytest.fixture
def store(tmp_path: Path) -> OccurrenceStore:
    return OccurrenceStore(tmp_path / "labeled.db")


def test_upsert_and_query_form(store: OccurrenceStore) -> None:
    rows = [
        ("run", "doc1", 0, "1", 2, None),
        ("run", "doc1", 50, "2", 1, None),
        ("walk", "doc2", 10, "1", 0, None),
    ]
    store.upsert_many(rows, model="test-model")
    df = store.query_form("run")
    assert len(df) == 2
    assert set(df["sense_key"].to_list()) == {"1", "2"}


def test_upsert_deduplication(store: OccurrenceStore) -> None:
    """INSERT OR REPLACE: later upsert with same PK wins."""
    store.upsert_many([("run", "doc1", 0, "1", 1, None)], model="model-a")
    store.upsert_many([("run", "doc1", 0, "2", 2, None)], model="model-b")
    df = store.query_form("run")
    assert len(df) == 1
    assert df["sense_key"][0] == "2"
    assert df["rating"][0] == 2


def test_query_form_missing_returns_empty(store: OccurrenceStore) -> None:
    df = store.query_form("nonexistent")
    assert len(df) == 0
    assert df.schema == {
        "form": pl.String,
        "doc_id": pl.String,
        "byte_offset": pl.Int64,
        "sense_key": pl.String,
        "rating": pl.Int64,
        "model": pl.String,
        "updated_at": pl.String,
        "synonyms": pl.String,
    }


def test_to_polars(store: OccurrenceStore) -> None:
    rows = [
        ("cat", "doc1", 0, "1", 2, None),
        ("dog", "doc2", 5, "1", 1, None),
    ]
    store.upsert_many(rows, model="test-model")
    df = store.to_polars()
    assert len(df) == 2
    assert set(df["form"].to_list()) == {"cat", "dog"}


def test_to_polars_empty(store: OccurrenceStore) -> None:
    df = store.to_polars()
    assert len(df) == 0


def test_count_by_form(store: OccurrenceStore) -> None:
    rows = [
        ("run", "doc1", 0, "1", 2, None),  # excellent → good
        ("run", "doc1", 10, "1", 1, None),  # okay → good
        ("run", "doc1", 20, "1", 0, None),  # poor → bad
        ("walk", "doc2", 0, "1", 0, None),  # poor → bad
    ]
    store.upsert_many(rows, model="test-model")
    df = store.count_by_form().sort("form")
    run_row = df.filter(pl.col("form") == "run").row(0, named=True)
    assert run_row["n_total"] == 3
    assert run_row["n_bad"] == 1
    assert run_row["n_excellent"] == 1
    walk_row = df.filter(pl.col("form") == "walk").row(0, named=True)
    assert walk_row["n_total"] == 1
    assert walk_row["n_bad"] == 1


def test_count_by_form_empty(store: OccurrenceStore) -> None:
    df = store.count_by_form()
    assert len(df) == 0


def test_delete_by_sense_id_removes_top_level(store: OccurrenceStore) -> None:
    """Deleting by UUID removes only that sense's rows."""
    uid_a = "aaaaaaaa-0000-0000-0000-000000000001"
    uid_b = "bbbbbbbb-0000-0000-0000-000000000002"
    store.upsert_many(
        [
            ("run", "doc1", 0, uid_a, 2, None),
            ("run", "doc1", 10, uid_b, 1, None),
        ],
        model="test-model",
    )
    store.delete_by_sense_id("run", uid_a)
    df = store.query_form("run")
    assert len(df) == 1
    assert df["sense_key"][0] == uid_b


def test_delete_by_sense_id_other_forms_unaffected(store: OccurrenceStore) -> None:
    """Deleting by UUID for one form does not affect another form."""
    uid = "aaaaaaaa-0000-0000-0000-000000000001"
    store.upsert_many(
        [
            ("run", "doc1", 0, uid, 2, None),
            ("walk", "doc1", 0, uid, 1, None),
        ],
        model="test-model",
    )
    store.delete_by_sense_id("run", uid)
    assert len(store.query_form("run")) == 0
    assert len(store.query_form("walk")) == 1


def test_upsert_synonyms_round_trip(store: OccurrenceStore) -> None:
    """synonyms=None (missing), []=empty, non-empty list all survive upsert/query."""
    import json

    store.upsert_many(
        [
            ("run", "doc1", 0, "1", 2, ["sprint", "dash"]),
            ("run", "doc1", 10, "1", 1, []),
            ("run", "doc1", 20, "1", 0, None),
        ],
        model="test-model",
    )
    df = store.query_form("run").sort("byte_offset")
    assert json.loads(df["synonyms"][0]) == ["sprint", "dash"]
    assert json.loads(df["synonyms"][1]) == []
    assert df["synonyms"][2] is None
