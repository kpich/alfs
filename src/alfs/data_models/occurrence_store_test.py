from pathlib import Path

import polars as pl
import pytest

from alfs.data_models.occurrence_store import OccurrenceStore


@pytest.fixture
def store(tmp_path: Path) -> OccurrenceStore:
    return OccurrenceStore(tmp_path / "labeled.db")


def test_upsert_and_query_form(store: OccurrenceStore) -> None:
    rows = [
        ("run", "doc1", 0, "1", 3),
        ("run", "doc1", 50, "2", 2),
        ("walk", "doc2", 10, "1", 1),
    ]
    store.upsert_many(rows)
    df = store.query_form("run")
    assert len(df) == 2
    assert set(df["sense_key"].to_list()) == {"1", "2"}


def test_upsert_deduplication(store: OccurrenceStore) -> None:
    """INSERT OR REPLACE: later upsert with same PK wins."""
    store.upsert_many([("run", "doc1", 0, "1", 2)])
    store.upsert_many([("run", "doc1", 0, "2", 3)])
    df = store.query_form("run")
    assert len(df) == 1
    assert df["sense_key"][0] == "2"
    assert df["rating"][0] == 3


def test_query_form_missing_returns_empty(store: OccurrenceStore) -> None:
    df = store.query_form("nonexistent")
    assert len(df) == 0
    assert df.schema == {
        "form": pl.String,
        "doc_id": pl.String,
        "byte_offset": pl.Int64,
        "sense_key": pl.String,
        "rating": pl.Int64,
        "updated_at": pl.String,
    }


def test_to_polars(store: OccurrenceStore) -> None:
    rows = [
        ("cat", "doc1", 0, "1", 3),
        ("dog", "doc2", 5, "1", 2),
    ]
    store.upsert_many(rows)
    df = store.to_polars()
    assert len(df) == 2
    assert set(df["form"].to_list()) == {"cat", "dog"}


def test_to_polars_empty(store: OccurrenceStore) -> None:
    df = store.to_polars()
    assert len(df) == 0


def test_count_by_form(store: OccurrenceStore) -> None:
    rows = [
        ("run", "doc1", 0, "1", 3),  # good
        ("run", "doc1", 10, "1", 2),  # good
        ("run", "doc1", 20, "1", 1),  # bad
        ("walk", "doc2", 0, "1", 0),  # bad
    ]
    store.upsert_many(rows)
    df = store.count_by_form().sort("form")
    run_row = df.filter(pl.col("form") == "run").row(0, named=True)
    assert run_row["n_total"] == 3
    assert run_row["n_good"] == 2
    assert run_row["n_bad"] == 1
    walk_row = df.filter(pl.col("form") == "walk").row(0, named=True)
    assert walk_row["n_total"] == 1
    assert walk_row["n_good"] == 0
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
            ("run", "doc1", 0, uid_a, 3),
            ("run", "doc1", 10, uid_b, 2),
        ]
    )
    store.delete_by_sense_id("run", uid_a)
    df = store.query_form("run")
    assert len(df) == 1
    assert df["sense_key"][0] == uid_b


def test_delete_by_sense_id_removes_subsenses(store: OccurrenceStore) -> None:
    """Deleting by UUID also removes subsense rows (UUID+letter)."""
    uid = "aaaaaaaa-0000-0000-0000-000000000001"
    store.upsert_many(
        [
            ("run", "doc1", 0, uid, 3),
            ("run", "doc1", 10, uid + "a", 2),
            ("run", "doc1", 20, uid + "b", 1),
            ("run", "doc1", 30, "bbbbbbbb-0000-0000-0000-000000000002", 3),
        ]
    )
    store.delete_by_sense_id("run", uid)
    df = store.query_form("run")
    assert len(df) == 1
    assert df["sense_key"][0] == "bbbbbbbb-0000-0000-0000-000000000002"


def test_delete_by_sense_id_other_forms_unaffected(store: OccurrenceStore) -> None:
    """Deleting by UUID for one form does not affect another form."""
    uid = "aaaaaaaa-0000-0000-0000-000000000001"
    store.upsert_many(
        [
            ("run", "doc1", 0, uid, 3),
            ("walk", "doc1", 0, uid, 2),
        ]
    )
    store.delete_by_sense_id("run", uid)
    assert len(store.query_form("run")) == 0
    assert len(store.query_form("walk")) == 1
