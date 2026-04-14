"""Tests for MWE candidate enqueuing."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from alfs.data_models.blocklist import Blocklist
from alfs.data_models.mwe_queue import MWEQueue
from alfs.data_models.sense_store import SenseStore
from alfs.mwe.enqueue_candidates import run


def _write_pmi(path: Path, rows: list[dict]) -> None:
    if not rows:
        df = pl.DataFrame(
            schema={
                "form": pl.String,
                "components": pl.List(pl.String),
                "count": pl.Int64,
                "pmi": pl.Float64,
            }
        )
    else:
        df = pl.DataFrame(rows)
    df.write_parquet(str(path))


def test_enqueue_basic(tmp_path: Path):
    pmi_path = tmp_path / "pmi.parquet"
    _write_pmi(
        pmi_path,
        [
            {
                "form": "a priori",
                "components": ["a", "priori"],
                "count": 100,
                "pmi": 12.0,
            },
            {
                "form": "take care",
                "components": ["take", "care"],
                "count": 50,
                "pmi": 8.0,
            },
        ],
    )

    senses_db = tmp_path / "senses.db"
    SenseStore(senses_db)  # creates empty db

    bl_path = tmp_path / "blocklist.yaml"
    Blocklist(bl_path).save({})

    queue_path = tmp_path / "mwe_queue.yaml"

    added = run(
        pmi_path,
        senses_db,
        bl_path,
        queue_path,
        top_n=10,
    )
    assert added == 2
    entries = MWEQueue(queue_path).load()
    assert len(entries) == 2
    assert entries[0].form == "a priori"


def test_enqueue_excludes_known(tmp_path: Path):
    pmi_path = tmp_path / "pmi.parquet"
    _write_pmi(
        pmi_path,
        [
            {
                "form": "a priori",
                "components": ["a", "priori"],
                "count": 100,
                "pmi": 12.0,
            },
            {
                "form": "take care",
                "components": ["take", "care"],
                "count": 50,
                "pmi": 8.0,
            },
        ],
    )

    senses_db = tmp_path / "senses.db"
    store = SenseStore(senses_db)
    from alfs.data_models.alf import Alf, Sense

    store.write(Alf(form="a priori", senses=[Sense(definition="test")]))

    bl_path = tmp_path / "blocklist.yaml"
    Blocklist(bl_path).save({})

    queue_path = tmp_path / "mwe_queue.yaml"

    added = run(pmi_path, senses_db, bl_path, queue_path, top_n=10)
    assert added == 1
    entries = MWEQueue(queue_path).load()
    assert entries[0].form == "take care"


def test_enqueue_excludes_blocklisted(tmp_path: Path):
    pmi_path = tmp_path / "pmi.parquet"
    _write_pmi(
        pmi_path,
        [
            {
                "form": "a priori",
                "components": ["a", "priori"],
                "count": 100,
                "pmi": 12.0,
            },
        ],
    )

    senses_db = tmp_path / "senses.db"
    SenseStore(senses_db)

    bl_path = tmp_path / "blocklist.yaml"
    bl = Blocklist(bl_path)
    bl.save({})
    bl.add("a priori", "test reason")

    queue_path = tmp_path / "mwe_queue.yaml"

    added = run(pmi_path, senses_db, bl_path, queue_path, top_n=10)
    assert added == 0


def test_enqueue_empty_pmi(tmp_path: Path):
    pmi_path = tmp_path / "pmi.parquet"
    _write_pmi(pmi_path, [])

    senses_db = tmp_path / "senses.db"
    SenseStore(senses_db)

    bl_path = tmp_path / "blocklist.yaml"
    Blocklist(bl_path).save({})

    queue_path = tmp_path / "mwe_queue.yaml"

    added = run(pmi_path, senses_db, bl_path, queue_path, top_n=10)
    assert added == 0
