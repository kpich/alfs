"""SQLite-backed store for labeled occurrence data (WAL mode)."""

from collections.abc import Iterable
from pathlib import Path
import sqlite3

import polars as pl

_SCHEMA = {
    "form": pl.String,
    "doc_id": pl.String,
    "byte_offset": pl.Int64,
    "sense_key": pl.String,
    "rating": pl.Int64,
}

_COUNT_SCHEMA = {
    "form": pl.String,
    "n_total": pl.Int64,
    "n_good": pl.Int64,
    "n_bad": pl.Int64,
}


class OccurrenceStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        with sqlite3.connect(db_path, timeout=30) as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute(
                "CREATE TABLE IF NOT EXISTS labeled ("
                "form        TEXT    NOT NULL, "
                "doc_id      TEXT    NOT NULL, "
                "byte_offset INTEGER NOT NULL, "
                "sense_key   TEXT    NOT NULL, "
                "rating      INTEGER NOT NULL CHECK (rating IN (0, 1, 2, 3)), "
                "PRIMARY KEY (form, doc_id, byte_offset)"
                ")"
            )
            con.commit()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._db_path, timeout=30)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def upsert_many(self, rows: Iterable[tuple[str, str, int, str, int]]) -> None:
        with self._connect() as con:
            con.executemany(
                "INSERT OR REPLACE INTO labeled "
                "(form, doc_id, byte_offset, sense_key, rating) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            con.commit()

    def query_form(self, form: str) -> pl.DataFrame:
        with self._connect() as con:
            rows = con.execute(
                "SELECT form, doc_id, byte_offset, sense_key, rating "
                "FROM labeled WHERE form = ?",
                (form,),
            ).fetchall()
        if not rows:
            return pl.DataFrame(schema=_SCHEMA)
        return pl.DataFrame(rows, schema=_SCHEMA, orient="row")

    def to_polars(self) -> pl.DataFrame:
        with self._connect() as con:
            rows = con.execute(
                "SELECT form, doc_id, byte_offset, sense_key, rating FROM labeled"
            ).fetchall()
        if not rows:
            return pl.DataFrame(schema=_SCHEMA)
        return pl.DataFrame(rows, schema=_SCHEMA, orient="row")

    def count_by_form(self) -> pl.DataFrame:
        with self._connect() as con:
            rows = con.execute(
                "SELECT form, COUNT(*) as n_total, "
                "SUM(CASE WHEN rating IN (2, 3) THEN 1 ELSE 0 END) as n_good, "
                "SUM(CASE WHEN rating IN (0, 1) THEN 1 ELSE 0 END) as n_bad "
                "FROM labeled GROUP BY form"
            ).fetchall()
        if not rows:
            return pl.DataFrame(schema=_COUNT_SCHEMA)
        return pl.DataFrame(rows, schema=_COUNT_SCHEMA, orient="row")
