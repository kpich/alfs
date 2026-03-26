"""SQLite-backed store for labeled occurrence data (WAL mode)."""

from collections.abc import Iterable
import contextlib
import json
from pathlib import Path
import sqlite3

import polars as pl

_SCHEMA = {
    "form": pl.String,
    "doc_id": pl.String,
    "byte_offset": pl.Int64,
    "sense_key": pl.String,
    "rating": pl.Int64,
    "model": pl.String,
    "updated_at": pl.String,
    "synonyms": pl.String,  # NULL=missing, '[]'=none, '["a","b"]'=list (JSON)
}

_COUNT_SCHEMA = {
    "form": pl.String,
    "n_total": pl.Int64,
    "n_bad": pl.Int64,
    "n_excellent": pl.Int64,
}


class OccurrenceStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path, timeout=30) as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute(
                "CREATE TABLE IF NOT EXISTS models ("
                "id   INTEGER PRIMARY KEY AUTOINCREMENT, "
                "name TEXT NOT NULL UNIQUE"
                ")"
            )
            con.execute(
                "CREATE TABLE IF NOT EXISTS labeled ("
                "form        TEXT    NOT NULL, "
                "doc_id      TEXT    NOT NULL, "
                "byte_offset INTEGER NOT NULL, "
                "sense_key   TEXT    NOT NULL, "
                "rating      INTEGER NOT NULL CHECK (rating IN (0, 1, 2)), "
                "updated_at  TEXT, "
                "PRIMARY KEY (form, doc_id, byte_offset)"
                ")"
            )
            with contextlib.suppress(sqlite3.OperationalError):
                con.execute(
                    "ALTER TABLE labeled ADD COLUMN"
                    " model_id INTEGER REFERENCES models(id)"
                )
            with contextlib.suppress(sqlite3.OperationalError):
                con.execute("ALTER TABLE labeled ADD COLUMN synonyms TEXT")
            con.commit()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._db_path, timeout=30)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _get_or_create_model_id(self, con: sqlite3.Connection, model: str) -> int:
        con.execute("INSERT OR IGNORE INTO models (name) VALUES (?)", (model,))
        row = con.execute("SELECT id FROM models WHERE name = ?", (model,)).fetchone()
        return int(row[0])

    def upsert_many(
        self,
        rows: Iterable[tuple[str, str, int, str, int, list[str] | None]],
        model: str,
    ) -> None:
        with self._connect() as con:
            model_id = self._get_or_create_model_id(con, model)
            con.executemany(
                "INSERT OR REPLACE INTO labeled "
                "(form, doc_id, byte_offset, sense_key, rating, model_id, updated_at,"
                " synonyms) "
                "VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)",
                (
                    (
                        f,
                        d,
                        b,
                        s,
                        r,
                        model_id,
                        json.dumps(syns) if syns is not None else None,
                    )
                    for f, d, b, s, r, syns in rows
                ),
            )
            con.commit()

    def query_form(self, form: str) -> pl.DataFrame:
        with self._connect() as con:
            rows = con.execute(
                "SELECT l.form, l.doc_id, l.byte_offset, l.sense_key, l.rating, "
                "m.name as model, l.updated_at, l.synonyms "
                "FROM labeled l LEFT JOIN models m ON l.model_id = m.id "
                "WHERE l.form = ?",
                (form,),
            ).fetchall()
        if not rows:
            return pl.DataFrame(schema=_SCHEMA)
        return pl.DataFrame(rows, schema=_SCHEMA, orient="row")

    def to_polars(self) -> pl.DataFrame:
        with self._connect() as con:
            rows = con.execute(
                "SELECT l.form, l.doc_id, l.byte_offset, l.sense_key, l.rating, "
                "m.name as model, l.updated_at, l.synonyms "
                "FROM labeled l LEFT JOIN models m ON l.model_id = m.id"
            ).fetchall()
        if not rows:
            return pl.DataFrame(schema=_SCHEMA)
        return pl.DataFrame(rows, schema=_SCHEMA, orient="row")

    def delete_by_sense_id(self, form: str, sense_id: str) -> None:
        """Delete all occurrences for a sense."""
        with self._connect() as con:
            con.execute(
                "DELETE FROM labeled WHERE form = ? AND sense_key = ?",
                (form, sense_id),
            )
            con.commit()

    def delete_by_form(self, form: str) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM labeled WHERE form = ?", (form,))
            con.commit()

    def count_by_form(self) -> pl.DataFrame:
        with self._connect() as con:
            rows = con.execute(
                "SELECT form, COUNT(*) as n_total, "
                "SUM(CASE WHEN rating = 0 THEN 1 ELSE 0 END) as n_bad, "
                "SUM(CASE WHEN rating = 2 THEN 1 ELSE 0 END) as n_excellent "
                "FROM labeled GROUP BY form"
            ).fetchall()
        if not rows:
            return pl.DataFrame(schema=_COUNT_SCHEMA)
        return pl.DataFrame(rows, schema=_COUNT_SCHEMA, orient="row")
