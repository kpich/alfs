"""SQLite-backed store for Alf sense entries."""

from collections.abc import Callable
from pathlib import Path
import sqlite3

from alfs.data_models.alf import Alf


class SenseStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        with sqlite3.connect(db_path, timeout=30) as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS senses ("
                "form TEXT PRIMARY KEY, "
                "data TEXT NOT NULL"
                ")"
            )
            con.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=30)

    def read(self, form: str) -> Alf | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT data FROM senses WHERE form = ?", (form,)
            ).fetchone()
        if row is None:
            return None
        return Alf.model_validate_json(row[0])

    def write(self, entry: Alf) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO senses (form, data) VALUES (?, ?)",
                (entry.form, entry.model_dump_json(exclude_none=True)),
            )
            con.commit()

    def update(self, form: str, fn: Callable[[Alf | None], Alf]) -> None:
        """Read-modify-write under BEGIN IMMEDIATE to prevent write-write races."""
        with sqlite3.connect(self._db_path, timeout=30) as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT data FROM senses WHERE form = ?", (form,)
            ).fetchone()
            existing = Alf.model_validate_json(row[0]) if row is not None else None
            updated = fn(existing)
            con.execute(
                "INSERT OR REPLACE INTO senses (form, data) VALUES (?, ?)",
                (form, updated.model_dump_json(exclude_none=True)),
            )
            con.commit()

    def delete(self, form: str) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM senses WHERE form = ?", (form,))
            con.commit()

    def all_forms(self) -> list[str]:
        with self._connect() as con:
            rows = con.execute("SELECT form FROM senses ORDER BY form").fetchall()
        return [r[0] for r in rows]

    def all_entries(self) -> dict[str, Alf]:
        with self._connect() as con:
            rows = con.execute("SELECT form, data FROM senses").fetchall()
        return {form: Alf.model_validate_json(data) for form, data in rows}
