"""SQLite-backed store for pending sense changes awaiting human review."""

from datetime import datetime
from enum import Enum
from pathlib import Path
import sqlite3
from typing import Any

from pydantic import BaseModel


class ChangeType(str, Enum):
    rewrite = "rewrite"


class ChangeStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class Change(BaseModel):
    id: str
    type: ChangeType
    form: str
    data: dict[str, Any]
    status: ChangeStatus
    created_at: datetime
    reviewed_at: datetime | None = None


_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS changes (
    id          TEXT PRIMARY KEY,
    type        TEXT NOT NULL,
    form        TEXT NOT NULL,
    data        TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    reviewed_at TEXT
)
"""


class ChangeStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path, timeout=30) as con:
            con.execute(_CREATE_SQL)
            con.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=30)

    def _row_to_change(self, row: tuple) -> Change:  # type: ignore[type-arg]
        id_, type_, form, data, status, created_at, reviewed_at = row
        return Change.model_validate(
            {
                "id": id_,
                "type": type_,
                "form": form,
                "data": __import__("json").loads(data),
                "status": status,
                "created_at": created_at,
                "reviewed_at": reviewed_at,
            }
        )

    def add(self, change: Change) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT INTO changes "
                "(id, type, form, data, status, created_at, reviewed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    change.id,
                    change.type.value,
                    change.form,
                    __import__("json").dumps(change.data),
                    change.status.value,
                    change.created_at.isoformat(),
                    change.reviewed_at.isoformat() if change.reviewed_at else None,
                ),
            )
            con.commit()

    def get(self, id: str) -> Change | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT id, type, form, data, status, created_at, reviewed_at "
                "FROM changes WHERE id = ?",
                (id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_change(row)

    def all_pending(self) -> list[Change]:
        """Return all pending changes, ordered by created_at ascending."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT id, type, form, data, status, created_at, reviewed_at "
                "FROM changes WHERE status = 'pending' ORDER BY created_at ASC"
            ).fetchall()
        return [self._row_to_change(r) for r in rows]

    def set_status(
        self, id: str, status: ChangeStatus, reviewed_at: datetime | None = None
    ) -> None:
        with self._connect() as con:
            con.execute(
                "UPDATE changes SET status = ?, reviewed_at = ? WHERE id = ?",
                (
                    status.value,
                    reviewed_at.isoformat() if reviewed_at else None,
                    id,
                ),
            )
            con.commit()
