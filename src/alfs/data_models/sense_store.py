"""SQLite-backed store for Alf sense entries."""

from collections import defaultdict
from collections.abc import Callable
import json
from pathlib import Path
import sqlite3

from alfs.data_models.alf import Alf, Sense
from alfs.data_models.pos import PartOfSpeech


class SenseStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS wordforms ("
                "form TEXT PRIMARY KEY, "
                "redirect TEXT, "
                "updated_at TEXT"
                ")"
            )
            con.execute(
                "CREATE TABLE IF NOT EXISTS senses ("
                "id TEXT PRIMARY KEY, "
                "form TEXT NOT NULL REFERENCES wordforms(form) ON DELETE CASCADE, "
                "position INTEGER NOT NULL, "
                "definition TEXT NOT NULL, "
                "pos TEXT, "
                "morph_base TEXT, "
                "morph_relation TEXT, "
                "subsenses TEXT, "
                "updated_by_model TEXT, "
                "updated_at TEXT, "
                "UNIQUE (form, position)"
                ")"
            )
            con.commit()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._db_path, timeout=30)
        con.execute("PRAGMA foreign_keys = ON")
        return con

    def _assemble(self, con: sqlite3.Connection, form: str) -> Alf | None:
        wf = con.execute(
            "SELECT redirect FROM wordforms WHERE form = ?", (form,)
        ).fetchone()
        if wf is None:
            return None
        rows = con.execute(
            "SELECT id, definition, pos, morph_base, morph_relation, subsenses,"
            " updated_by_model FROM senses WHERE form = ? ORDER BY position",
            (form,),
        ).fetchall()
        senses = [
            Sense(
                id=r[0],
                definition=r[1],
                pos=PartOfSpeech(r[2]) if r[2] else None,
                morph_base=r[3],
                morph_relation=r[4],
                subsenses=json.loads(r[5]) if r[5] else None,
                updated_by_model=r[6],
            )
            for r in rows
        ]
        return Alf(form=form, senses=senses, redirect=wf[0])

    def _write_entry(self, con: sqlite3.Connection, entry: Alf) -> None:
        """Write entry within an already-open transaction."""
        con.execute(
            "INSERT INTO wordforms (form, redirect, updated_at)"
            " VALUES (?, ?, CURRENT_TIMESTAMP)"
            " ON CONFLICT(form) DO UPDATE SET"
            " redirect=excluded.redirect, updated_at=CURRENT_TIMESTAMP",
            (entry.form, entry.redirect),
        )
        existing = {
            r[0]: r[1:]
            for r in con.execute(
                "SELECT id, definition, pos, morph_base, morph_relation,"
                " subsenses, updated_by_model, updated_at FROM senses WHERE form = ?",
                (entry.form,),
            ).fetchall()
        }
        # Delete removed senses first to free up positions before
        # repositioning survivors.
        new_ids = {sense.id for sense in entry.senses}
        removed = set(existing) - new_ids
        if removed:
            con.executemany(
                "DELETE FROM senses WHERE id = ?", [(sid,) for sid in removed]
            )
        for pos_idx, sense in enumerate(entry.senses):
            subsenses_json = json.dumps(sense.subsenses) if sense.subsenses else None
            pos_val = sense.pos.value if sense.pos else None
            if sense.id in existing:
                old = existing[sense.id]
                content_changed = (old[0], old[1], old[2], old[3], old[4], old[5]) != (
                    sense.definition,
                    pos_val,
                    sense.morph_base,
                    sense.morph_relation,
                    subsenses_json,
                    sense.updated_by_model,
                )
                con.execute(
                    "UPDATE senses SET form=?, position=?, definition=?, pos=?, "
                    "morph_base=?, morph_relation=?, subsenses=?, updated_by_model=?"
                    + (", updated_at=CURRENT_TIMESTAMP" if content_changed else "")
                    + " WHERE id=?",
                    (
                        entry.form,
                        pos_idx,
                        sense.definition,
                        pos_val,
                        sense.morph_base,
                        sense.morph_relation,
                        subsenses_json,
                        sense.updated_by_model,
                        sense.id,
                    ),
                )
            else:
                con.execute(
                    "INSERT INTO senses (id, form, position, definition, pos, "
                    "morph_base, morph_relation, subsenses, updated_by_model,"
                    " updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                    (
                        sense.id,
                        entry.form,
                        pos_idx,
                        sense.definition,
                        pos_val,
                        sense.morph_base,
                        sense.morph_relation,
                        subsenses_json,
                        sense.updated_by_model,
                    ),
                )

    def read(self, form: str) -> Alf | None:
        with self._connect() as con:
            return self._assemble(con, form)

    def write(self, entry: Alf) -> None:
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            self._write_entry(con, entry)
            con.commit()

    def update(self, form: str, fn: Callable[[Alf | None], Alf]) -> None:
        """Read-modify-write under BEGIN IMMEDIATE to prevent write-write races."""
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            existing = self._assemble(con, form)
            updated = fn(existing)
            self._write_entry(con, updated)
            con.commit()

    def delete(self, form: str) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM wordforms WHERE form = ?", (form,))
            con.commit()

    def all_forms(self) -> list[str]:
        with self._connect() as con:
            rows = con.execute("SELECT form FROM wordforms ORDER BY form").fetchall()
        return [r[0] for r in rows]

    def all_entries(self) -> dict[str, Alf]:
        with self._connect() as con:
            wf_rows = con.execute("SELECT form, redirect FROM wordforms").fetchall()
            sense_rows = con.execute(
                "SELECT form, id, definition, pos, morph_base,"
                " morph_relation, subsenses, updated_by_model"
                " FROM senses ORDER BY form, position"
            ).fetchall()
        senses_by_form: dict[str, list[Sense]] = defaultdict(list)
        for row in sense_rows:
            (
                form,
                id_,
                definition,
                pos,
                morph_base,
                morph_relation,
                subsenses,
                updated_by_model,
            ) = row
            senses_by_form[form].append(
                Sense(
                    id=id_,
                    definition=definition,
                    pos=PartOfSpeech(pos) if pos else None,
                    morph_base=morph_base,
                    morph_relation=morph_relation,
                    subsenses=json.loads(subsenses) if subsenses else None,
                    updated_by_model=updated_by_model,
                )
            )
        return {
            form: Alf(form=form, senses=senses_by_form.get(form, []), redirect=redirect)
            for form, redirect in wf_rows
        }

    def all_timestamps(self) -> dict[str, str | None]:
        with self._connect() as con:
            rows = con.execute("SELECT form, updated_at FROM wordforms").fetchall()
        return dict(rows)
