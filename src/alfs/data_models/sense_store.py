"""SQLite-backed store for Alf sense entries."""

from collections import defaultdict
from collections.abc import Callable, Generator
from contextlib import contextmanager
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
                "spelling_variant_of TEXT, "
                "updated_at TEXT"
                ")"
            )
            existing_cols = {
                row[1] for row in con.execute("PRAGMA table_info(wordforms)").fetchall()
            }
            if "spelling_variant_of" not in existing_cols:
                con.execute("ALTER TABLE wordforms ADD COLUMN spelling_variant_of TEXT")
            if "redirect" in existing_cols:
                con.execute("ALTER TABLE wordforms DROP COLUMN redirect")
            con.execute(
                "CREATE TABLE IF NOT EXISTS senses ("
                "id TEXT PRIMARY KEY, "
                "form TEXT NOT NULL REFERENCES wordforms(form) ON DELETE CASCADE, "
                "position INTEGER NOT NULL, "
                "definition TEXT NOT NULL, "
                "pos TEXT, "
                "morph_base TEXT, "
                "morph_relation TEXT, "
                "updated_by_model TEXT, "
                "updated_at TEXT, "
                "UNIQUE (form, position)"
                ")"
            )
            existing_sense_cols = {
                row[1] for row in con.execute("PRAGMA table_info(senses)").fetchall()
            }
            if "subsenses" in existing_sense_cols:
                con.execute("ALTER TABLE senses DROP COLUMN subsenses")
            con.commit()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection]:
        con = sqlite3.connect(self._db_path, timeout=30)
        con.execute("PRAGMA foreign_keys = ON")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _assemble(self, con: sqlite3.Connection, form: str) -> Alf | None:
        wf = con.execute(
            "SELECT spelling_variant_of FROM wordforms WHERE form = ?",
            (form,),
        ).fetchone()
        if wf is None:
            return None
        rows = con.execute(
            "SELECT id, definition, pos, morph_base, morph_relation,"
            " updated_by_model, updated_at"
            " FROM senses WHERE form = ? ORDER BY position",
            (form,),
        ).fetchall()
        senses = [
            Sense(
                id=r[0],
                definition=r[1],
                pos=PartOfSpeech(r[2]) if r[2] else None,
                morph_base=r[3],
                morph_relation=r[4],
                updated_by_model=r[5],
                updated_at=r[6],
            )
            for r in rows
        ]
        return Alf(form=form, senses=senses, spelling_variant_of=wf[0])

    def _write_entry(self, con: sqlite3.Connection, entry: Alf) -> None:
        """Write entry within an already-open transaction."""
        con.execute(
            "INSERT INTO wordforms (form, spelling_variant_of, updated_at)"
            " VALUES (?, ?, CURRENT_TIMESTAMP)"
            " ON CONFLICT(form) DO UPDATE SET"
            " spelling_variant_of=excluded.spelling_variant_of,"
            " updated_at=CURRENT_TIMESTAMP",
            (entry.form, entry.spelling_variant_of),
        )
        existing = {
            r[0]: r[1:]
            for r in con.execute(
                "SELECT id, definition, pos, morph_base, morph_relation,"
                " updated_by_model, updated_at FROM senses WHERE form = ?",
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
            pos_val = sense.pos.value if sense.pos else None
            if sense.id in existing:
                old = existing[sense.id]
                content_changed = (old[0], old[1], old[2], old[3], old[4]) != (
                    sense.definition,
                    pos_val,
                    sense.morph_base,
                    sense.morph_relation,
                    sense.updated_by_model,
                )
                con.execute(
                    "UPDATE senses SET form=?, position=?, definition=?, pos=?, "
                    "morph_base=?, morph_relation=?, updated_by_model=?"
                    + (", updated_at=CURRENT_TIMESTAMP" if content_changed else "")
                    + " WHERE id=?",
                    (
                        entry.form,
                        pos_idx,
                        sense.definition,
                        pos_val,
                        sense.morph_base,
                        sense.morph_relation,
                        sense.updated_by_model,
                        sense.id,
                    ),
                )
            else:
                con.execute(
                    "INSERT INTO senses (id, form, position, definition, pos, "
                    "morph_base, morph_relation, updated_by_model, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "form=excluded.form, position=excluded.position, "
                    "definition=excluded.definition, pos=excluded.pos, "
                    "morph_base=excluded.morph_base, "
                    "morph_relation=excluded.morph_relation, "
                    "updated_by_model=excluded.updated_by_model, "
                    "updated_at=CURRENT_TIMESTAMP",
                    (
                        sense.id,
                        entry.form,
                        pos_idx,
                        sense.definition,
                        pos_val,
                        sense.morph_base,
                        sense.morph_relation,
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
            wf_rows = con.execute(
                "SELECT form, spelling_variant_of FROM wordforms"
            ).fetchall()
            sense_rows = con.execute(
                "SELECT form, id, definition, pos, morph_base,"
                " morph_relation, updated_by_model, updated_at"
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
                updated_by_model,
                updated_at,
            ) = row
            senses_by_form[form].append(
                Sense(
                    id=id_,
                    definition=definition,
                    pos=PartOfSpeech(pos) if pos else None,
                    morph_base=morph_base,
                    morph_relation=morph_relation,
                    updated_by_model=updated_by_model,
                    updated_at=updated_at,
                )
            )
        return {
            form: Alf(
                form=form,
                senses=senses_by_form.get(form, []),
                spelling_variant_of=spelling_variant_of,
            )
            for form, spelling_variant_of in wf_rows
        }

    def read_case_variants(self, form: str) -> list[Alf]:
        """Return all entries whose form lowercases to the same string as form."""
        form_lower = form.lower()
        with self._connect() as con:
            rows = con.execute("SELECT form FROM wordforms").fetchall()
        variants = [r[0] for r in rows if r[0].lower() == form_lower]
        result = []
        for variant_form in variants:
            entry = self.read(variant_form)
            if entry is not None:
                result.append(entry)
        return result

    def sense_id_to_form(self) -> dict[str, str]:
        """Return {sense_id: form} for every sense in the store."""
        with self._connect() as con:
            rows = con.execute("SELECT id, form FROM senses").fetchall()
        return {r[0]: r[1] for r in rows}

    def max_sense_updated_at_by_form(self) -> dict[str, str]:
        """Return {form: max(updated_at)} for forms with at least one
        timestamped sense."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT form, MAX(updated_at) FROM senses "
                "WHERE updated_at IS NOT NULL GROUP BY form"
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def all_timestamps(self) -> dict[str, str | None]:
        with self._connect() as con:
            rows = con.execute("SELECT form, updated_at FROM wordforms").fetchall()
        return dict(rows)
