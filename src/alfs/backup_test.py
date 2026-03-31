"""Tests for write_mutation_log in backup.py."""

from datetime import UTC, datetime
import json
from pathlib import Path
import tempfile
import uuid

from pydantic import TypeAdapter

from alfs.backup import write_mutation_log
from alfs.clerk.request import (
    AddSensesRequest,
    ChangeRequest,
    DeleteEntryRequest,
)
from alfs.data_models.alf import Sense

_adapter: TypeAdapter[ChangeRequest] = TypeAdapter(ChangeRequest)


def _write_done(done_dir: Path, request: ChangeRequest) -> None:
    path = done_dir / f"{request.id}.json"  # type: ignore[union-attr]
    path.write_bytes(_adapter.dump_json(request))


def _make_add_senses(form: str, dt: datetime) -> AddSensesRequest:
    return AddSensesRequest(
        id=str(uuid.uuid4()),
        created_at=dt,
        form=form,
        new_senses=[Sense(definition="test definition")],
    )


def _make_delete(form: str, dt: datetime) -> DeleteEntryRequest:
    return DeleteEntryRequest(
        id=str(uuid.uuid4()),
        created_at=dt,
        form=form,
        reason="test deletion",
    )


def test_creates_monthly_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        queue_dir = Path(tmp) / "queue"
        done_dir = queue_dir / "done"
        done_dir.mkdir(parents=True)
        senses_repo = Path(tmp) / "repo"
        senses_repo.mkdir()

        jan = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
        feb = datetime(2026, 2, 20, 12, 0, tzinfo=UTC)

        req1 = _make_add_senses("dog", jan)
        req2 = _make_delete("Dogs", jan)
        req3 = _make_add_senses("cat", feb)

        _write_done(done_dir, req1)
        _write_done(done_dir, req2)
        _write_done(done_dir, req3)

        write_mutation_log(queue_dir, senses_repo)

        jan_file = senses_repo / "mutations" / "2026-01.jsonl"
        feb_file = senses_repo / "mutations" / "2026-02.jsonl"
        assert jan_file.exists()
        assert feb_file.exists()

        jan_lines = [ln for ln in jan_file.read_text().splitlines() if ln.strip()]
        feb_lines = [ln for ln in feb_file.read_text().splitlines() if ln.strip()]
        assert len(jan_lines) == 2
        assert len(feb_lines) == 1


def test_entries_sorted_by_created_at() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        queue_dir = Path(tmp) / "queue"
        done_dir = queue_dir / "done"
        done_dir.mkdir(parents=True)
        senses_repo = Path(tmp) / "repo"
        senses_repo.mkdir()

        t1 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
        t2 = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)  # earlier, but second file written
        t3 = datetime(2026, 3, 1, 11, 0, tzinfo=UTC)

        _write_done(done_dir, _make_add_senses("a", t1))
        _write_done(done_dir, _make_add_senses("b", t2))
        _write_done(done_dir, _make_add_senses("c", t3))

        write_mutation_log(queue_dir, senses_repo)

        out = senses_repo / "mutations" / "2026-03.jsonl"
        lines = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
        timestamps = [ln["created_at"] for ln in lines]
        assert timestamps == sorted(timestamps)


def test_idempotent_no_duplicates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        queue_dir = Path(tmp) / "queue"
        done_dir = queue_dir / "done"
        done_dir.mkdir(parents=True)
        senses_repo = Path(tmp) / "repo"
        senses_repo.mkdir()

        t = datetime(2026, 3, 10, tzinfo=UTC)
        req = _make_add_senses("dog", t)
        _write_done(done_dir, req)

        write_mutation_log(queue_dir, senses_repo)
        write_mutation_log(queue_dir, senses_repo)  # second call — should not duplicate

        out = senses_repo / "mutations" / "2026-03.jsonl"
        lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1


def test_appends_only_new_entries() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        queue_dir = Path(tmp) / "queue"
        done_dir = queue_dir / "done"
        done_dir.mkdir(parents=True)
        senses_repo = Path(tmp) / "repo"
        senses_repo.mkdir()

        t1 = datetime(2026, 3, 1, tzinfo=UTC)
        t2 = datetime(2026, 3, 2, tzinfo=UTC)

        req1 = _make_add_senses("dog", t1)
        _write_done(done_dir, req1)
        write_mutation_log(queue_dir, senses_repo)

        req2 = _make_delete("dogs", t2)
        _write_done(done_dir, req2)
        write_mutation_log(queue_dir, senses_repo)

        out = senses_repo / "mutations" / "2026-03.jsonl"
        lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
        assert len(lines) == 2


def test_entries_round_trip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        queue_dir = Path(tmp) / "queue"
        done_dir = queue_dir / "done"
        done_dir.mkdir(parents=True)
        senses_repo = Path(tmp) / "repo"
        senses_repo.mkdir()

        t = datetime(2026, 3, 5, tzinfo=UTC)
        req = _make_add_senses("foo", t)
        _write_done(done_dir, req)

        write_mutation_log(queue_dir, senses_repo)

        out = senses_repo / "mutations" / "2026-03.jsonl"
        for line in out.read_text().splitlines():
            if line.strip():
                parsed = _adapter.validate_json(line)
                assert parsed.type == "add_senses"  # type: ignore[union-attr]
