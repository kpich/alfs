"""Tests for cleanup.run()."""

from pathlib import Path

from alfs.clerk.queue import drain
from alfs.data_models.alf import Alf, Sense
from alfs.data_models.sense_store import SenseStore
from alfs.update.refinement.cleanup import run


def test_cleanup_enqueues_redirect_entries_with_senses(tmp_path: Path) -> None:
    store = SenseStore(tmp_path / "senses.db")
    # This entry should be cleared: has redirect AND senses
    store.write(
        Alf(form="colour", senses=[Sense(definition="a colour")], redirect="color")
    )
    # This entry should be left alone: no redirect
    store.write(Alf(form="color", senses=[Sense(definition="a color")]))

    queue_dir = tmp_path / "queue"
    result = run(tmp_path / "senses.db", queue_dir)

    assert result == 1
    assert len(list((queue_dir / "pending").glob("*.json"))) == 1

    drain(queue_dir, store, None)
    entry = store.read("colour")
    assert entry is not None
    assert entry.senses == []


def test_cleanup_skips_redirect_with_no_senses(tmp_path: Path) -> None:
    store = SenseStore(tmp_path / "senses.db")
    # Redirect with no senses — already clean, should not be touched
    store.write(Alf(form="colour", senses=[], redirect="color"))

    queue_dir = tmp_path / "queue"
    result = run(tmp_path / "senses.db", queue_dir)

    assert result == 0
    assert (
        not (queue_dir / "pending").exists()
        or len(list((queue_dir / "pending").glob("*.json"))) == 0
    )
