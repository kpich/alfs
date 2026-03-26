"""Tests for prune.run()."""

from pathlib import Path

from alfs.data_models.alf import Alf, Sense
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore
from alfs.update.refinement.prune import run


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    return tmp_path / "senses.db", tmp_path / "labeled.db", tmp_path / "queue"


def test_prune_enqueues_low_quality_sense(tmp_path: Path) -> None:
    senses_db, labeled_db, queue_dir = _setup(tmp_path)

    store = SenseStore(senses_db)
    s1 = Sense(definition="good sense")
    s2 = Sense(definition="bad sense")
    store.write(Alf(form="run", senses=[s1, s2]))

    occ_store = OccurrenceStore(labeled_db)
    # s2 has 5 occurrences all rated 0 (bad) → pct_lt3 = 1.0 > 0.20
    for i in range(5):
        occ_store.upsert_many([("run", f"doc{i}", i * 10, s2.id, 0, None)], model="m")

    result = run(senses_db, labeled_db, queue_dir, n=5)
    assert result == 1
    assert len(list((queue_dir / "pending").glob("*.json"))) == 1


def test_prune_skips_when_would_remove_all_senses(tmp_path: Path) -> None:
    senses_db, labeled_db, queue_dir = _setup(tmp_path)

    store = SenseStore(senses_db)
    s1 = Sense(definition="only sense")
    store.write(Alf(form="run", senses=[s1]))

    occ_store = OccurrenceStore(labeled_db)
    # s1 has bad quality — but it's the only sense, so pruning is unsafe
    for i in range(5):
        occ_store.upsert_many([("run", f"doc{i}", i * 10, s1.id, 0, None)], model="m")

    result = run(senses_db, labeled_db, queue_dir, n=5)
    # Must not enqueue — would remove all senses
    assert result == 0


def test_prune_skips_redirect_entries(tmp_path: Path) -> None:
    senses_db, labeled_db, queue_dir = _setup(tmp_path)

    store = SenseStore(senses_db)
    s1 = Sense(definition="a sense")
    s2 = Sense(definition="another sense")
    store.write(Alf(form="colour", senses=[s1, s2], redirect="color"))

    occ_store = OccurrenceStore(labeled_db)
    for i in range(5):
        occ_store.upsert_many(
            [("colour", f"doc{i}", i * 10, s1.id, 0, None)], model="m"
        )

    result = run(senses_db, labeled_db, queue_dir, n=5)
    assert result == 0
