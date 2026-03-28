"""Tests for enqueue_poor_coverage."""

from pathlib import Path

from alfs.data_models.blocklist import Blocklist
from alfs.data_models.induction_queue import InductionQueue
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.update.induction.enqueue_poor_coverage import run


def _populate_labeled(labeled_db: Path, rows: list[tuple]) -> None:
    """Populate labeled.db with (form, doc_id, byte_offset, sense_key, rating) rows."""
    store = OccurrenceStore(labeled_db)
    # upsert_many takes (form, doc_id, byte_offset, sense_key, rating, synonyms)
    store.upsert_many([(r[0], r[1], r[2], r[3], r[4], None) for r in rows], "test")


def test_basic_enqueue(tmp_path: Path):
    labeled_db = tmp_path / "labeled.db"
    queue_file = tmp_path / "queue.yaml"
    blocklist_file = tmp_path / "blocklist.yaml"

    _populate_labeled(
        labeled_db,
        [
            ("cat", "d1", 0, "1", 0),  # poor
            ("cat", "d2", 10, "1", 0),  # poor
            ("dog", "d1", 0, "1", 1),  # okay but still bad
        ],
    )

    added = run(labeled_db, queue_file, blocklist_file, min_bad=1)
    assert added == 2
    forms = {e.form for e in InductionQueue(queue_file).load()}
    assert forms == {"cat", "dog"}


def test_skip_excluded_from_poor_coverage(tmp_path: Path):
    labeled_db = tmp_path / "labeled.db"
    queue_file = tmp_path / "queue.yaml"
    blocklist_file = tmp_path / "blocklist.yaml"

    _populate_labeled(
        labeled_db,
        [
            ("weird", "d1", 0, "_skip", 0),  # skip label — should not count as poor
            ("cat", "d1", 0, "1", 0),  # genuinely poor
        ],
    )

    added = run(labeled_db, queue_file, blocklist_file, min_bad=1)
    forms = {e.form for e in InductionQueue(queue_file).load()}
    assert "weird" not in forms
    assert "cat" in forms
    assert added == 1


def test_rating_2_excluded(tmp_path: Path):
    labeled_db = tmp_path / "labeled.db"
    queue_file = tmp_path / "queue.yaml"
    blocklist_file = tmp_path / "blocklist.yaml"

    _populate_labeled(
        labeled_db,
        [
            ("cat", "d1", 0, "1", 2),  # excellent — not poor
        ],
    )

    added = run(labeled_db, queue_file, blocklist_file, min_bad=1)
    assert added == 0


def test_blocklisted_excluded(tmp_path: Path):
    labeled_db = tmp_path / "labeled.db"
    queue_file = tmp_path / "queue.yaml"
    blocklist_file = tmp_path / "blocklist.yaml"

    _populate_labeled(
        labeled_db,
        [
            ("cat", "d1", 0, "1", 0),
            ("thrumbly", "d1", 0, "1", 0),
        ],
    )
    Blocklist(blocklist_file).add("thrumbly")

    run(labeled_db, queue_file, blocklist_file, min_bad=1)
    forms = {e.form for e in InductionQueue(queue_file).load()}
    assert "thrumbly" not in forms
    assert "cat" in forms


def test_already_queued_excluded(tmp_path: Path):
    labeled_db = tmp_path / "labeled.db"
    queue_file = tmp_path / "queue.yaml"
    blocklist_file = tmp_path / "blocklist.yaml"

    _populate_labeled(
        labeled_db,
        [
            ("cat", "d1", 0, "1", 0),
            ("dog", "d1", 0, "1", 0),
        ],
    )
    InductionQueue(queue_file).add_forms(["cat"])

    added = run(labeled_db, queue_file, blocklist_file, min_bad=1)
    assert added == 1
    forms = {e.form for e in InductionQueue(queue_file).load()}
    assert "dog" in forms


def test_min_bad_threshold(tmp_path: Path):
    labeled_db = tmp_path / "labeled.db"
    queue_file = tmp_path / "queue.yaml"
    blocklist_file = tmp_path / "blocklist.yaml"

    _populate_labeled(
        labeled_db,
        [
            ("cat", "d1", 0, "1", 0),  # 1 bad — below min_bad=2
            ("dog", "d1", 0, "1", 0),  # 2 bad — at threshold
            ("dog", "d2", 10, "1", 0),
        ],
    )

    added = run(labeled_db, queue_file, blocklist_file, min_bad=2)
    assert added == 1
    forms = {e.form for e in InductionQueue(queue_file).load()}
    assert "dog" in forms
    assert "cat" not in forms


def test_occurrence_refs_are_bad_occurrences(tmp_path: Path):
    labeled_db = tmp_path / "labeled.db"
    queue_file = tmp_path / "queue.yaml"
    blocklist_file = tmp_path / "blocklist.yaml"

    _populate_labeled(
        labeled_db,
        [
            ("cat", "d1", 100, "1", 0),  # poor
            ("cat", "d2", 200, "1", 2),  # excellent — not included
        ],
    )

    run(labeled_db, queue_file, blocklist_file, min_bad=1)
    entries = {e.form: e for e in InductionQueue(queue_file).load()}
    refs = {(o.doc_id, o.byte_offset) for o in entries["cat"].occurrences}
    assert ("d1", 100) in refs
    assert ("d2", 200) not in refs  # only poor ones


def test_sorted_by_bad_count(tmp_path: Path):
    labeled_db = tmp_path / "labeled.db"
    queue_file = tmp_path / "queue.yaml"
    blocklist_file = tmp_path / "blocklist.yaml"

    # dog has 3 bad, cat has 1 bad
    _populate_labeled(
        labeled_db,
        [
            ("cat", "d1", 0, "1", 0),
            ("dog", "d1", 0, "1", 0),
            ("dog", "d2", 10, "1", 0),
            ("dog", "d3", 20, "1", 0),
        ],
    )

    run(labeled_db, queue_file, blocklist_file, min_bad=1)
    entries = InductionQueue(queue_file).load()
    forms = [e.form for e in entries]
    assert forms.index("dog") < forms.index("cat")


def test_top_n_limit(tmp_path: Path):
    labeled_db = tmp_path / "labeled.db"
    queue_file = tmp_path / "queue.yaml"
    blocklist_file = tmp_path / "blocklist.yaml"

    _populate_labeled(
        labeled_db,
        [
            ("cat", "d1", 0, "1", 0),
            ("dog", "d1", 0, "1", 0),
            ("fish", "d1", 0, "1", 0),
        ],
    )

    added = run(labeled_db, queue_file, blocklist_file, min_bad=1, top_n=2)
    assert added == 2


def test_idempotent_double_run(tmp_path: Path):
    labeled_db = tmp_path / "labeled.db"
    queue_file = tmp_path / "queue.yaml"
    blocklist_file = tmp_path / "blocklist.yaml"

    _populate_labeled(labeled_db, [("cat", "d1", 0, "1", 0)])

    run(labeled_db, queue_file, blocklist_file, min_bad=1)
    added_second = run(labeled_db, queue_file, blocklist_file, min_bad=1)
    assert added_second == 0
