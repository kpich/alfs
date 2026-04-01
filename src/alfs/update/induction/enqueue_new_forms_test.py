"""Tests for enqueue_new_forms."""

from pathlib import Path

import polars as pl
import pytest

from alfs.data_models.alf import Alf
from alfs.data_models.blocklist import Blocklist
from alfs.data_models.induction_queue import InductionQueue
from alfs.data_models.sense_store import SenseStore
from alfs.update.induction.enqueue_new_forms import run


def _make_seg_data(seg_dir: Path, prefix: str, rows: list[dict]) -> None:
    """Create a fake occurrences.parquet for a given prefix."""
    prefix_dir = seg_dir / prefix
    prefix_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        rows, schema={"form": pl.String, "doc_id": pl.String, "byte_offset": pl.Int64}
    )
    df.write_parquet(str(prefix_dir / "occurrences.parquet"))


@pytest.fixture
def seg_dir(tmp_path: Path) -> Path:
    """Seg data with a few forms. Prefix is first character (form_prefix("cat") ==
    "c")."""
    d = tmp_path / "seg"
    # Each form needs at least min_count occurrences
    _make_seg_data(
        d,
        "c",
        [
            {"form": "cat", "doc_id": "d1", "byte_offset": 0},
            {"form": "cat", "doc_id": "d2", "byte_offset": 10},
            {"form": "cat", "doc_id": "d3", "byte_offset": 20},
            {"form": "car", "doc_id": "d1", "byte_offset": 30},
            {"form": "car", "doc_id": "d2", "byte_offset": 40},
            {"form": "car", "doc_id": "d3", "byte_offset": 50},
            {"form": "car", "doc_id": "d4", "byte_offset": 60},
        ],
    )
    _make_seg_data(
        d,
        "d",
        [
            {"form": "dog", "doc_id": "d1", "byte_offset": 0},
            {"form": "dog", "doc_id": "d2", "byte_offset": 10},
            {"form": "dog", "doc_id": "d3", "byte_offset": 20},
            {"form": "dog", "doc_id": "d4", "byte_offset": 30},
            {"form": "dog", "doc_id": "d5", "byte_offset": 40},
        ],
    )
    return d


def test_basic_enqueue(tmp_path: Path, seg_dir: Path):
    senses_db = tmp_path / "senses.db"
    queue_file = tmp_path / "queue.yaml"
    blocklist_file = tmp_path / "blocklist.yaml"

    added = run(seg_dir, senses_db, queue_file, blocklist_file, top_n=10, min_count=3)
    assert added == 3  # cat, car, dog
    entries = InductionQueue(queue_file).load()
    forms = {e.form for e in entries}
    assert forms == {"cat", "car", "dog"}


def test_excludes_senses_db_forms(tmp_path: Path, seg_dir: Path):
    senses_db = tmp_path / "senses.db"
    queue_file = tmp_path / "queue.yaml"
    blocklist_file = tmp_path / "blocklist.yaml"

    # Pre-populate senses.db with "cat"
    SenseStore(senses_db).update("cat", lambda _: Alf(form="cat", senses=[]))

    added = run(seg_dir, senses_db, queue_file, blocklist_file, top_n=10, min_count=3)
    assert added == 2
    forms = {e.form for e in InductionQueue(queue_file).load()}
    assert "cat" not in forms


def test_excludes_blocklisted_forms(tmp_path: Path, seg_dir: Path):
    senses_db = tmp_path / "senses.db"
    queue_file = tmp_path / "queue.yaml"
    blocklist_file = tmp_path / "blocklist.yaml"

    Blocklist(blocklist_file).add("dog")

    added = run(seg_dir, senses_db, queue_file, blocklist_file, top_n=10, min_count=3)
    assert added == 2
    forms = {e.form for e in InductionQueue(queue_file).load()}
    assert "dog" not in forms


def test_excludes_already_queued_forms(tmp_path: Path, seg_dir: Path):
    senses_db = tmp_path / "senses.db"
    queue_file = tmp_path / "queue.yaml"
    blocklist_file = tmp_path / "blocklist.yaml"

    # Pre-add "car" to the queue
    InductionQueue(queue_file).add_forms(["car"])

    added = run(seg_dir, senses_db, queue_file, blocklist_file, top_n=10, min_count=3)
    assert added == 2  # only cat and dog
    forms = {e.form for e in InductionQueue(queue_file).load()}
    assert "car" in forms  # was already there
    assert len(forms) == 3


def test_respects_min_count(tmp_path: Path, seg_dir: Path):
    senses_db = tmp_path / "senses.db"
    queue_file = tmp_path / "queue.yaml"
    blocklist_file = tmp_path / "blocklist.yaml"

    # cat has 3 occurrences, car has 4, dog has 5
    # With min_count=4, only car and dog qualify
    added = run(seg_dir, senses_db, queue_file, blocklist_file, top_n=10, min_count=4)
    forms = {e.form for e in InductionQueue(queue_file).load()}
    assert "cat" not in forms
    assert added == 2


def test_top_n_limit(tmp_path: Path, seg_dir: Path):
    senses_db = tmp_path / "senses.db"
    queue_file = tmp_path / "queue.yaml"
    blocklist_file = tmp_path / "blocklist.yaml"

    added = run(seg_dir, senses_db, queue_file, blocklist_file, top_n=2, min_count=3)
    assert added == 2


def test_sorted_by_frequency(tmp_path: Path, seg_dir: Path):
    """Forms should be enqueued in frequency order (most common first)."""
    senses_db = tmp_path / "senses.db"
    queue_file = tmp_path / "queue.yaml"
    blocklist_file = tmp_path / "blocklist.yaml"

    run(seg_dir, senses_db, queue_file, blocklist_file, top_n=10, min_count=3)
    entries = InductionQueue(queue_file).load()
    forms = [e.form for e in entries]
    # dog (5) > car (4) > cat (3)
    assert forms.index("dog") < forms.index("car")
    assert forms.index("car") < forms.index("cat")


def test_occurrence_refs_attached(tmp_path: Path, seg_dir: Path):
    senses_db = tmp_path / "senses.db"
    queue_file = tmp_path / "queue.yaml"
    blocklist_file = tmp_path / "blocklist.yaml"

    run(
        seg_dir,
        senses_db,
        queue_file,
        blocklist_file,
        top_n=10,
        min_count=3,
        n_occurrence_refs=2,
    )
    entries = {e.form: e for e in InductionQueue(queue_file).load()}
    assert len(entries["cat"].occurrences) == 2
    assert len(entries["dog"].occurrences) == 2


def test_excludes_case_variants_of_known_forms(tmp_path: Path, seg_dir: Path):
    senses_db = tmp_path / "senses.db"
    queue_file = tmp_path / "queue.yaml"
    blocklist_file = tmp_path / "blocklist.yaml"

    # "CAT" in senses.db should block "cat" from the corpus (case variant)
    SenseStore(senses_db).update("CAT", lambda _: Alf(form="CAT", senses=[]))

    added = run(seg_dir, senses_db, queue_file, blocklist_file, top_n=10, min_count=3)
    forms = {e.form for e in InductionQueue(queue_file).load()}
    assert "cat" not in forms
    assert added == 2  # car and dog, not cat


def test_excludes_sentence_initial_capitalization(tmp_path: Path):
    senses_db = tmp_path / "senses.db"
    queue_file = tmp_path / "queue.yaml"
    blocklist_file = tmp_path / "blocklist.yaml"
    seg_dir = tmp_path / "seg"

    # Simulate "What" appearing capitalized at sentence starts in corpus
    _make_seg_data(
        seg_dir,
        "W",
        [
            {"form": "What", "doc_id": "d1", "byte_offset": 0},
            {"form": "What", "doc_id": "d2", "byte_offset": 0},
            {"form": "What", "doc_id": "d3", "byte_offset": 0},
            {"form": "What", "doc_id": "d4", "byte_offset": 0},
            {"form": "What", "doc_id": "d5", "byte_offset": 0},
        ],
    )

    # "what" (lowercase) is already in the dictionary
    SenseStore(senses_db).update("what", lambda _: Alf(form="what", senses=[]))

    added = run(seg_dir, senses_db, queue_file, blocklist_file, top_n=10, min_count=3)
    forms = {e.form for e in InductionQueue(queue_file).load()}
    assert "What" not in forms
    assert added == 0


def test_idempotent_double_run(tmp_path: Path, seg_dir: Path):
    senses_db = tmp_path / "senses.db"
    queue_file = tmp_path / "queue.yaml"
    blocklist_file = tmp_path / "blocklist.yaml"

    run(seg_dir, senses_db, queue_file, blocklist_file, top_n=10, min_count=3)
    added_second = run(
        seg_dir, senses_db, queue_file, blocklist_file, top_n=10, min_count=3
    )
    assert added_second == 0  # nothing new
