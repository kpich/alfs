"""Tests for CC apply module."""

import json
from pathlib import Path

from alfs.cc.apply import run
from alfs.cc.models import (
    CCInductionOutput,
    CCInductionTask,
    ContextLabel,
    InductionSense,
)
from alfs.data_models.alf import Alf
from alfs.data_models.occurrence import Occurrence
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    cc_dir = tmp_path / "cc_tasks"
    (cc_dir / "pending" / "induction").mkdir(parents=True)
    (cc_dir / "done" / "induction").mkdir(parents=True)
    senses_db = tmp_path / "senses.db"
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    return cc_dir, senses_db, queue_dir


def test_apply_induction_new_senses(tmp_path: Path):
    cc_dir, senses_db, queue_dir = _setup(tmp_path)

    store = SenseStore(senses_db)
    store.update("cat", lambda _: Alf(form="cat", senses=[]))

    output = CCInductionOutput(
        id="test1",
        form="cat",
        new_senses=[
            InductionSense(definition="a small domesticated feline", pos="noun")
        ],
    )
    (cc_dir / "done" / "induction" / "test1.json").write_text(output.model_dump_json())

    run(cc_dir, senses_db, queue_dir)

    # Output file should be deleted
    assert not (cc_dir / "done" / "induction" / "test1.json").exists()
    # Clerk request should be enqueued
    pending_files = list((queue_dir / "pending").glob("*.json"))
    assert len(pending_files) == 1
    req = json.loads(pending_files[0].read_text())
    assert req["type"] == "add_senses"
    assert req["form"] == "cat"


def test_apply_induction_adds_to_blocklist(tmp_path: Path):
    cc_dir, senses_db, queue_dir = _setup(tmp_path)
    blocklist_file = tmp_path / "blocklist.yaml"

    output = CCInductionOutput(
        id="test-bl",
        form="thrumbly",
        add_to_blocklist=True,
        blocklist_reason="tokenization artifact",
    )
    (cc_dir / "done" / "induction" / "test-bl.json").write_text(
        output.model_dump_json()
    )

    run(cc_dir, senses_db, queue_dir, blocklist_file=str(blocklist_file))

    # Output file deleted
    assert not (cc_dir / "done" / "induction" / "test-bl.json").exists()
    # Blocklist file updated
    assert blocklist_file.exists()
    content = blocklist_file.read_text()
    assert "thrumbly" in content
    # No clerk request
    assert not list((queue_dir / "pending").glob("*.json"))


def test_apply_induction_skip_labels(tmp_path: Path):
    cc_dir, senses_db, queue_dir = _setup(tmp_path)
    labeled_db = tmp_path / "labeled.db"

    store = SenseStore(senses_db)
    store.update("cat", lambda _: Alf(form="cat", senses=[]))

    # Write the task file so apply can get occurrence_refs
    task = CCInductionTask(
        id="test-labels",
        form="cat",
        contexts=["The cat sat.", "The cat meowed."],
        existing_defs=[],
        occurrence_refs=[
            Occurrence(doc_id="doc1", byte_offset=100),
            Occurrence(doc_id="doc2", byte_offset=200),
        ],
    )
    (cc_dir / "pending" / "induction" / "test-labels.json").write_text(
        task.model_dump_json()
    )

    output = CCInductionOutput(
        id="test-labels",
        form="cat",
        new_senses=[
            InductionSense(definition="a small domesticated feline", pos="noun")
        ],
        context_labels=[
            ContextLabel(context_idx=0, sense_idx=1),  # sense assignment
            ContextLabel(context_idx=1, sense_idx=None),  # skip
        ],
    )
    (cc_dir / "done" / "induction" / "test-labels.json").write_text(
        output.model_dump_json()
    )

    run(cc_dir, senses_db, queue_dir, labeled_db=str(labeled_db))

    # Task file should be deleted after processing
    assert not (cc_dir / "pending" / "induction" / "test-labels.json").exists()
    # Check labeled occurrences written
    occ_store = OccurrenceStore(labeled_db)
    df = occ_store.query_form("cat")
    assert len(df) == 2
    rows = {(r["doc_id"], r["byte_offset"]): r for r in df.iter_rows(named=True)}
    assert rows[("doc1", 100)]["sense_key"] == "1"
    assert rows[("doc1", 100)]["rating"] == 2
    assert rows[("doc2", 200)]["sense_key"] == "_skip"
    assert rows[("doc2", 200)]["rating"] == 0


def test_apply_induction_no_labeled_db_skip_ignored(tmp_path: Path):
    """If labeled_db not passed, skip labels are silently omitted (no crash)."""
    cc_dir, senses_db, queue_dir = _setup(tmp_path)

    store = SenseStore(senses_db)
    store.update("cat", lambda _: Alf(form="cat", senses=[]))

    output = CCInductionOutput(
        id="test-nolabeled",
        form="cat",
        new_senses=[InductionSense(definition="a small feline", pos="noun")],
        context_labels=[ContextLabel(context_idx=0, sense_idx=None)],
    )
    (cc_dir / "done" / "induction" / "test-nolabeled.json").write_text(
        output.model_dump_json()
    )

    # Should not raise even though there's no labeled_db
    run(cc_dir, senses_db, queue_dir)

    assert not (cc_dir / "done" / "induction" / "test-nolabeled.json").exists()


def test_apply_empty_done_dir(tmp_path: Path):
    cc_dir, senses_db, queue_dir = _setup(tmp_path)
    # Should not error with empty done/
    run(cc_dir, senses_db, queue_dir)
