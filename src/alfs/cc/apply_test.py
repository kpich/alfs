"""Tests for CC apply module."""

import json
from pathlib import Path

from alfs.cc.apply import run
from alfs.cc.models import (
    CCInductionOutput,
    CCRewriteOutput,
    CCTrimSenseOutput,
    InductionSense,
    RewrittenSense,
)
from alfs.data_models.alf import Alf, Sense
from alfs.data_models.pos import PartOfSpeech
from alfs.data_models.sense_store import SenseStore


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    cc_dir = tmp_path / "cc_tasks"
    (cc_dir / "pending").mkdir(parents=True)
    (cc_dir / "done").mkdir(parents=True)
    senses_db = tmp_path / "senses.db"
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    return cc_dir, senses_db, queue_dir


def test_apply_induction(tmp_path: Path):
    cc_dir, senses_db, queue_dir = _setup(tmp_path)

    store = SenseStore(senses_db)
    store.update("cat", lambda _: Alf(form="cat", senses=[]))

    output = CCInductionOutput(
        id="test1",
        form="cat",
        senses=[InductionSense(definition="a small domesticated feline", pos="noun")],
    )
    (cc_dir / "done" / "test1.json").write_text(output.model_dump_json())

    run(cc_dir, senses_db, queue_dir)

    # Output file should be deleted
    assert not (cc_dir / "done" / "test1.json").exists()
    # Clerk request should be enqueued
    pending_files = list((queue_dir / "pending").glob("*.json"))
    assert len(pending_files) == 1
    req = json.loads(pending_files[0].read_text())
    assert req["type"] == "add_senses"
    assert req["form"] == "cat"


def test_apply_rewrite(tmp_path: Path):
    cc_dir, senses_db, queue_dir = _setup(tmp_path)

    store = SenseStore(senses_db)
    store.update(
        "run",
        lambda _: Alf(
            form="run",
            senses=[Sense(id="s1", definition="to move fast", pos=PartOfSpeech.verb)],
        ),
    )

    output = CCRewriteOutput(
        id="test2",
        form="run",
        senses=[RewrittenSense(definition="to move swiftly on foot")],
    )
    (cc_dir / "done" / "test2.json").write_text(output.model_dump_json())

    run(cc_dir, senses_db, queue_dir)

    assert not (cc_dir / "done" / "test2.json").exists()
    pending_files = list((queue_dir / "pending").glob("*.json"))
    assert len(pending_files) == 1
    req = json.loads(pending_files[0].read_text())
    assert req["type"] == "rewrite"
    assert req["after"][0]["definition"] == "to move swiftly on foot"


def test_apply_rewrite_sense_count_mismatch(tmp_path: Path):
    cc_dir, senses_db, queue_dir = _setup(tmp_path)

    store = SenseStore(senses_db)
    store.update(
        "run",
        lambda _: Alf(
            form="run",
            senses=[
                Sense(id="s1", definition="to move fast", pos=PartOfSpeech.verb),
                Sense(id="s2", definition="a jog", pos=PartOfSpeech.noun),
            ],
        ),
    )

    # Output has wrong number of senses
    output = CCRewriteOutput(
        id="test3",
        form="run",
        senses=[RewrittenSense(definition="to move swiftly on foot")],
    )
    (cc_dir / "done" / "test3.json").write_text(output.model_dump_json())

    run(cc_dir, senses_db, queue_dir)

    # File should NOT be deleted (error case)
    assert (cc_dir / "done" / "test3.json").exists()
    # No clerk request
    assert not list((queue_dir / "pending").glob("*.json"))


def test_apply_trim_sense(tmp_path: Path):
    cc_dir, senses_db, queue_dir = _setup(tmp_path)

    store = SenseStore(senses_db)
    store.update(
        "bank",
        lambda _: Alf(
            form="bank",
            senses=[
                Sense(id="s1", definition="financial institution"),
                Sense(id="s2", definition="side of a river"),
            ],
        ),
    )

    output = CCTrimSenseOutput(id="test4", form="bank", sense_num=2, reason="redundant")
    (cc_dir / "done" / "test4.json").write_text(output.model_dump_json())

    run(cc_dir, senses_db, queue_dir)

    assert not (cc_dir / "done" / "test4.json").exists()
    pending_files = list((queue_dir / "pending").glob("*.json"))
    assert len(pending_files) == 1
    req = json.loads(pending_files[0].read_text())
    assert req["type"] == "trim_sense"
    assert req["sense_id"] == "s2"


def test_apply_trim_sense_null(tmp_path: Path):
    cc_dir, senses_db, queue_dir = _setup(tmp_path)

    output = CCTrimSenseOutput(
        id="test5", form="bank", sense_num=None, reason="all distinct"
    )
    (cc_dir / "done" / "test5.json").write_text(output.model_dump_json())

    run(cc_dir, senses_db, queue_dir)

    # File deleted (no-op but ok)
    assert not (cc_dir / "done" / "test5.json").exists()
    # No clerk request
    assert not list((queue_dir / "pending").glob("*.json"))


def test_apply_empty_done_dir(tmp_path: Path):
    cc_dir, senses_db, queue_dir = _setup(tmp_path)
    # Should not error with empty done/
    run(cc_dir, senses_db, queue_dir)
