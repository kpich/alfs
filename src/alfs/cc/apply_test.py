"""Tests for CC apply module."""

import json
from pathlib import Path

from alfs.cc.apply import run
from alfs.cc.models import (
    CCInductionOutput,
    CCMorphRelBlockOutput,
    CCQCOutput,
    ContextLabel,
    DeletedSenseEntry,
    InductionMorphRel,
    InductionSense,
    MorphRelEntry,
    PosCorrection,
    SenseRewrite,
)
from alfs.data_models.alf import Alf, Sense
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
        occurrence_refs=[
            Occurrence(doc_id="doc1", byte_offset=100),
            Occurrence(doc_id="doc2", byte_offset=200),
        ],
    )
    (cc_dir / "done" / "induction" / "test-labels.json").write_text(
        output.model_dump_json()
    )

    run(cc_dir, senses_db, queue_dir, labeled_db=str(labeled_db))

    # Check labeled occurrences written
    occ_store = OccurrenceStore(labeled_db)
    df = occ_store.query_form("cat")
    assert len(df) == 2
    rows = {(r["doc_id"], r["byte_offset"]): r for r in df.iter_rows(named=True)}
    sense_key = rows[("doc1", 100)]["sense_key"]
    assert len(sense_key) == 36, f"Expected UUID sense_key, got {sense_key!r}"
    assert rows[("doc1", 100)]["rating"] == 2
    assert rows[("doc2", 200)]["sense_key"] == "_skip"
    assert rows[("doc2", 200)]["rating"] == 0


def test_apply_induction_morph_rel_label_stores_base_uuid(tmp_path: Path):
    cc_dir, senses_db, queue_dir = _setup(tmp_path)
    labeled_db = tmp_path / "labeled.db"

    store = SenseStore(senses_db)
    store.update("dogs", lambda _: Alf(form="dogs", senses=[]))

    output = CCInductionOutput(
        id="test-morph-label",
        form="dogs",
        new_senses=[
            InductionSense(
                definition="a domesticated carnivorous mammal",
                pos="noun",
                morph_rel=InductionMorphRel(base_form="dog", relation="plural"),
            )
        ],
        context_labels=[ContextLabel(context_idx=0, sense_idx=1)],
        occurrence_refs=[Occurrence(doc_id="doc1", byte_offset=0)],
    )
    (cc_dir / "done" / "induction" / "test-morph-label.json").write_text(
        output.model_dump_json()
    )

    run(cc_dir, senses_db, queue_dir, labeled_db=str(labeled_db))

    occ_store = OccurrenceStore(labeled_db)
    df = occ_store.query_form("dogs")
    assert len(df) == 1
    sense_key = df.to_dicts()[0]["sense_key"]

    # Must be a UUID, not a numeric index
    assert len(sense_key) == 36, f"Expected UUID sense_key, got {sense_key!r}"

    # UUID must match the new base sense enqueued for "dog"
    pending = list((queue_dir / "pending").glob("*.json"))
    dog_reqs = [
        json.loads(f.read_text())
        for f in pending
        if json.loads(f.read_text()).get("form") == "dog"
    ]
    assert len(dog_reqs) == 1
    assert sense_key == dog_reqs[0]["new_senses"][0]["id"]


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


def test_normalize_case_does_not_add_to_blocklist(tmp_path: Path):
    cc_dir, senses_db, queue_dir = _setup(tmp_path)
    blocklist_file = tmp_path / "blocklist.yaml"
    (cc_dir / "done" / "morphrel_block").mkdir(parents=True, exist_ok=True)

    SenseStore(senses_db).update("abreha", lambda _: Alf(form="abreha", senses=[]))

    output = CCMorphRelBlockOutput(
        id="test-nc", form="abreha", action="normalize_case", canonical_form="Abreha"
    )
    (cc_dir / "done" / "morphrel_block" / "test-nc.json").write_text(
        output.model_dump_json()
    )

    run(cc_dir, senses_db, queue_dir, blocklist_file=str(blocklist_file))

    assert not blocklist_file.exists()


def test_apply_empty_done_dir(tmp_path: Path):
    cc_dir, senses_db, queue_dir = _setup(tmp_path)
    # Should not error with empty done/
    run(cc_dir, senses_db, queue_dir)


def _setup_qc(tmp_path: Path) -> tuple[Path, Path, Path]:
    cc_dir = tmp_path / "cc_tasks"
    (cc_dir / "pending" / "qc").mkdir(parents=True)
    (cc_dir / "done" / "qc").mkdir(parents=True)
    senses_db = tmp_path / "senses.db"
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    return cc_dir, senses_db, queue_dir


def test_apply_qc_sense_rewrite(tmp_path: Path):
    cc_dir, senses_db, queue_dir = _setup_qc(tmp_path)

    store = SenseStore(senses_db)
    store.update(
        "cat",
        lambda _: Alf(
            form="cat",
            senses=[
                Sense(id="s0", definition="a domestic feline"),
                Sense(id="s1", definition="cool person lol"),
            ],
        ),
    )

    output = CCQCOutput(
        id="test-rw",
        form="cat",
        sense_rewrites=[
            SenseRewrite(sense_idx=1, definition="a cool or hip person (slang)")
        ],
    )
    (cc_dir / "done" / "qc" / "test-rw.json").write_text(output.model_dump_json())

    run(cc_dir, senses_db, queue_dir)

    assert not (cc_dir / "done" / "qc" / "test-rw.json").exists()
    pending = list((queue_dir / "pending").glob("*.json"))
    assert len(pending) == 1
    req = json.loads(pending[0].read_text())
    assert req["type"] == "rewrite"
    assert req["form"] == "cat"
    assert req["after"]["definition"] == "a cool or hip person (slang)"


def test_apply_qc_pos_correction(tmp_path: Path):
    cc_dir, senses_db, queue_dir = _setup_qc(tmp_path)

    store = SenseStore(senses_db)
    store.update(
        "run",
        lambda _: Alf(
            form="run",
            senses=[Sense(id="s0", definition="to move on foot at speed")],
        ),
    )

    output = CCQCOutput(
        id="test-pos",
        form="run",
        pos_corrections=[PosCorrection(sense_idx=0, pos="verb")],
    )
    (cc_dir / "done" / "qc" / "test-pos.json").write_text(output.model_dump_json())

    run(cc_dir, senses_db, queue_dir)

    assert not (cc_dir / "done" / "qc" / "test-pos.json").exists()
    pending = list((queue_dir / "pending").glob("*.json"))
    assert len(pending) == 1
    req = json.loads(pending[0].read_text())
    assert req["type"] == "update_pos"
    assert req["form"] == "run"
    assert req["after"]["pos"] == "verb"


def test_apply_qc_deleted_senses(tmp_path: Path):
    cc_dir, senses_db, queue_dir = _setup_qc(tmp_path)

    store = SenseStore(senses_db)
    store.update(
        "dog",
        lambda _: Alf(
            form="dog",
            senses=[
                Sense(id="s0", definition="a domestic canine"),
                Sense(id="s1", definition="a canine animal"),  # redundant
                Sense(id="s2", definition="to follow persistently"),
            ],
        ),
    )

    output = CCQCOutput(
        id="test-del",
        form="dog",
        deleted_senses=[
            DeletedSenseEntry(sense_idx=1, reason="redundant with sense 0")
        ],
    )
    (cc_dir / "done" / "qc" / "test-del.json").write_text(output.model_dump_json())

    run(cc_dir, senses_db, queue_dir)

    assert not (cc_dir / "done" / "qc" / "test-del.json").exists()
    pending = list((queue_dir / "pending").glob("*.json"))
    assert len(pending) == 1
    req = json.loads(pending[0].read_text())
    assert req["type"] == "prune"
    assert req["form"] == "dog"
    assert req["removed_ids"] == ["s1"]
    assert len(req["after"]) == 2


def test_apply_qc_multiple_deleted_senses(tmp_path: Path):
    cc_dir, senses_db, queue_dir = _setup_qc(tmp_path)

    store = SenseStore(senses_db)
    store.update(
        "dog",
        lambda _: Alf(
            form="dog",
            senses=[
                Sense(id="s0", definition="a domestic canine"),
                Sense(id="s1", definition="a canine animal"),
                Sense(id="s2", definition="canine"),
            ],
        ),
    )

    output = CCQCOutput(
        id="test-multidel",
        form="dog",
        deleted_senses=[
            DeletedSenseEntry(sense_idx=1, reason="redundant"),
            DeletedSenseEntry(sense_idx=2, reason="too terse"),
        ],
    )
    (cc_dir / "done" / "qc" / "test-multidel.json").write_text(output.model_dump_json())

    run(cc_dir, senses_db, queue_dir)

    pending = list((queue_dir / "pending").glob("*.json"))
    # Multiple deletions produce a single prune request
    assert len(pending) == 1
    req = json.loads(pending[0].read_text())
    assert req["type"] == "prune"
    assert set(req["removed_ids"]) == {"s1", "s2"}
    assert len(req["after"]) == 1


def test_apply_qc_delete_entry(tmp_path: Path):
    cc_dir, senses_db, queue_dir = _setup_qc(tmp_path)
    blocklist_file = tmp_path / "blocklist.yaml"

    output = CCQCOutput(
        id="test-del-entry",
        form="5*!",
        delete_entry=True,
        delete_entry_reason="tokenization artifact",
    )
    (cc_dir / "done" / "qc" / "test-del-entry.json").write_text(
        output.model_dump_json()
    )

    run(cc_dir, senses_db, queue_dir, blocklist_file=str(blocklist_file))

    assert not (cc_dir / "done" / "qc" / "test-del-entry.json").exists()
    pending = list((queue_dir / "pending").glob("*.json"))
    assert len(pending) == 1
    req = json.loads(pending[0].read_text())
    assert req["type"] == "delete_entry"
    assert req["form"] == "5*!"
    assert blocklist_file.exists()
    assert "5*!" in blocklist_file.read_text()


def test_apply_qc_normalize_case(tmp_path: Path):
    cc_dir, senses_db, queue_dir = _setup_qc(tmp_path)

    store = SenseStore(senses_db)
    store.update(
        "aaron",
        lambda _: Alf(form="aaron", senses=[Sense(id="s0", definition="a given name")]),
    )

    output = CCQCOutput(id="test-nc", form="aaron", normalize_case="Aaron")
    (cc_dir / "done" / "qc" / "test-nc.json").write_text(output.model_dump_json())

    run(cc_dir, senses_db, queue_dir)

    assert not (cc_dir / "done" / "qc" / "test-nc.json").exists()
    pending = list((queue_dir / "pending").glob("*.json"))
    types = {json.loads(f.read_text())["type"] for f in pending}
    assert "add_senses" in types
    assert "delete_entry" in types


def test_apply_qc_spelling_variant_of(tmp_path: Path):
    cc_dir, senses_db, queue_dir = _setup_qc(tmp_path)

    store = SenseStore(senses_db)
    store.update(
        "colour",
        lambda _: Alf(
            form="colour", senses=[Sense(id="s0", definition="a visual property")]
        ),
    )

    output = CCQCOutput(id="test-sv", form="colour", spelling_variant_of="color")
    (cc_dir / "done" / "qc" / "test-sv.json").write_text(output.model_dump_json())

    run(cc_dir, senses_db, queue_dir)

    assert not (cc_dir / "done" / "qc" / "test-sv.json").exists()
    pending = list((queue_dir / "pending").glob("*.json"))
    assert len(pending) == 1
    req = json.loads(pending[0].read_text())
    assert req["type"] == "set_spelling_variant"
    assert req["form"] == "colour"
    assert req["preferred_form"] == "color"


def test_apply_qc_morph_rels(tmp_path: Path):
    cc_dir, senses_db, queue_dir = _setup_qc(tmp_path)

    store = SenseStore(senses_db)
    store.update(
        "dog", lambda _: Alf(form="dog", senses=[Sense(id="s0", definition="a canine")])
    )
    store.update(
        "dogs",
        lambda _: Alf(
            form="dogs", senses=[Sense(id="s1", definition="multiple canines")]
        ),
    )

    output = CCQCOutput(
        id="test-mr",
        form="dogs",
        morph_rels=[
            MorphRelEntry(
                sense_idx=0,
                morph_base="dog",
                morph_relation="plural",
                proposed_definition="plural of dog (n.)",
                promote_to_parent=False,
            )
        ],
    )
    (cc_dir / "done" / "qc" / "test-mr.json").write_text(output.model_dump_json())

    run(cc_dir, senses_db, queue_dir)

    assert not (cc_dir / "done" / "qc" / "test-mr.json").exists()
    pending = list((queue_dir / "pending").glob("*.json"))
    assert len(pending) == 1
    req = json.loads(pending[0].read_text())
    assert req["type"] == "morph_redirect"
    assert req["form"] == "dogs"


def test_apply_qc_combined_rewrite_and_delete_senses(tmp_path: Path):
    cc_dir, senses_db, queue_dir = _setup_qc(tmp_path)

    store = SenseStore(senses_db)
    store.update(
        "bat",
        lambda _: Alf(
            form="bat",
            senses=[
                Sense(id="s0", definition="a stick used for hitting"),
                Sense(id="s1", definition="flying mammal lol"),
                Sense(id="s2", definition="cricket implement"),  # redundant with s0
            ],
        ),
    )

    output = CCQCOutput(
        id="test-combo",
        form="bat",
        sense_rewrites=[
            SenseRewrite(
                sense_idx=1,
                definition="a nocturnal flying mammal of the order Chiroptera",
            )
        ],
        deleted_senses=[
            DeletedSenseEntry(sense_idx=2, reason="redundant with sense 0")
        ],
    )
    (cc_dir / "done" / "qc" / "test-combo.json").write_text(output.model_dump_json())

    run(cc_dir, senses_db, queue_dir)

    assert not (cc_dir / "done" / "qc" / "test-combo.json").exists()
    pending = list((queue_dir / "pending").glob("*.json"))
    types = [json.loads(f.read_text())["type"] for f in pending]
    assert "rewrite" in types
    assert "prune" in types
    assert len(pending) == 2
