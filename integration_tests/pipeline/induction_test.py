"""Integration tests for the full induction pipeline:
LLM stub → induce_senses.run() → update_inventory.run() → clerk drain → senses.db.
"""

from pathlib import Path

import polars as pl

from alfs.clerk.queue import drain
from alfs.data_models.alf import Alf
from alfs.data_models.sense_store import SenseStore
from alfs.data_models.update_target import UpdateTarget
from alfs.update.induction import induce_senses, update_inventory
from integration_tests.fake_llm import FakeLLM


def _write_occurrences(
    seg_data_dir: Path, form: str, doc_id: str, byte_offset: int
) -> None:
    prefix = form[0] if form[0].isascii() and form[0].isalpha() else "other"
    prefix_dir = seg_data_dir / prefix
    prefix_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {"form": [form], "doc_id": [doc_id], "byte_offset": [byte_offset]}
    )
    df.write_parquet(prefix_dir / "occurrences.parquet")


def _write_docs(docs_path: Path, doc_id: str, text: str) -> None:
    df = pl.DataFrame({"doc_id": [doc_id], "text": [text]})
    df.write_parquet(docs_path)


def test_induction_pipeline_adds_sense_to_senses_db(
    tmp_path: Path, monkeypatch
) -> None:
    form = "drip"
    doc_id = "doc1"
    text = "water began to drip from the ceiling slowly"
    byte_offset = text.index(form)

    target_path = tmp_path / "target.json"
    target_path.write_text(UpdateTarget(form=form).model_dump_json())
    _write_occurrences(tmp_path / "seg", form, doc_id, byte_offset)
    _write_docs(tmp_path / "docs.parquet", doc_id, text)

    output_path = tmp_path / "output.json"
    queue_dir = tmp_path / "queue"
    senses_db = tmp_path / "senses.db"

    fake = FakeLLM(
        [
            {
                "all_covered": False,
                "senses": [
                    {"definition": "to fall in drops", "examples": [1], "pos": "verb"}
                ],
            },
            {"is_valid": True, "reason": ""},
        ]
    )
    monkeypatch.setattr(
        "alfs.update.induction.induce_senses.llm.chat_json", fake.chat_json
    )

    induce_senses.run(
        target_file=target_path,
        seg_data_dir=tmp_path / "seg",
        docs=tmp_path / "docs.parquet",
        output=output_path,
    )
    update_inventory.run(
        senses_file=output_path, senses_db=senses_db, queue_dir=queue_dir
    )

    store = SenseStore(senses_db)
    drain(queue_dir, store, None)

    entry = store.read(form)
    assert entry is not None
    assert len(entry.senses) == 1
    assert entry.senses[0].definition == "to fall in drops"
    assert len(list((queue_dir / "done").glob("*.json"))) == 1


def test_induction_pipeline_critic_rejection_skips_sense(
    tmp_path: Path, monkeypatch
) -> None:
    form = "drip"
    doc_id = "doc1"
    text = "water began to drip from the ceiling slowly"
    byte_offset = text.index(form)

    target_path = tmp_path / "target.json"
    target_path.write_text(UpdateTarget(form=form).model_dump_json())
    _write_occurrences(tmp_path / "seg", form, doc_id, byte_offset)
    _write_docs(tmp_path / "docs.parquet", doc_id, text)

    output_path = tmp_path / "output.json"
    queue_dir = tmp_path / "queue"
    senses_db = tmp_path / "senses.db"

    fake = FakeLLM(
        [
            {
                "all_covered": False,
                "senses": [
                    {"definition": "to fall in drops", "examples": [1], "pos": "verb"}
                ],
            },
            {"is_valid": False, "reason": "too vague"},
        ]
    )
    monkeypatch.setattr(
        "alfs.update.induction.induce_senses.llm.chat_json", fake.chat_json
    )

    induce_senses.run(
        target_file=target_path,
        seg_data_dir=tmp_path / "seg",
        docs=tmp_path / "docs.parquet",
        output=output_path,
    )
    update_inventory.run(
        senses_file=output_path, senses_db=senses_db, queue_dir=queue_dir
    )

    store = SenseStore(senses_db)
    drain(queue_dir, store, None)

    entry = store.read(form)
    assert entry is None
    assert len(list((queue_dir / "pending").glob("*.json"))) == 0


def test_induction_pipeline_all_covered_skips_critic(
    tmp_path: Path, monkeypatch
) -> None:
    form = "drip"
    doc_id = "doc1"
    text = "water began to drip from the ceiling slowly"
    byte_offset = text.index(form)

    target_path = tmp_path / "target.json"
    target_path.write_text(UpdateTarget(form=form).model_dump_json())
    _write_occurrences(tmp_path / "seg", form, doc_id, byte_offset)
    _write_docs(tmp_path / "docs.parquet", doc_id, text)

    output_path = tmp_path / "output.json"
    senses_db = tmp_path / "senses.db"

    # Pre-populate senses.db so all_covered=True is meaningful
    store = SenseStore(senses_db)
    store.write(Alf(form=form, senses=[]))

    fake = FakeLLM(
        [
            {"all_covered": True, "senses": []},
        ]
    )
    monkeypatch.setattr(
        "alfs.update.induction.induce_senses.llm.chat_json", fake.chat_json
    )

    induce_senses.run(
        target_file=target_path,
        seg_data_dir=tmp_path / "seg",
        docs=tmp_path / "docs.parquet",
        output=output_path,
        senses_db=senses_db,
    )

    assert len(fake.calls) == 1
    entry = SenseStore(senses_db).read(form)
    assert entry is not None
    assert len(entry.senses) == 0
