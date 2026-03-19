"""Integration tests for the full labeling pipeline:
LLM stub → label_occurrences.run() → labeled.db.
"""

from pathlib import Path

import polars as pl

from alfs.data_models.alf import Alf, Sense
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore
from alfs.data_models.update_target import UpdateTarget
from alfs.update.labeling import label_occurrences
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


def test_labeling_pipeline_writes_to_labeled_db(tmp_path: Path, monkeypatch) -> None:
    form = "bark"
    doc_id = "doc1"
    text = "the dog began to bark loudly at night"
    byte_offset = text.index(form)

    sense = Sense(definition="the sound a dog makes")
    senses_db = tmp_path / "senses.db"
    sense_store = SenseStore(senses_db)
    sense_store.write(Alf(form=form, senses=[sense]))

    labeled_db = tmp_path / "labeled.db"
    target_path = tmp_path / "target.json"
    target_path.write_text(UpdateTarget(form=form).model_dump_json())
    _write_occurrences(tmp_path / "seg", form, doc_id, byte_offset)
    _write_docs(tmp_path / "docs.parquet", doc_id, text)

    fake = FakeLLM([{"sense_key": "1", "rating": 2}])
    monkeypatch.setattr(
        "alfs.update.labeling.label_occurrences.llm.chat_json", fake.chat_json
    )

    label_occurrences.run(
        target_file=target_path,
        seg_data_dir=tmp_path / "seg",
        docs=tmp_path / "docs.parquet",
        senses_db=senses_db,
        labeled_db=labeled_db,
    )

    occ_store = OccurrenceStore(labeled_db)
    df = occ_store.query_form(form)
    assert len(df) == 1
    assert df["sense_key"][0] == sense.id
    assert df["rating"][0] == 2


def test_labeling_pipeline_skips_already_labeled(tmp_path: Path, monkeypatch) -> None:
    form = "bark"
    doc_id = "doc1"
    text = "the dog began to bark loudly at night"
    byte_offset = text.index(form)

    sense = Sense(definition="the sound a dog makes")
    senses_db = tmp_path / "senses.db"
    sense_store = SenseStore(senses_db)
    sense_store.write(Alf(form=form, senses=[sense]))

    labeled_db = tmp_path / "labeled.db"
    occ_store = OccurrenceStore(labeled_db)
    occ_store.upsert_many([(form, doc_id, byte_offset, sense.id, 2)])

    target_path = tmp_path / "target.json"
    target_path.write_text(UpdateTarget(form=form).model_dump_json())
    _write_occurrences(tmp_path / "seg", form, doc_id, byte_offset)
    _write_docs(tmp_path / "docs.parquet", doc_id, text)

    fake = FakeLLM([])
    monkeypatch.setattr(
        "alfs.update.labeling.label_occurrences.llm.chat_json", fake.chat_json
    )

    label_occurrences.run(
        target_file=target_path,
        seg_data_dir=tmp_path / "seg",
        docs=tmp_path / "docs.parquet",
        senses_db=senses_db,
        labeled_db=labeled_db,
    )

    assert len(fake.calls) == 0


def test_labeling_pipeline_skips_form_with_no_senses(
    tmp_path: Path, monkeypatch
) -> None:
    form = "bark"
    doc_id = "doc1"
    text = "the dog began to bark loudly at night"
    byte_offset = text.index(form)

    senses_db = tmp_path / "senses.db"
    sense_store = SenseStore(senses_db)
    sense_store.write(Alf(form=form, senses=[]))

    labeled_db = tmp_path / "labeled.db"
    target_path = tmp_path / "target.json"
    target_path.write_text(UpdateTarget(form=form).model_dump_json())
    _write_occurrences(tmp_path / "seg", form, doc_id, byte_offset)
    _write_docs(tmp_path / "docs.parquet", doc_id, text)

    fake = FakeLLM([])
    monkeypatch.setattr(
        "alfs.update.labeling.label_occurrences.llm.chat_json", fake.chat_json
    )

    label_occurrences.run(
        target_file=target_path,
        seg_data_dir=tmp_path / "seg",
        docs=tmp_path / "docs.parquet",
        senses_db=senses_db,
        labeled_db=labeled_db,
    )

    assert len(fake.calls) == 0
    occ_store = OccurrenceStore(labeled_db)
    assert len(occ_store.query_form(form)) == 0
