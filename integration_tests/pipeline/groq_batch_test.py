"""Integration tests: groq_batch_prepare → fake Groq output → groq_batch_ingest."""

import json
from pathlib import Path

import polars as pl

from alfs.data_models.alf import Alf, Sense
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore
from alfs.update.labeling import groq_batch_ingest, groq_batch_prepare


def _write_occurrences(
    seg_data_dir: Path, form: str, doc_id: str, byte_offset: int
) -> None:
    prefix = form[0].lower() if form[0].isascii() and form[0].isalpha() else "other"
    prefix_dir = seg_data_dir / prefix
    prefix_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"form": [form], "doc_id": [doc_id], "byte_offset": [byte_offset]}
    ).write_parquet(prefix_dir / "occurrences.parquet")


def _write_docs(docs_path: Path, doc_id: str, text: str) -> None:
    pl.DataFrame({"doc_id": [doc_id], "text": [text]}).write_parquet(docs_path)


def test_prepare_and_ingest_roundtrip(tmp_path: Path) -> None:
    """prepare → fake Groq output → ingest writes correct UUID sense key to
    labeled.db."""
    form = "bark"
    doc_id = "doc1"
    text = "the dog began to bark loudly at night"
    byte_offset = text.index(form)

    sense = Sense(definition="the sound a dog makes")
    senses_db = tmp_path / "senses.db"
    labeled_db = tmp_path / "labeled.db"

    store = SenseStore(senses_db)
    store.write(Alf(form=form, senses=[sense]))
    OccurrenceStore(labeled_db)  # initialize empty DB

    _write_occurrences(tmp_path / "seg", form, doc_id, byte_offset)
    _write_docs(tmp_path / "docs.parquet", doc_id, text)

    batch_path, metadata_path = groq_batch_prepare.run(
        senses_db=senses_db,
        labeled_db=labeled_db,
        seg_data_dir=tmp_path / "seg",
        docs=tmp_path / "docs.parquet",
        output_dir=tmp_path / "batch",
        n=1,
        min_count=1,
    )

    # Verify batch structure
    requests = [
        json.loads(line) for line in batch_path.read_text().splitlines() if line
    ]
    assert len(requests) == 1
    req = requests[0]
    assert req["method"] == "POST"
    assert req["url"] == "/v1/chat/completions"
    sys_msg = req["body"]["messages"][0]
    assert sys_msg["role"] == "system"
    assert form in sys_msg["content"]
    assert "sense_key" in sys_msg["content"]
    user_msg = req["body"]["messages"][1]
    assert user_msg["role"] == "user"
    assert form in user_msg["content"]
    assert req["body"]["response_format"] == {"type": "json_object"}

    # Verify metadata
    meta = [json.loads(line) for line in metadata_path.read_text().splitlines() if line]
    assert len(meta) == 1
    assert meta[0]["form"] == form
    assert meta[0]["doc_id"] == doc_id
    assert meta[0]["byte_offset"] == byte_offset

    # Synthesize fake Groq output (same structure as real Groq batch output)
    custom_id = meta[0]["custom_id"]
    groq_output = tmp_path / "batch" / "batch_output.jsonl"
    groq_output.write_text(
        json.dumps(
            {
                "custom_id": custom_id,
                "response": {
                    "body": {
                        "choices": [
                            {"message": {"content": '{"sense_key": "1", "rating": 2}'}}
                        ]
                    }
                },
            }
        )
        + "\n"
    )

    n_ingested = groq_batch_ingest.ingest(
        batch_output=groq_output,
        metadata_path=metadata_path,
        senses_db=senses_db,
        labeled_db=labeled_db,
    )
    assert n_ingested == 1

    occ_store = OccurrenceStore(labeled_db)
    df = occ_store.query_form(form)
    assert len(df) == 1
    row = df.row(0, named=True)
    assert row["form"] == form
    assert row["doc_id"] == doc_id
    assert row["byte_offset"] == byte_offset
    assert row["sense_key"] == sense.id  # display key "1" mapped to UUID
    assert row["rating"] == 2


def test_prepare_excludes_redirect_forms(tmp_path: Path) -> None:
    """Redirect forms are not included in the batch."""
    form = "Bark"  # redirect to "bark"
    canonical = "bark"
    doc_id = "d1"
    text = "the Bark is loud"
    byte_offset = text.index(form)

    sense = Sense(definition="the sound a dog makes")
    senses_db = tmp_path / "senses.db"
    labeled_db = tmp_path / "labeled.db"

    store = SenseStore(senses_db)
    store.write(Alf(form=canonical, senses=[sense]))
    store.write(Alf(form=form, redirect=canonical))
    OccurrenceStore(labeled_db)

    _write_occurrences(tmp_path / "seg", form, doc_id, byte_offset)
    _write_docs(tmp_path / "docs.parquet", doc_id, text)

    batch_path, _ = groq_batch_prepare.run(
        senses_db=senses_db,
        labeled_db=labeled_db,
        seg_data_dir=tmp_path / "seg",
        docs=tmp_path / "docs.parquet",
        output_dir=tmp_path / "batch",
        n=100,
        min_count=1,
    )
    lines = [ln for ln in batch_path.read_text().splitlines() if ln]
    assert len(lines) == 0


def test_prepare_excludes_form_with_no_senses(tmp_path: Path) -> None:
    """Forms with no effective senses produce an empty batch."""
    form = "xyz"
    doc_id = "d1"
    text = "xyz is here"

    senses_db = tmp_path / "senses.db"
    labeled_db = tmp_path / "labeled.db"
    store = SenseStore(senses_db)
    store.write(Alf(form=form, senses=[]))
    OccurrenceStore(labeled_db)

    _write_occurrences(tmp_path / "seg", form, doc_id, 0)
    _write_docs(tmp_path / "docs.parquet", doc_id, text)

    batch_path, _ = groq_batch_prepare.run(
        senses_db=senses_db,
        labeled_db=labeled_db,
        seg_data_dir=tmp_path / "seg",
        docs=tmp_path / "docs.parquet",
        output_dir=tmp_path / "batch",
        n=100,
        min_count=1,
    )
    lines = [ln for ln in batch_path.read_text().splitlines() if ln]
    assert len(lines) == 0


def test_ingest_skips_malformed_and_unknown(tmp_path: Path) -> None:
    """Ingest skips malformed JSON lines and unknown custom_ids without crashing."""
    form = "cat"
    sense = Sense(definition="a small furry animal")
    senses_db = tmp_path / "senses.db"
    labeled_db = tmp_path / "labeled.db"
    SenseStore(senses_db).write(Alf(form=form, senses=[sense]))
    OccurrenceStore(labeled_db)

    metadata_path = tmp_path / "meta.jsonl"
    metadata_path.write_text(
        json.dumps({"custom_id": "0", "form": form, "doc_id": "d1", "byte_offset": 0})
        + "\n"
    )

    batch_output = tmp_path / "output.jsonl"
    batch_output.write_text(
        "not json at all\n"
        + json.dumps(
            {
                "custom_id": "999",  # unknown id
                "response": {
                    "body": {
                        "choices": [
                            {"message": {"content": '{"sense_key": "1", "rating": 2}'}}
                        ]
                    }
                },
            }
        )
        + "\n"
    )

    n = groq_batch_ingest.ingest(
        batch_output=batch_output,
        metadata_path=metadata_path,
        senses_db=senses_db,
        labeled_db=labeled_db,
    )
    assert n == 0


def test_ingest_rating_zero_stores_zero_sense_key(tmp_path: Path) -> None:
    """A rating-0 response stores sense_key '0' directly without UUID lookup."""
    form = "dog"
    sense = Sense(definition="a domesticated animal")
    senses_db = tmp_path / "senses.db"
    labeled_db = tmp_path / "labeled.db"
    SenseStore(senses_db).write(Alf(form=form, senses=[sense]))
    OccurrenceStore(labeled_db)

    metadata_path = tmp_path / "meta.jsonl"
    metadata_path.write_text(
        json.dumps({"custom_id": "0", "form": form, "doc_id": "d1", "byte_offset": 5})
        + "\n"
    )
    batch_output = tmp_path / "output.jsonl"
    batch_output.write_text(
        json.dumps(
            {
                "custom_id": "0",
                "response": {
                    "body": {
                        "choices": [
                            {"message": {"content": '{"sense_key": "0", "rating": 0}'}}
                        ]
                    }
                },
            }
        )
        + "\n"
    )

    n = groq_batch_ingest.ingest(
        batch_output=batch_output,
        metadata_path=metadata_path,
        senses_db=senses_db,
        labeled_db=labeled_db,
    )
    assert n == 1

    df = OccurrenceStore(labeled_db).query_form(form)
    assert df.row(0, named=True)["sense_key"] == "0"
    assert df.row(0, named=True)["rating"] == 0
