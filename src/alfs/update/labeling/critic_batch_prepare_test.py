"""Unit tests for critic_batch_prepare."""

import json
from pathlib import Path

import polars as pl

from alfs.data_models.alf import Alf, Sense
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore
from alfs.update.labeling.critic_batch_prepare import (
    build_critic_system_message,
    run,
)


def _sense_store(tmp_path: Path, *entries: Alf) -> SenseStore:
    store = SenseStore(tmp_path / "senses.db")
    for entry in entries:
        store.write(entry)
    return store


def _occ_store(tmp_path: Path) -> OccurrenceStore:
    return OccurrenceStore(tmp_path / "labeled.db")


def _docs_parquet(tmp_path: Path, docs: dict[str, str]) -> Path:
    path = tmp_path / "docs.parquet"
    pl.DataFrame(
        {"doc_id": list(docs.keys()), "text": list(docs.values())}
    ).write_parquet(str(path))
    return path


# --- build_critic_system_message ---


def test_system_message_contains_form_and_definition() -> None:
    msg = build_critic_system_message("dog", "a domesticated carnivorous mammal")
    assert '"dog"' in msg
    assert "a domesticated carnivorous mammal" in msg
    assert "bad_indices" in msg


# --- run ---


def test_run_empty_labeled_returns_no_chunks(tmp_path: Path) -> None:
    sense = Sense(definition="a domestic animal")
    store = _sense_store(tmp_path, Alf(form="dog", senses=[sense]))
    occ = _occ_store(tmp_path)
    docs = _docs_parquet(tmp_path, {"doc1": "The dog ran fast."})

    chunks = run(
        senses_db=store._db_path,
        labeled_db=occ._db_path,
        docs=docs,
        output_dir=tmp_path / "out",
        min_instances=1,
    )
    assert chunks == []


def test_run_produces_batch_and_metadata(tmp_path: Path) -> None:
    sense = Sense(definition="a domestic animal")
    sense_id = sense.id
    store = _sense_store(tmp_path, Alf(form="dog", senses=[sense]))
    occ = _occ_store(tmp_path)

    # Add enough instances to exceed min_instances
    text = "The dog ran fast down the street to find its owner."
    doc_ids = [f"doc{i}" for i in range(6)]
    occ.upsert_many(
        [
            (store.sense_id_to_form()[sense_id], d, 4, sense_id, 2, None)
            for d in doc_ids
        ],
        model="test-model",
    )
    docs = _docs_parquet(tmp_path, {d: text for d in doc_ids})

    chunks = run(
        senses_db=store._db_path,
        labeled_db=occ._db_path,
        docs=docs,
        output_dir=tmp_path / "out",
        min_instances=5,
        instances_per_sense=10,
        seed=0,
    )

    assert len(chunks) == 1
    batch_path, meta_path = chunks[0]

    # batch_input has one request
    batch_lines = [
        json.loads(line) for line in batch_path.read_text().splitlines() if line
    ]
    assert len(batch_lines) == 1
    req = batch_lines[0]
    assert req["method"] == "POST"
    assert len(req["body"]["messages"]) == 2
    assert "dog" in req["body"]["messages"][0]["content"]

    # metadata has one entry with 6 instances
    meta_lines = [
        json.loads(line) for line in meta_path.read_text().splitlines() if line
    ]
    assert len(meta_lines) == 1
    meta = meta_lines[0]
    assert meta["form"] == "dog"
    assert meta["sense_uuid"] == sense_id
    assert len(meta["instances"]) == 6


def test_run_skips_sense_below_min_instances(tmp_path: Path) -> None:
    sense = Sense(definition="a domestic animal")
    sense_id = sense.id
    store = _sense_store(tmp_path, Alf(form="dog", senses=[sense]))
    occ = _occ_store(tmp_path)

    text = "The dog ran fast."
    # Only 3 instances — below default min_instances=5
    occ.upsert_many(
        [
            (store.sense_id_to_form()[sense_id], f"doc{i}", 4, sense_id, 2, None)
            for i in range(3)
        ],
        model="test-model",
    )
    docs = _docs_parquet(tmp_path, {f"doc{i}": text for i in range(3)})

    chunks = run(
        senses_db=store._db_path,
        labeled_db=occ._db_path,
        docs=docs,
        output_dir=tmp_path / "out",
        min_instances=5,
    )
    assert chunks == []


def test_run_skips_already_reviewed_instances(tmp_path: Path) -> None:
    sense = Sense(definition="a domestic animal")
    sense_id = sense.id
    store = _sense_store(tmp_path, Alf(form="dog", senses=[sense]))
    occ = _occ_store(tmp_path)

    text = "The dog ran fast."
    form = store.sense_id_to_form()[sense_id]
    occ.upsert_many(
        [(form, f"doc{i}", 4, sense_id, 2, None) for i in range(6)],
        model="test-model",
    )
    # Mark all as reviewed with a timestamp far in the future (after any sense update)
    reviewed = [(form, f"doc{i}", 4) for i in range(6)]
    occ.mark_critic_reviewed(reviewed, "2099-01-01T00:00:00Z", model="critic-model")
    docs = _docs_parquet(tmp_path, {f"doc{i}": text for i in range(6)})

    chunks = run(
        senses_db=store._db_path,
        labeled_db=occ._db_path,
        docs=docs,
        output_dir=tmp_path / "out",
        min_instances=5,
    )
    # All reviewed and no sense update since review → no new work
    assert chunks == []


def test_run_uses_surface_form_for_context_highlighting(tmp_path: Path) -> None:
    """Occurrences stored under a different surface form (e.g. 'dogs') must be
    highlighted as **dogs** in the critic prompt, not as **dog**s."""
    sense = Sense(definition="a domestic animal")
    sense_id = sense.id
    store = _sense_store(tmp_path, Alf(form="dog", senses=[sense]))
    occ = _occ_store(tmp_path)

    # Label occurrences under surface form "dogs" but with dog's sense UUID
    text = "The dogs barked loudly outside."  # "dogs" starts at byte 4
    doc_ids = [f"doc{i}" for i in range(5)]
    occ.upsert_many(
        [("dogs", d, 4, sense_id, 1, None) for d in doc_ids],
        model="test-model",
    )
    docs = _docs_parquet(tmp_path, {d: text for d in doc_ids})

    chunks = run(
        senses_db=store._db_path,
        labeled_db=occ._db_path,
        docs=docs,
        output_dir=tmp_path / "out",
        min_instances=5,
        seed=0,
    )

    assert len(chunks) == 1
    batch_lines = [
        json.loads(line) for line in chunks[0][0].read_text().splitlines() if line
    ]
    user_msg = batch_lines[0]["body"]["messages"][1]["content"]
    assert "**dogs**" in user_msg
    assert "**dog**s" not in user_msg


def test_run_sub_batch_splitting(tmp_path: Path) -> None:
    """With max_batch_size=1 and 2 senses, produces 2 chunk files."""
    sense_a = Sense(definition="a domestic animal")
    sense_b = Sense(definition="to pursue an animal")
    form_a = "dog"
    form_b = "hunt"
    store = _sense_store(
        tmp_path,
        Alf(form=form_a, senses=[sense_a]),
        Alf(form=form_b, senses=[sense_b]),
    )
    occ = _occ_store(tmp_path)

    text_a = "The dog ran fast down the street."
    text_b = "They hunt for deer in the forest every autumn."
    for form, sense_id, _ in [
        (form_a, sense_a.id, text_a),
        (form_b, sense_b.id, text_b),
    ]:
        occ.upsert_many(
            [(form, f"{form}_doc{i}", 4, sense_id, 2, None) for i in range(5)],
            model="test-model",
        )

    # write combined docs parquet
    all_docs: dict[str, str] = {}
    for form, text in [(form_a, text_a), (form_b, text_b)]:
        for i in range(5):
            all_docs[f"{form}_doc{i}"] = text
    docs = _docs_parquet(tmp_path, all_docs)

    chunks = run(
        senses_db=store._db_path,
        labeled_db=occ._db_path,
        docs=docs,
        output_dir=tmp_path / "out",
        min_instances=5,
        max_batch_size=1,
        seed=0,
    )

    assert len(chunks) == 2
    # Verify file names follow the pattern
    for batch_path, meta_path in chunks:
        assert "critic_input_" in batch_path.name
        assert "critic_metadata_" in meta_path.name
