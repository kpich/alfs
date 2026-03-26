"""Tests for delete_entry.run()."""

from pathlib import Path

import polars as pl

from alfs.data_models.alf import Alf, Sense
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore
from alfs.update.refinement.delete_entry import run


def _write_empty_docs(path: Path) -> None:
    pl.DataFrame(
        {"doc_id": [], "text": []}, schema={"doc_id": pl.String, "text": pl.String}
    ).write_parquet(path)


def test_delete_entry_selects_rarest_first(tmp_path: Path, monkeypatch) -> None:
    store = SenseStore(tmp_path / "senses.db")
    s_common = Sense(definition="a common word")
    s_rare = Sense(definition="a rare word")
    store.write(Alf(form="common", senses=[s_common]))
    store.write(Alf(form="rare", senses=[s_rare]))

    occ_store = OccurrenceStore(tmp_path / "labeled.db")
    # "common" has 10 occurrences, "rare" has 1
    for i in range(10):
        occ_store.upsert_many(
            [("common", f"doc{i}", i, s_common.id, 1, None)], model="m"
        )
    occ_store.upsert_many([("rare", "doc0", 0, s_rare.id, 1, None)], model="m")

    docs = tmp_path / "docs.parquet"
    _write_empty_docs(docs)

    selected_forms: list[str] = []

    def fake_chat_json(model, prompt, retries=3, format=None):
        # Record which form was prompted
        selected_forms.append(prompt)
        return {"should_delete": False, "reason": "keep"}

    monkeypatch.setattr(
        "alfs.update.refinement.delete_entry.llm.chat_json", fake_chat_json
    )

    queue_dir = tmp_path / "queue"
    # n=1 selects only the rarest form
    run(tmp_path / "senses.db", tmp_path / "labeled.db", docs, queue_dir, n=1)

    assert len(selected_forms) == 1
    assert "rare" in selected_forms[0]
    assert "common" not in selected_forms[0]


def test_delete_entry_skips_redirect_entries(tmp_path: Path, monkeypatch) -> None:
    store = SenseStore(tmp_path / "senses.db")
    store.write(
        Alf(form="colour", senses=[Sense(definition="a colour")], redirect="color")
    )
    store.write(Alf(form="color", senses=[Sense(definition="a color")]))

    OccurrenceStore(tmp_path / "labeled.db")
    docs = tmp_path / "docs.parquet"
    _write_empty_docs(docs)

    calls: list = []

    def fake_chat_json(*a, **kw):  # type: ignore[no-untyped-def]
        calls.append(a[1])
        return {"should_delete": False, "reason": ""}

    monkeypatch.setattr(
        "alfs.update.refinement.delete_entry.llm.chat_json", fake_chat_json
    )

    queue_dir = tmp_path / "queue"
    run(tmp_path / "senses.db", tmp_path / "labeled.db", docs, queue_dir, n=10)

    for prompt in calls:
        assert "colour" not in prompt
