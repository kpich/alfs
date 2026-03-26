"""Tests for trim_sense.run()."""

from pathlib import Path

import polars as pl

from alfs.data_models.alf import Alf, Sense
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore
from alfs.update.refinement.trim_sense import run


def _write_empty_docs(path: Path) -> None:
    pl.DataFrame(
        {"doc_id": [], "text": []}, schema={"doc_id": pl.String, "text": pl.String}
    ).write_parquet(path)


def test_trim_sense_requires_at_least_two_senses(tmp_path: Path, monkeypatch) -> None:
    store = SenseStore(tmp_path / "senses.db")
    # one sense only — not eligible
    store.write(Alf(form="cat", senses=[Sense(definition="a feline")]))
    # two senses — eligible
    store.write(
        Alf(
            form="run",
            senses=[Sense(definition="to move"), Sense(definition="to manage")],
        )
    )

    OccurrenceStore(tmp_path / "labeled.db")
    docs = tmp_path / "docs.parquet"
    _write_empty_docs(docs)

    calls: list = []

    def fake_chat_json(model, prompt, retries=3, format=None):
        calls.append(prompt)
        return {"sense_num": None, "reason": "all distinct"}

    monkeypatch.setattr(
        "alfs.update.refinement.trim_sense.llm.chat_json", fake_chat_json
    )

    queue_dir = tmp_path / "queue"
    run(tmp_path / "senses.db", tmp_path / "labeled.db", docs, queue_dir, n=10)

    # Only "run" (2 senses) should be evaluated; "cat" (1 sense) is ineligible
    assert len(calls) == 1
    assert "run" in calls[0]
    assert "cat" not in calls[0]


def test_trim_sense_skips_redirect_entries(tmp_path: Path, monkeypatch) -> None:
    store = SenseStore(tmp_path / "senses.db")
    store.write(
        Alf(
            form="colour",
            senses=[Sense(definition="s1"), Sense(definition="s2")],
            redirect="color",
        )
    )

    OccurrenceStore(tmp_path / "labeled.db")
    docs = tmp_path / "docs.parquet"
    _write_empty_docs(docs)

    calls: list = []

    def fake_chat_json(*a, **kw):  # type: ignore[no-untyped-def]
        calls.append(a)
        return {"sense_num": None, "reason": ""}

    monkeypatch.setattr(
        "alfs.update.refinement.trim_sense.llm.chat_json", fake_chat_json
    )

    queue_dir = tmp_path / "queue"
    run(tmp_path / "senses.db", tmp_path / "labeled.db", docs, queue_dir, n=10)

    assert len(calls) == 0
