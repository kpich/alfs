"""Tests for retag.run()."""

from pathlib import Path

import polars as pl

from alfs.data_models.alf import Alf, Sense
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore
from alfs.update.refinement.retag import run


def _write_empty_docs(path: Path) -> None:
    pl.DataFrame(
        {"doc_id": [], "text": []}, schema={"doc_id": pl.String, "text": pl.String}
    ).write_parquet(path)


def test_retag_skips_redirect_entries(tmp_path: Path, monkeypatch) -> None:
    store = SenseStore(tmp_path / "senses.db")
    store.write(
        Alf(form="colour", senses=[Sense(definition="a colour")], redirect="color")
    )
    store.write(Alf(form="color", senses=[Sense(definition="a color")]))

    OccurrenceStore(tmp_path / "labeled.db")
    docs = tmp_path / "docs.parquet"
    _write_empty_docs(docs)

    calls: list = []

    def fake_chat_json(model, prompt, retries=3, format=None):
        calls.append(prompt)
        return {"pos": "noun"}

    monkeypatch.setattr("alfs.update.refinement.retag.llm.chat_json", fake_chat_json)

    queue_dir = tmp_path / "queue"
    # n=10 — selects all eligible (only "color", not "colour")
    run(tmp_path / "senses.db", tmp_path / "labeled.db", docs, queue_dir, n=10)

    # Only "color" is eligible; "colour" is a redirect and must be skipped
    # Each eligible sense triggers 2 LLM calls (pos + critic) if POS changes
    # Since both return "noun" which matches the initial None POS, there may be 1 call
    # The important thing: no calls at all for "colour"
    for prompt in calls:
        assert "colour" not in prompt
