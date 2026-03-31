"""Tests for postag._make_tagger()."""

import polars as pl

from alfs.data_models.alf import Alf, Sense
from alfs.data_models.pos import PartOfSpeech
from alfs.update.refinement.postag import _make_tagger


def _empty_dfs() -> tuple[pl.DataFrame, pl.DataFrame]:
    labeled_df = pl.DataFrame(
        {
            "form": [],
            "doc_id": [],
            "byte_offset": [],
            "sense_key": [],
            "rating": [],
            "model": [],
            "updated_at": [],
            "synonyms": [],
        },
        schema={
            "form": pl.String,
            "doc_id": pl.String,
            "byte_offset": pl.Int64,
            "sense_key": pl.String,
            "rating": pl.Int64,
            "model": pl.String,
            "updated_at": pl.String,
            "synonyms": pl.String,
        },
    )
    docs_df = pl.DataFrame(
        {"doc_id": [], "text": []}, schema={"doc_id": pl.String, "text": pl.String}
    )
    return labeled_df, docs_df


def test_make_tagger_skips_senses_with_existing_pos(monkeypatch) -> None:
    labeled_df, docs_df = _empty_dfs()
    calls: list = []

    def fake_chat_json(model, prompt, retries=3, format=None):
        calls.append(prompt)
        return {"pos": "verb"}

    monkeypatch.setattr("alfs.update.refinement.postag.llm.chat_json", fake_chat_json)

    sense_with_pos = Sense(definition="to run", pos=PartOfSpeech.verb)
    sense_without_pos = Sense(definition="a sprint")
    existing = Alf(form="run", senses=[sense_with_pos, sense_without_pos])

    tagger = _make_tagger("run", labeled_df, docs_df, "test-model")
    result = tagger(existing)

    # Only the sense without POS should have triggered LLM calls (2: tag + critic)
    assert len(calls) == 2
    # sense_with_pos unchanged
    assert result.senses[0].pos == PartOfSpeech.verb
    assert result.senses[0].id == sense_with_pos.id
    # sense_without_pos now has POS set
    assert result.senses[1].pos == PartOfSpeech.verb
