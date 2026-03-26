"""Tests for morph_redirect.run()."""

from pathlib import Path

from alfs.data_models.alf import Alf, Sense
from alfs.data_models.sense_store import SenseStore
from alfs.update.refinement.morph_redirect import run


def test_morph_redirect_skips_redirect_entries(tmp_path: Path, monkeypatch) -> None:
    store = SenseStore(tmp_path / "senses.db")
    store.write(
        Alf(form="colour", senses=[Sense(definition="a colour")], redirect="color")
    )
    store.write(Alf(form="color", senses=[Sense(definition="a color")]))

    calls: list = []

    def fake_chat_json(model, prompt, retries=3, format=None):
        calls.append(prompt)
        return {"candidates": []}

    monkeypatch.setattr(
        "alfs.update.refinement.morph_redirect.llm.chat_json", fake_chat_json
    )

    queue_dir = tmp_path / "queue"
    # Only "color" is eligible (not "colour" which is a redirect)
    # With n=10, seed=0, batch_size=10 — all eligible forms are sampled
    run(tmp_path / "senses.db", queue_dir, n=10, seed=0)

    # Screen prompt should only include "color", not "colour"
    assert len(calls) == 1
    assert "colour" not in calls[0]
    assert "color" in calls[0]


def test_morph_redirect_seeded_sampling_is_reproducible(
    tmp_path: Path, monkeypatch
) -> None:
    store = SenseStore(tmp_path / "senses.db")
    for word in ("cat", "dog", "fish", "bird", "ant"):
        store.write(Alf(form=word, senses=[Sense(definition=f"a {word}")]))

    sampled_batches: list[list] = []

    def fake_chat_json(model, prompt, retries=3, format=None):
        sampled_batches.append(prompt)
        return {"candidates": []}

    monkeypatch.setattr(
        "alfs.update.refinement.morph_redirect.llm.chat_json", fake_chat_json
    )

    queue_dir = tmp_path / "queue"
    run(tmp_path / "senses.db", queue_dir, n=3, batch_size=10, seed=42)
    first_run = list(sampled_batches)

    sampled_batches.clear()
    run(tmp_path / "senses.db", queue_dir, n=3, batch_size=10, seed=42)
    second_run = list(sampled_batches)

    assert first_run == second_run
