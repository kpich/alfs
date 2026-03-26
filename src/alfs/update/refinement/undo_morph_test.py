"""Tests for undo_morph.run()."""

from pathlib import Path

from alfs.data_models.alf import Alf, Sense
from alfs.data_models.sense_store import SenseStore
from alfs.update.refinement.undo_morph import run


def test_undo_morph_only_samples_morph_linked_senses(
    tmp_path: Path, monkeypatch
) -> None:
    store = SenseStore(tmp_path / "senses.db")
    # sense with morph link
    linked = Sense(definition="running", morph_base="run", morph_relation="gerund")
    # sense without morph link
    plain = Sense(definition="to move quickly")
    store.write(Alf(form="running", senses=[linked]))
    store.write(Alf(form="sprint", senses=[plain]))

    screen_prompts: list = []

    def fake_chat_json(model, prompt, retries=3, format=None):
        screen_prompts.append(prompt)
        return {"bad_links": []}  # LLM finds no bad links

    monkeypatch.setattr(
        "alfs.update.refinement.undo_morph.llm.chat_json", fake_chat_json
    )

    queue_dir = tmp_path / "queue"
    result = run(tmp_path / "senses.db", queue_dir, n=10, seed=0)

    assert result == 0
    # Should have called LLM once (screening)
    assert len(screen_prompts) == 1
    # "sprint" should not appear in the screening prompt (it has no morph_base)
    assert "sprint" not in screen_prompts[0]
    assert "running" in screen_prompts[0]


def test_undo_morph_skips_when_no_morph_linked_senses(
    tmp_path: Path, monkeypatch
) -> None:
    store = SenseStore(tmp_path / "senses.db")
    store.write(Alf(form="cat", senses=[Sense(definition="a feline")]))

    calls: list = []

    def fake_chat_json(*a, **kw):  # type: ignore[no-untyped-def]
        calls.append(a)
        return {}

    monkeypatch.setattr(
        "alfs.update.refinement.undo_morph.llm.chat_json", fake_chat_json
    )

    queue_dir = tmp_path / "queue"
    result = run(tmp_path / "senses.db", queue_dir)

    assert result == 0
    assert len(calls) == 0  # no LLM calls when nothing to screen
