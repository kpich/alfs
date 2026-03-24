"""Integration tests for the full rewrite pipeline:
LLM stub → rewrite.run() → clerk drain → senses.db.
"""

from pathlib import Path

from alfs.clerk.queue import drain
from alfs.data_models.alf import Alf, Sense
from alfs.data_models.sense_store import SenseStore
from alfs.update.refinement import rewrite
from integration_tests.fake_llm import FakeLLM


def test_rewrite_pipeline_updates_sense_definition(tmp_path: Path, monkeypatch) -> None:
    sense = Sense(definition="old definition", updated_by_model=None)
    senses_db = tmp_path / "senses.db"
    store = SenseStore(senses_db)
    store.write(Alf(form="word", senses=[sense]))

    queue_dir = tmp_path / "queue"

    fake = FakeLLM(
        [
            {
                "rewrites": [
                    {
                        "sense_num": 1,
                        "definition": "new improved definition",
                    }
                ]
            },
            {"is_improvement": True, "reason": "better"},
        ]
    )
    monkeypatch.setattr("alfs.update.refinement.rewrite.llm.chat_json", fake.chat_json)

    rewrite.run(senses_db=senses_db, queue_dir=queue_dir, n=1, model="test-model")

    store2 = SenseStore(senses_db)
    drain(queue_dir, store2, None)

    entry = store2.read("word")
    assert entry is not None
    assert entry.senses[0].definition == "new improved definition"
    assert entry.senses[0].id == sense.id
    assert len(list((queue_dir / "done").glob("*.json"))) == 1


def test_rewrite_pipeline_critic_rejection_leaves_definition_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    sense = Sense(definition="original definition", updated_by_model=None)
    senses_db = tmp_path / "senses.db"
    store = SenseStore(senses_db)
    store.write(Alf(form="word", senses=[sense]))

    queue_dir = tmp_path / "queue"

    fake = FakeLLM(
        [
            {"rewrites": [{"sense_num": 1, "definition": "worse rewrite"}]},
            {"is_improvement": False, "reason": "not better"},
        ]
    )
    monkeypatch.setattr("alfs.update.refinement.rewrite.llm.chat_json", fake.chat_json)

    rewrite.run(senses_db=senses_db, queue_dir=queue_dir, n=1, model="test-model")

    entry = SenseStore(senses_db).read("word")
    assert entry is not None
    assert entry.senses[0].definition == "original definition"
    assert len(list((queue_dir / "pending").glob("*.json"))) == 0


def test_rewrite_pipeline_skips_redirect_entries(tmp_path: Path, monkeypatch) -> None:
    senses_db = tmp_path / "senses.db"
    store = SenseStore(senses_db)
    store.write(Alf(form="colour", senses=[], redirect="color"))

    queue_dir = tmp_path / "queue"

    fake = FakeLLM([])
    monkeypatch.setattr("alfs.update.refinement.rewrite.llm.chat_json", fake.chat_json)

    rewrite.run(senses_db=senses_db, queue_dir=queue_dir, n=1, model="test-model")

    assert len(fake.calls) == 0
