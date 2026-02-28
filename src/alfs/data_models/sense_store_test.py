from pathlib import Path
import threading

import pytest

from alfs.data_models.alf import Alf, Sense
from alfs.data_models.sense_store import SenseStore


def _alf(form: str, *definitions: str) -> Alf:
    return Alf(form=form, senses=[Sense(definition=d) for d in definitions])


@pytest.fixture
def store(tmp_path: Path) -> SenseStore:
    return SenseStore(tmp_path / "senses.db")


def test_write_and_read(store: SenseStore) -> None:
    entry = _alf("run", "to move quickly")
    store.write(entry)
    result = store.read("run")
    assert result is not None
    assert result.form == "run"
    assert result.senses[0].definition == "to move quickly"


def test_read_missing_returns_none(store: SenseStore) -> None:
    assert store.read("nonexistent") is None


def test_write_replaces_existing(store: SenseStore) -> None:
    store.write(_alf("run", "to move quickly"))
    store.write(_alf("run", "to manage"))
    result = store.read("run")
    assert result is not None
    assert len(result.senses) == 1
    assert result.senses[0].definition == "to manage"


def test_delete(store: SenseStore) -> None:
    store.write(_alf("run", "to move quickly"))
    store.delete("run")
    assert store.read("run") is None


def test_all_forms(store: SenseStore) -> None:
    store.write(_alf("zebra", "striped animal"))
    store.write(_alf("apple", "a fruit"))
    forms = store.all_forms()
    assert forms == ["apple", "zebra"]  # sorted


def test_all_entries(store: SenseStore) -> None:
    store.write(_alf("cat", "a feline"))
    store.write(_alf("dog", "a canine"))
    entries = store.all_entries()
    assert set(entries.keys()) == {"cat", "dog"}
    assert entries["cat"].senses[0].definition == "a feline"


def test_update_existing(store: SenseStore) -> None:
    store.write(_alf("run", "to move quickly"))

    def add_sense(existing: Alf | None) -> Alf:
        assert existing is not None
        return Alf(
            form="run",
            senses=list(existing.senses) + [Sense(definition="to manage")],
        )

    store.update("run", add_sense)
    result = store.read("run")
    assert result is not None
    assert len(result.senses) == 2


def test_update_missing_form(store: SenseStore) -> None:
    def create_entry(existing: Alf | None) -> Alf:
        assert existing is None
        return _alf("new", "a new entry")

    store.update("new", create_entry)
    result = store.read("new")
    assert result is not None
    assert result.senses[0].definition == "a new entry"


def test_concurrent_rmw(tmp_path: Path) -> None:
    """Concurrent update() calls must not lose data."""
    db_path = tmp_path / "senses.db"
    store = SenseStore(db_path)
    store.write(Alf(form="word", senses=[]))

    errors: list[Exception] = []

    def append_sense(label: str) -> None:
        try:

            def fn(existing: Alf | None) -> Alf:
                existing_senses = list(existing.senses) if existing else []
                return Alf(
                    form="word",
                    senses=existing_senses + [Sense(definition=label)],
                )

            SenseStore(db_path).update("word", fn)
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=append_sense, args=(f"sense{i}",)) for i in range(5)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    result = store.read("word")
    assert result is not None
    assert len(result.senses) == 5
