from pathlib import Path

from alfs.data_models.mwe_skipped import MWESkipped


def test_load_empty(tmp_path: Path):
    s = MWESkipped(tmp_path / "mwe_skipped.yaml")
    assert s.load() == []


def test_add_and_load(tmp_path: Path):
    s = MWESkipped(tmp_path / "mwe_skipped.yaml")
    s.add("the president")
    assert "the president" in s.load()


def test_add_idempotent(tmp_path: Path):
    s = MWESkipped(tmp_path / "mwe_skipped.yaml")
    s.add("the president")
    s.add("the president")
    assert s.load().count("the president") == 1


def test_add_many(tmp_path: Path):
    s = MWESkipped(tmp_path / "mwe_skipped.yaml")
    added = s.add_many(["the president", "take note", "the president"])
    assert added == 2
    assert len(s.load()) == 2


def test_sorted(tmp_path: Path):
    s = MWESkipped(tmp_path / "mwe_skipped.yaml")
    s.add("zebra crossing")
    s.add("apple pie")
    forms = s.load()
    assert forms == sorted(forms)
