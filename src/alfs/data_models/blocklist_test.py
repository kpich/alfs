"""Tests for Blocklist."""

from pathlib import Path

import yaml

from alfs.data_models.blocklist import Blocklist


def test_load_absent_file_returns_empty(tmp_path: Path):
    bl = Blocklist(tmp_path / "blocklist.yaml")
    assert bl.load() == {}


def test_contains_false_when_absent(tmp_path: Path):
    bl = Blocklist(tmp_path / "blocklist.yaml")
    assert not bl.contains("thrumbly")


def test_add_and_contains(tmp_path: Path):
    bl = Blocklist(tmp_path / "blocklist.yaml")
    bl.add("thrumbly", reason="tokenization artifact")
    assert bl.contains("thrumbly")
    assert not bl.contains("cat")


def test_add_with_reason(tmp_path: Path):
    bl = Blocklist(tmp_path / "blocklist.yaml")
    bl.add("de", reason="foreign word")
    data = bl.load()
    assert data["de"] == "foreign word"


def test_add_without_reason(tmp_path: Path):
    bl = Blocklist(tmp_path / "blocklist.yaml")
    bl.add("xyz")
    data = bl.load()
    assert "xyz" in data
    assert data["xyz"] is None


def test_add_many_returns_count(tmp_path: Path):
    bl = Blocklist(tmp_path / "blocklist.yaml")
    count = bl.add_many(["a", "b", "c"])
    assert count == 3


def test_add_many_idempotent(tmp_path: Path):
    bl = Blocklist(tmp_path / "blocklist.yaml")
    bl.add_many(["a", "b"])
    count = bl.add_many(["b", "c"])  # b is already there
    assert count == 1
    assert set(bl.load().keys()) == {"a", "b", "c"}


def test_sorted_output(tmp_path: Path):
    path = tmp_path / "blocklist.yaml"
    bl = Blocklist(path)
    bl.add_many(["zebra", "apple", "mango"])
    raw = yaml.safe_load(path.read_text())
    keys = list(raw.keys())
    assert keys == sorted(keys)


def test_load_round_trip(tmp_path: Path):
    bl = Blocklist(tmp_path / "blocklist.yaml")
    bl.add("cat", reason="test reason")
    bl.add("dog", reason=None)
    data = bl.load()
    assert data["cat"] == "test reason"
    assert data["dog"] is None


def test_add_does_not_overwrite_reason(tmp_path: Path):
    bl = Blocklist(tmp_path / "blocklist.yaml")
    bl.add("cat", reason="original reason")
    bl.add("cat", reason="new reason")  # should not overwrite
    assert bl.load()["cat"] == "original reason"
