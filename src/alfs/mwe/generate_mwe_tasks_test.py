"""Tests for MWE task generation."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from alfs.data_models.mwe_queue import MWEQueue, MWEQueueEntry
from alfs.data_models.occurrence import Occurrence
from alfs.mwe.generate_mwe_tasks import run


def _setup_fixtures(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Create seg data, docs, queue, and cc_tasks dirs. Returns paths."""
    # Seg data: tokens for "a priori" in a doc
    seg_dir = tmp_path / "by_prefix"
    for prefix in ["a", "p"]:
        pdir = seg_dir / prefix
        pdir.mkdir(parents=True)

    text = "This is a priori knowledge."
    doc_id = "doc1"
    # "a" at byte 8, "priori" at byte 10
    a_occ = pl.DataFrame(
        {
            "form": ["a"],
            "doc_id": [doc_id],
            "byte_offset": [8],
        }
    )
    p_occ = pl.DataFrame(
        {
            "form": ["priori"],
            "doc_id": [doc_id],
            "byte_offset": [10],
        }
    )
    a_occ.write_parquet(str(seg_dir / "a" / "occurrences.parquet"))
    p_occ.write_parquet(str(seg_dir / "p" / "occurrences.parquet"))

    # Docs
    docs_path = tmp_path / "docs.parquet"
    pl.DataFrame({"doc_id": [doc_id], "text": [text]}).write_parquet(str(docs_path))

    # Queue
    queue_path = tmp_path / "mwe_queue.yaml"
    queue = MWEQueue(queue_path)
    queue.add_candidates(
        [
            MWEQueueEntry(
                form="a priori",
                components=["a", "priori"],
                pmi=12.0,
                corpus_count=100,
                occurrences=[Occurrence(doc_id=doc_id, byte_offset=8)],
            ),
        ]
    )

    cc_tasks_dir = tmp_path / "cc_tasks"
    cc_tasks_dir.mkdir()

    return seg_dir, docs_path, queue_path, cc_tasks_dir


def test_generate_basic(tmp_path: Path):
    seg_dir, docs_path, queue_path, cc_tasks_dir = _setup_fixtures(tmp_path)

    count = run(queue_path, seg_dir, docs_path, cc_tasks_dir, n=10)
    assert count == 1

    pending = list((cc_tasks_dir / "pending" / "mwe").glob("*.json"))
    assert len(pending) == 1

    task = json.loads(pending[0].read_text())
    assert task["type"] == "mwe"
    assert task["form"] == "a priori"
    assert task["components"] == ["a", "priori"]
    assert task["pmi"] == 12.0
    assert len(task["contexts"]) == 1
    assert "priori" in task["contexts"][0]

    # Queue should be drained
    assert len(MWEQueue(queue_path).load()) == 0


def test_generate_empty_queue(tmp_path: Path):
    queue_path = tmp_path / "mwe_queue.yaml"
    seg_dir = tmp_path / "by_prefix"
    seg_dir.mkdir(parents=True)
    docs_path = tmp_path / "docs.parquet"
    pl.DataFrame({"doc_id": [], "text": []}).write_parquet(str(docs_path))
    cc_tasks_dir = tmp_path / "cc_tasks"

    count = run(queue_path, seg_dir, docs_path, cc_tasks_dir, n=10)
    assert count == 0
