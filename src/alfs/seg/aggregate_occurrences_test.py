from pathlib import Path

import polars as pl

from alfs.seg.aggregate_occurrences import aggregate


def _make_occ_df(rows: list[tuple[str, str, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "form": [r[0] for r in rows],
            "doc_id": [r[1] for r in rows],
            "byte_offset": [r[2] for r in rows],
        }
    )


def test_aggregate_creates_prefix_dirs(tmp_path: Path) -> None:
    df = _make_occ_df([("apple", "doc1", 0), ("banana", "doc2", 5)])
    aggregate(df, tmp_path)
    assert (tmp_path / "a" / "occurrences.parquet").exists()
    assert (tmp_path / "b" / "occurrences.parquet").exists()


def test_aggregate_merge_true_preserves_existing_rows(tmp_path: Path) -> None:
    existing = _make_occ_df([("ant", "doc1", 0)])
    aggregate(existing, tmp_path)

    new_row = _make_occ_df([("arc", "doc2", 10)])
    aggregate(new_row, tmp_path, merge=True)

    result = pl.read_parquet(tmp_path / "a" / "occurrences.parquet")
    assert set(result["doc_id"].to_list()) == {"doc1", "doc2"}
    assert len(result) == 2


def test_aggregate_merge_false_overwrites(tmp_path: Path) -> None:
    existing = _make_occ_df([("ant", "doc1", 0)])
    aggregate(existing, tmp_path)

    new_row = _make_occ_df([("arc", "doc2", 10)])
    aggregate(new_row, tmp_path, merge=False)

    result = pl.read_parquet(tmp_path / "a" / "occurrences.parquet")
    assert result["doc_id"].to_list() == ["doc2"]
    assert len(result) == 1
