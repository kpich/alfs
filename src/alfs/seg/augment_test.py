from pathlib import Path

import polars as pl

from alfs.seg.augment import _get_segmented_doc_ids


def _write_occ_parquet(path: Path, doc_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "doc_id": doc_ids,
            "form": ["x"] * len(doc_ids),
            "byte_offset": [0] * len(doc_ids),
        }
    ).write_parquet(path)


def test_get_segmented_doc_ids_empty_when_no_parquets(tmp_path: Path) -> None:
    assert _get_segmented_doc_ids(tmp_path) == set()


def test_get_segmented_doc_ids_collects_ids_across_prefixes(tmp_path: Path) -> None:
    _write_occ_parquet(tmp_path / "a" / "occurrences.parquet", ["abc"])
    _write_occ_parquet(tmp_path / "b" / "occurrences.parquet", ["xyz"])
    assert _get_segmented_doc_ids(tmp_path) == {"abc", "xyz"}
