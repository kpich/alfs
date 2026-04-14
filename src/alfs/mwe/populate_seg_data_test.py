"""Tests for MWE seg data population."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from alfs.data_models.alf import Alf, Sense
from alfs.data_models.pos import PartOfSpeech
from alfs.data_models.sense_store import SenseStore
from alfs.mwe.populate_seg_data import find_mwe_forms, populate


@pytest.fixture()
def seg_data_dir(tmp_path: Path) -> Path:
    """Create minimal seg data with unigram tokens."""
    d = tmp_path / "by_prefix"
    (d / "a").mkdir(parents=True)
    (d / "p").mkdir(parents=True)
    (d / "t").mkdir(parents=True)

    # "a" tokens
    pl.DataFrame(
        {
            "form": ["a", "a"],
            "doc_id": ["d1", "d2"],
            "byte_offset": [0, 0],
        }
    ).write_parquet(d / "a" / "occurrences.parquet")

    # "priori" and "posteriori" tokens
    pl.DataFrame(
        {
            "form": ["priori", "posteriori"],
            "doc_id": ["d1", "d2"],
            "byte_offset": [2, 2],
        }
    ).write_parquet(d / "p" / "occurrences.parquet")

    # "take", "the" tokens
    pl.DataFrame(
        {
            "form": ["take", "the"],
            "doc_id": ["d1", "d2"],
            "byte_offset": [100, 100],
        }
    ).write_parquet(d / "t" / "occurrences.parquet")

    return d


@pytest.fixture()
def senses_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "senses.db"
    store = SenseStore(db_path)
    # Add an MWE form and a simplex form
    store.write(
        Alf(
            form="a priori",
            senses=[
                Sense(
                    id="s1", definition="from the earlier", pos=PartOfSpeech.adjective
                )
            ],
        )
    )
    store.write(
        Alf(
            form="take",
            senses=[Sense(id="s2", definition="to grab", pos=PartOfSpeech.verb)],
        )
    )
    return db_path


def test_find_mwe_forms():
    import spacy

    nlp = spacy.load("en_core_web_sm")
    nlp.select_pipes(enable=[])

    forms = ["a priori", "take", "well-known", "cat"]
    mwe_forms = find_mwe_forms(nlp, forms)
    mwe_form_names = {f for f, _ in mwe_forms}
    assert "a priori" in mwe_form_names
    assert "well-known" in mwe_form_names
    assert "take" not in mwe_form_names
    assert "cat" not in mwe_form_names


def test_populate_basic(tmp_path: Path, seg_data_dir: Path, senses_db: Path):
    n = populate(senses_db, seg_data_dir)
    # "a priori" should be found in d1 (a@0 + priori@2)
    assert n >= 1

    # Verify the MWE form now appears in seg data
    a_parquet = pl.read_parquet(seg_data_dir / "a" / "occurrences.parquet")
    mwe_rows = a_parquet.filter(pl.col("form") == "a priori")
    assert len(mwe_rows) >= 1
    assert mwe_rows["doc_id"][0] == "d1"


def test_populate_idempotent(tmp_path: Path, seg_data_dir: Path, senses_db: Path):
    n1 = populate(senses_db, seg_data_dir)
    assert n1 >= 1
    n2 = populate(senses_db, seg_data_dir)
    assert n2 == 0


def test_populate_no_mwe_forms(tmp_path: Path, seg_data_dir: Path):
    db_path = tmp_path / "empty_senses.db"
    store = SenseStore(db_path)
    store.write(
        Alf(
            form="cat",
            senses=[Sense(id="s1", definition="a feline", pos=PartOfSpeech.noun)],
        )
    )
    n = populate(db_path, seg_data_dir)
    assert n == 0
