import numpy as np
import polars as pl

from alfs.update.labeling.select_targets import select_top_n


def _occurrences(form_counts: dict[str, int]) -> pl.DataFrame:
    forms = []
    for form, count in form_counts.items():
        forms.extend([form] * count)
    return pl.DataFrame({"form": forms})


def _labeled(rows: list[tuple[str, str, int]], rating: int = 1) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            {"form": [], "doc_id": [], "byte_offset": [], "rating": []},
            schema={
                "form": pl.String,
                "doc_id": pl.String,
                "byte_offset": pl.Int64,
                "rating": pl.Int64,
            },
        )
    forms, doc_ids, offsets = zip(*rows, strict=False)
    return pl.DataFrame(
        {
            "form": list(forms),
            "doc_id": list(doc_ids),
            "byte_offset": list(offsets),
            "rating": [rating] * len(rows),
        }
    )


def test_basic_unlabeled_sort():
    occ = _occurrences({"A": 100, "B": 90})
    lab = _labeled([("A", "doc1", i) for i in range(20)])
    result = select_top_n(occ, lab, top_n=2, rng=np.random.default_rng(0))
    # A has 20 bad labels → bad_rate≈0.95, score~76; B has none → bad_rate=0.5, score~45
    # A ranks first due to high bad rate despite fewer raw unlabeled occurrences
    assert result[0] == "A"


def test_all_labeled_form_loses():
    occ = _occurrences({"A": 50, "B": 40})
    lab = _labeled([("A", "doc1", i) for i in range(50)])
    result = select_top_n(occ, lab, top_n=2, rng=np.random.default_rng(0))
    # A has 0 unlabeled → score=0 always; B has 40 unlabeled → B first
    assert result[0] == "B"


def test_no_labeled_data():
    occ = _occurrences({"A": 100, "B": 90})
    lab = _labeled([])
    result = select_top_n(occ, lab, top_n=2, rng=np.random.default_rng(0))
    # No labeled data → both cold-start with bad_rate=0.5; both should be returned
    assert set(result) == {"A", "B"}


def test_top_n_caps_result():
    occ = _occurrences({"a": 5, "b": 4, "c": 3, "d": 2, "e": 1})
    lab = _labeled([])
    result = select_top_n(occ, lab, top_n=3, rng=np.random.default_rng(0))
    assert len(result) == 3


def test_non_letter_forms_filtered():
    occ = _occurrences({"!!": 100, "ok": 90})
    lab = _labeled([])
    result = select_top_n(occ, lab, top_n=10, rng=np.random.default_rng(0))
    assert "!!" not in result
    assert "ok" in result


def test_duplicate_labeled_rows_deduped():
    occ = _occurrences({"the": 10})
    # 3 rows with same (doc_id, byte_offset) — should count as 1 after dedup
    lab = _labeled([("the", "doc1", 0), ("the", "doc1", 0), ("the", "doc1", 0)])
    result = select_top_n(occ, lab, top_n=1, rng=np.random.default_rng(0))
    # 10 total - 1 unique labeled = 9 unlabeled; form should still be returned
    assert result == ["the"]


def test_rating_zero_treated_as_unlabeled():
    occ = _occurrences({"A": 10})
    lab = _labeled([("A", "doc1", i) for i in range(10)], rating=0)
    result = select_top_n(occ, lab, top_n=1, rng=np.random.default_rng(0))
    assert result == ["A"]  # only form with letters; returned even with score=0


def test_high_bad_rate_beats_high_volume():
    # X: few occurrences but 80% bad rate; Y: many occurrences but all good labels
    occ = _occurrences({"X": 20, "Y": 200})
    lab_x = _labeled([("X", "doc1", i) for i in range(4)], rating=1)  # 4 bad
    lab_y = _labeled([("Y", "doc2", i) for i in range(50)], rating=3)  # 50 good
    lab = pl.concat([lab_x, lab_y])
    result = select_top_n(occ, lab, top_n=2, rng=np.random.default_rng(0))
    # X: 16 unlabeled, bad_rate=5/6≈0.83 → expected score~13
    # Y: 150 unlabeled, bad_rate=1/52≈0.02 → expected score~3
    assert result[0] == "X"


def test_cold_start_form_gets_half_rate():
    # Form with no labels gets bad_rate = (0+1)/(0+2) = 0.5 via Beta(1,1) prior
    occ = _occurrences({"coldword": 20})
    lab = _labeled([])
    result = select_top_n(occ, lab, top_n=1, rng=np.random.default_rng(0))
    # With bad_rate=0.5 and 20 unlabeled, expected score=10; form should be selected
    assert result == ["coldword"]


def test_all_excellent_ratings_deprioritized():
    # Form with all EXCELLENT labels gets very low bad_rate → low score
    occ = _occurrences({"excellent": 50, "unknown": 10})
    lab = _labeled([("excellent", "doc1", i) for i in range(30)], rating=3)
    result = select_top_n(occ, lab, top_n=2, rng=np.random.default_rng(0))
    # "unknown": 10 unlabeled, bad_rate=0.5, expected score=5
    # "excellent": 20 unlabeled, bad_rate=1/32≈0.03, expected score~0.6
    assert result[0] == "unknown"


def test_redirect_forms_excluded():
    occ = _occurrences({"The": 100, "the": 90})
    lab = _labeled([])
    result = select_top_n(
        occ, lab, top_n=10, rng=np.random.default_rng(0), redirect_forms={"The"}
    )
    assert "The" not in result
    assert "the" in result
