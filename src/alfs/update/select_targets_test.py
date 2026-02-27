import polars as pl

from alfs.update.select_targets import select_top_n


def _occurrences(form_counts: dict[str, int]) -> pl.DataFrame:
    forms = []
    for form, count in form_counts.items():
        forms.extend([form] * count)
    return pl.DataFrame({"form": forms})


def _labeled(rows: list[tuple[str, str, int]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            {"form": [], "doc_id": [], "byte_offset": []},
            schema={"form": pl.String, "doc_id": pl.String, "byte_offset": pl.Int64},
        )
    forms, doc_ids, offsets = zip(*rows, strict=False)
    return pl.DataFrame(
        {"form": list(forms), "doc_id": list(doc_ids), "byte_offset": list(offsets)}
    )


def test_basic_unlabeled_sort():
    occ = _occurrences({"A": 100, "B": 90})
    lab = _labeled([("A", "doc1", i) for i in range(20)])
    result = select_top_n(occ, lab, top_n=2)
    # A has 80 unlabeled, B has 90 unlabeled → B first
    assert result[0] == "B"


def test_all_labeled_form_loses():
    occ = _occurrences({"A": 50, "B": 40})
    lab = _labeled([("A", "doc1", i) for i in range(50)])
    result = select_top_n(occ, lab, top_n=2)
    # A has 0 unlabeled, B has 40 unlabeled → B first
    assert result[0] == "B"


def test_no_labeled_data():
    occ = _occurrences({"A": 100, "B": 90})
    lab = _labeled([])
    result = select_top_n(occ, lab, top_n=2)
    # No labeled data → sort by total → A first
    assert result[0] == "A"


def test_top_n_caps_result():
    occ = _occurrences({"a": 5, "b": 4, "c": 3, "d": 2, "e": 1})
    lab = _labeled([])
    result = select_top_n(occ, lab, top_n=3)
    assert len(result) == 3


def test_non_letter_forms_filtered():
    occ = _occurrences({"!!": 100, "ok": 90})
    lab = _labeled([])
    result = select_top_n(occ, lab, top_n=10)
    assert "!!" not in result
    assert "ok" in result


def test_duplicate_labeled_rows_deduped():
    occ = _occurrences({"the": 10})
    # 3 rows with same (doc_id, byte_offset) — should count as 1 after dedup
    lab = _labeled([("the", "doc1", 0), ("the", "doc1", 0), ("the", "doc1", 0)])
    result = select_top_n(occ, lab, top_n=1)
    # 10 total - 1 unique labeled = 9 unlabeled; form should still be returned
    assert result == ["the"]
