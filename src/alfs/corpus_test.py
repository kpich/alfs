import polars as pl

from alfs.corpus import fetch_instances


def _labeled(rows: list[tuple]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "form": pl.String,
                "doc_id": pl.String,
                "byte_offset": pl.Int64,
                "sense_key": pl.String,
                "rating": pl.Int64,
            }
        )
    forms, doc_ids, offsets, sense_keys, ratings = zip(*rows, strict=False)
    return pl.DataFrame(
        {
            "form": list(forms),
            "doc_id": list(doc_ids),
            "byte_offset": list(offsets),
            "sense_key": list(sense_keys),
            "rating": list(ratings),
        }
    )


def _docs(rows: list[tuple]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema={"doc_id": pl.String, "text": pl.String})
    doc_ids, texts = zip(*rows, strict=False)
    return pl.DataFrame({"doc_id": list(doc_ids), "text": list(texts)})


def test_returns_context_snippets():
    text = "The quick brown fox jumps over the lazy dog"
    labeled = _labeled([("fox", "d1", 10, "1", 3)])
    docs = _docs([("d1", text)])

    result = fetch_instances("fox", "1", labeled, docs)

    assert len(result) == 1
    assert "fox" in result[0]


def test_filters_by_min_rating():
    labeled = _labeled(
        [
            ("run", "d1", 0, "1", 3),
            ("run", "d2", 0, "1", 2),  # below min_rating
        ]
    )
    docs = _docs([("d1", "run fast"), ("d2", "run slow")])

    result = fetch_instances("run", "1", labeled, docs, min_rating=3)

    assert len(result) == 1
    assert "run fast" in result[0]


def test_filters_by_form_and_sense_key():
    labeled = _labeled(
        [
            ("run", "d1", 0, "1", 3),
            ("run", "d2", 0, "2", 3),  # different sense
            ("walk", "d3", 0, "1", 3),  # different form
        ]
    )
    docs = _docs([("d1", "run fast"), ("d2", "run slow"), ("d3", "walk away")])

    result = fetch_instances("run", "1", labeled, docs)

    assert len(result) == 1
    assert "run fast" in result[0]


def test_respects_max_instances():
    labeled = _labeled([("go", f"d{i}", 0, "1", 3) for i in range(20)])
    docs = _docs([(f"d{i}", f"go somewhere {i}") for i in range(20)])

    result = fetch_instances("go", "1", labeled, docs, max_instances=5)

    assert len(result) == 5


def test_skips_missing_docs():
    labeled = _labeled([("cat", "missing_doc", 0, "1", 3)])
    docs = _docs([])

    result = fetch_instances("cat", "1", labeled, docs)

    assert result == []


def test_empty_labeled_returns_empty():
    labeled = _labeled([])
    docs = _docs([("d1", "some text")])

    result = fetch_instances("word", "1", labeled, docs)

    assert result == []
