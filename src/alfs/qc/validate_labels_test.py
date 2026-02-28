import polars as pl

from alfs.qc.validate_labels import validate


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


def test_valid_label_not_flagged():
    text = "The quick brown fox jumps"
    byte_offset = len(b"The quick brown ")
    labeled = _labeled([("fox", "d1", byte_offset, "1", 3)])
    docs = _docs([("d1", text)])

    stale = validate(labeled, docs)

    assert len(stale) == 0


def test_stale_label_flagged():
    # Label says "fox" but text at that offset is "dog"
    text = "The quick brown dog jumps"
    byte_offset = len(b"The quick brown ")
    labeled = _labeled([("fox", "d1", byte_offset, "1", 3)])
    docs = _docs([("d1", text)])

    stale = validate(labeled, docs)

    assert len(stale) == 1
    assert stale["form"][0] == "fox"


def test_orphaned_label_not_flagged():
    labeled = _labeled([("fox", "missing_doc", 0, "1", 3)])
    docs = _docs([])

    stale = validate(labeled, docs)

    assert len(stale) == 0


def test_mixed_returns_only_stale():
    text = "hello world"
    labeled = _labeled(
        [
            ("hello", "d1", 0, "1", 3),  # valid
            ("world", "d1", len(b"hello "), "1", 3),  # valid
            ("goodbye", "d1", 0, "1", 3),  # stale — "hello" != "goodbye"
            ("fox", "missing", 0, "1", 3),  # orphaned — not flagged
        ]
    )
    docs = _docs([("d1", text)])

    stale = validate(labeled, docs)

    assert len(stale) == 1
    assert stale["form"][0] == "goodbye"
