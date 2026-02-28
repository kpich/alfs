import polars as pl

from alfs.data_models.alf import Alf, Alfs, Sense
from alfs.viewer.compile import compile_entries


def _alfs(*alfs: Alf) -> Alfs:
    return Alfs(entries={alf.form: alf for alf in alfs})


def _labeled(rows: list[tuple]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            {
                "form": [],
                "doc_id": [],
                "byte_offset": [],
                "sense_key": [],
                "rating": [],
            },
            schema={
                "form": pl.String,
                "doc_id": pl.String,
                "byte_offset": pl.Int64,
                "sense_key": pl.String,
                "rating": pl.Int64,
            },
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
        return pl.DataFrame(
            schema={"doc_id": pl.String, "year": pl.Int64, "text": pl.String}
        )
    doc_ids, years, texts = zip(*rows, strict=False)
    return pl.DataFrame(
        {"doc_id": list(doc_ids), "year": list(years), "text": list(texts)}
    )


def test_redirect_forms_excluded():
    alfs = _alfs(
        Alf(form="the", senses=[Sense(definition="definite article")]),
        Alf(form="The", senses=[], redirect="the"),
    )
    labeled = _labeled([])
    docs = _docs([("doc1", 2020, "")])

    result = compile_entries(alfs, labeled, docs)

    assert "the" in result
    assert "The" not in result
    assert 1 <= result["the"]["percentile"] <= 100


def test_non_redirect_forms_included():
    alfs = _alfs(
        Alf(form="run", senses=[Sense(definition="to move quickly")]),
    )
    labeled = _labeled([("run", "doc1", 0, "1", 2)])
    docs = _docs([("doc1", 2020, "run fast")])

    result = compile_entries(alfs, labeled, docs)

    assert "run" in result
    assert result["run"]["senses"][0]["definition"] == "to move quickly"
    assert result["run"]["by_year"]["2020"]["1"] == 1
    assert 1 <= result["run"]["percentile"] <= 100


def test_percentile_ordering():
    alfs = _alfs(
        Alf(form="common", senses=[Sense(definition="frequently seen")]),
        Alf(form="rare", senses=[Sense(definition="seldom seen")]),
    )
    labeled = _labeled(
        [
            ("common", "doc1", 0, "1", 2),
            ("common", "doc2", 0, "1", 2),
            ("common", "doc3", 0, "1", 2),
            ("rare", "doc4", 0, "1", 2),
        ]
    )
    docs = _docs(
        [("doc1", 2020, ""), ("doc2", 2020, ""), ("doc3", 2020, ""), ("doc4", 2020, "")]
    )

    result = compile_entries(alfs, labeled, docs)

    assert result["common"]["percentile"] < result["rare"]["percentile"]


def test_instances_included_per_sense():
    alfs = _alfs(
        Alf(form="run", senses=[Sense(definition="to move quickly")]),
    )
    labeled = _labeled([("run", "doc1", 0, "1", 3)])
    docs = _docs([("doc1", 2020, "run fast through the park")])

    result = compile_entries(alfs, labeled, docs)

    instances = result["run"]["senses"][0]["instances"]
    assert len(instances) == 1
    assert "<strong>run</strong>" in instances[0]


def test_instances_empty_when_no_rating3():
    alfs = _alfs(
        Alf(form="walk", senses=[Sense(definition="to move on foot")]),
    )
    labeled = _labeled([("walk", "doc1", 0, "1", 2)])
    docs = _docs([("doc1", 2020, "walk slowly")])

    result = compile_entries(alfs, labeled, docs)

    assert result["walk"]["senses"][0]["instances"] == []
