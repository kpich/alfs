import polars as pl

from alfs.data_models.alf import Alf, Alfs, Sense
from alfs.viewer.compile_qc import compile_qc_coverage


def _sense(defn: str = "a word") -> Sense:
    return Sense(id="test-id", definition=defn)


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


def test_pct_instances_covered_uses_total_key():
    # _total includes untracked corpus tokens — denominator should be 1000 not 100
    alfs = _alfs(Alf(form="cat", senses=[_sense()]))
    corpus_counts = {"cat": 80, "_total": 1000}
    result = compile_qc_coverage(_labeled([]), alfs, corpus_counts, set())
    assert result["pct_instances_covered"] == 8.0  # 80/1000


def test_pct_instances_covered_fallback_without_total():
    alfs = _alfs(
        Alf(form="cat", senses=[_sense()]),
        Alf(form="xyz", senses=[]),
    )
    corpus_counts = {"cat": 80, "xyz": 20}
    result = compile_qc_coverage(_labeled([]), alfs, corpus_counts, set())
    assert result["pct_instances_covered"] == 80.0  # 80/100 fallback


def test_pct_instances_covered_no_defs():
    alfs = _alfs(Alf(form="foo", senses=[]), Alf(form="bar", senses=[]))
    corpus_counts = {"foo": 50, "bar": 50, "_total": 200}
    result = compile_qc_coverage(_labeled([]), alfs, corpus_counts, set())
    assert result["pct_instances_covered"] == 0.0


def test_pct_senses_covered_est_all_excellent():
    alfs = _alfs(Alf(form="cat", senses=[_sense()]))
    corpus_counts = {"cat": 100, "_total": 100}
    labeled = _labeled([("cat", "d1", 0, "s1", 2)] * 20)
    result = compile_qc_coverage(labeled, alfs, corpus_counts, set())
    # 20 excellent/20 labeled → rate=1.0; smoothed=(20+5)/(20+5)=1.0
    assert result["pct_senses_covered_est"] == 100.0


def test_pct_senses_covered_est_smoothing_zero_labeled():
    # global rate = 0 (no labeled), cat has 0 labeled → smoothed = 0
    alfs = _alfs(Alf(form="cat", senses=[_sense()]))
    corpus_counts = {"cat": 100, "_total": 100}
    result = compile_qc_coverage(_labeled([]), alfs, corpus_counts, set())
    assert result["pct_senses_covered_est"] == 0.0


def test_pct_senses_covered_est_uses_global_prior():
    # dog has 10 excellent labels, global rate=1.0; cat has 0 → gets prior 1.0
    alfs = _alfs(
        Alf(form="cat", senses=[_sense()]),
        Alf(form="dog", senses=[_sense()]),
    )
    corpus_counts = {"cat": 50, "dog": 50, "_total": 100}
    labeled = _labeled([("dog", "d1", 0, "s1", 2)] * 10)
    result = compile_qc_coverage(labeled, alfs, corpus_counts, set())
    assert result["pct_senses_covered_est"] == 100.0


def test_first_uncovered_rank_basic():
    # rank 1: cat (has def), rank 2: the (blocklisted), rank 3: xyz (no def, not
    # blocked)
    alfs = _alfs(
        Alf(form="cat", senses=[_sense()]),
        Alf(form="the", senses=[]),
        Alf(form="xyz", senses=[]),
    )
    corpus_counts = {"cat": 300, "the": 200, "xyz": 100}
    result = compile_qc_coverage(_labeled([]), alfs, corpus_counts, {"the"})
    assert result["first_uncovered_rank"] == 3


def test_first_uncovered_rank_none_when_all_covered():
    alfs = _alfs(
        Alf(form="cat", senses=[_sense()]),
        Alf(form="dog", senses=[_sense()]),
    )
    corpus_counts = {"cat": 200, "dog": 100}
    result = compile_qc_coverage(_labeled([]), alfs, corpus_counts, set())
    assert result["first_uncovered_rank"] is None


def test_first_uncovered_rank_blocklisted_not_counted():
    alfs = _alfs(
        Alf(form="cat", senses=[_sense()]),
        Alf(form="xyz", senses=[]),
    )
    corpus_counts = {"cat": 200, "xyz": 100}
    result = compile_qc_coverage(_labeled([]), alfs, corpus_counts, {"xyz"})
    assert result["first_uncovered_rank"] is None


def test_bucket_arrays_length():
    alfs = _alfs(*[Alf(form=f"w{i}", senses=[_sense()]) for i in range(100)])
    corpus_counts = {f"w{i}": 100 - i for i in range(100)}
    result = compile_qc_coverage(_labeled([]), alfs, corpus_counts, set(), n_buckets=10)
    assert len(result["bucket_counts_covered"]) == 10
    assert len(result["bucket_counts_uncovered"]) == 10


def test_bucket_arrays_sum_to_tracked_total():
    alfs = _alfs(
        Alf(form="cat", senses=[_sense()]),
        Alf(form="dog", senses=[]),
    )
    corpus_counts = {"cat": 80, "dog": 20}
    result = compile_qc_coverage(_labeled([]), alfs, corpus_counts, set(), n_buckets=2)
    total = sum(result["bucket_counts_covered"]) + sum(
        result["bucket_counts_uncovered"]
    )
    assert total == 100


def test_first_uncovered_bucket_x_set():
    # 4 forms, n_buckets=2, bucket_size=2
    # rank 1: cat (def), rank 2: dog (def), rank 3: xyz (no def), rank 4: foo (no def)
    # first uncovered = rank 3, bucket 1 (0-based), bucket_x = 1 + 0.5 = 1.5
    alfs = _alfs(
        Alf(form="cat", senses=[_sense()]),
        Alf(form="dog", senses=[_sense()]),
        Alf(form="xyz", senses=[]),
        Alf(form="foo", senses=[]),
    )
    corpus_counts = {"cat": 400, "dog": 300, "xyz": 200, "foo": 100}
    result = compile_qc_coverage(_labeled([]), alfs, corpus_counts, set(), n_buckets=2)
    assert result["first_uncovered_bucket_x"] == 1.5


def test_first_uncovered_bucket_x_none_when_all_covered():
    alfs = _alfs(Alf(form="cat", senses=[_sense()]))
    corpus_counts = {"cat": 100}
    result = compile_qc_coverage(_labeled([]), alfs, corpus_counts, set())
    assert result["first_uncovered_bucket_x"] is None


def test_empty_corpus_returns_zeros():
    alfs = _alfs(Alf(form="cat", senses=[_sense()]))
    result = compile_qc_coverage(_labeled([]), alfs, {}, set())
    assert result["pct_instances_covered"] == 0.0
    assert result["pct_senses_covered_est"] == 0.0
    assert result["first_uncovered_rank"] is None
    assert result["bucket_counts_covered"] == []
    assert result["bucket_counts_uncovered"] == []
