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


# --- case-insensitive coverage ---


def test_case_insensitive_coverage():
    # "the" has a def; "The" in corpus should count as covered
    alfs = _alfs(Alf(form="the", senses=[_sense()]))
    corpus_counts = {"the": 500, "The": 200, "_total": 700}
    result = compile_qc_coverage(_labeled([]), alfs, corpus_counts, set())
    assert result["pct_instances_covered"] == 100.0


def test_case_insensitive_first_uncovered():
    # "the" has def; "The" in top_corpus_forms should not count as uncovered
    alfs = _alfs(Alf(form="the", senses=[_sense()]))
    corpus_counts = {"the": 500, "_total": 700}
    top_corpus_forms = {"The": 200, "the": 500, "xyz": 100}
    result = compile_qc_coverage(
        _labeled([]),
        alfs,
        corpus_counts,
        set(),
        top_corpus_forms=top_corpus_forms,
    )
    assert result["first_uncovered_form"] == "xyz"
    assert result["first_uncovered_rank"] == 3


def test_case_insensitive_sense_coverage():
    # "the" has def; labeled data uses "The" → should be aggregated
    alfs = _alfs(Alf(form="the", senses=[_sense()]))
    corpus_counts = {"the": 100, "_total": 100}
    labeled = _labeled([("The", "d1", 0, "s1", 2)] * 20)
    result = compile_qc_coverage(labeled, alfs, corpus_counts, set())
    # Global rate = 1.0 (all excellent), smoothed = 1.0
    assert result["pct_senses_covered_est"] == 100.0


# --- winsorization (right-truncation) ---


def test_winsorization_drops_sparse_tail():
    # 4 forms: ranks 1-2 have high counts (>= min_bucket_count), rank 3-4 are sparse
    alfs = _alfs(
        Alf(form="cat", senses=[_sense()]),
        Alf(form="dog", senses=[_sense()]),
        Alf(form="xyz", senses=[]),
        Alf(form="foo", senses=[]),
    )
    corpus_counts = {"cat": 400, "dog": 300, "xyz": 10, "foo": 5}
    # n_buckets=2, bucket_size=2; bucket0=(cat,dog)=700, bucket1=(xyz,foo)=15
    # min_bucket_count=50 → bucket1 dropped
    result = compile_qc_coverage(
        _labeled([]),
        alfs,
        corpus_counts,
        set(),
        n_buckets=2,
        min_bucket_count=50,
    )
    assert len(result["bucket_counts_covered"]) == 1
    assert len(result["bucket_counts_uncovered"]) == 1


def test_winsorization_line_outside_shown_range():
    # first uncovered is in a bucket that gets winsorized away → no line
    alfs = _alfs(Alf(form="cat", senses=[_sense()]))
    corpus_counts = {"cat": 400, "xyz": 5}
    top_corpus_forms = {"cat": 400, "xyz": 5}
    result = compile_qc_coverage(
        _labeled([]),
        alfs,
        corpus_counts,
        set(),
        n_buckets=2,
        min_bucket_count=50,
        top_corpus_forms=top_corpus_forms,
    )
    # xyz is uncovered (rank 2) but its bucket is winsorized
    assert result["first_uncovered_rank"] == 2
    assert result["first_uncovered_bucket_x"] is None


# --- instance coverage ---


def test_pct_instances_covered_uses_total_key():
    alfs = _alfs(Alf(form="cat", senses=[_sense()]))
    corpus_counts = {"cat": 80, "_total": 1000}
    result = compile_qc_coverage(_labeled([]), alfs, corpus_counts, set())
    assert result["pct_instances_covered"] == 8.0


def test_pct_instances_covered_fallback_without_total():
    alfs = _alfs(
        Alf(form="cat", senses=[_sense()]),
        Alf(form="xyz", senses=[]),
    )
    corpus_counts = {"cat": 80, "xyz": 20}
    result = compile_qc_coverage(_labeled([]), alfs, corpus_counts, set())
    assert result["pct_instances_covered"] == 80.0


def test_pct_instances_covered_no_defs():
    alfs = _alfs(Alf(form="foo", senses=[]), Alf(form="bar", senses=[]))
    corpus_counts = {"foo": 50, "bar": 50, "_total": 200}
    result = compile_qc_coverage(_labeled([]), alfs, corpus_counts, set())
    assert result["pct_instances_covered"] == 0.0


# --- sense coverage ---


def test_pct_senses_covered_est_all_excellent():
    alfs = _alfs(Alf(form="cat", senses=[_sense()]))
    corpus_counts = {"cat": 100, "_total": 100}
    labeled = _labeled([("cat", "d1", 0, "s1", 2)] * 20)
    result = compile_qc_coverage(labeled, alfs, corpus_counts, set())
    assert result["pct_senses_covered_est"] == 100.0


def test_pct_senses_covered_est_smoothing_zero_labeled():
    alfs = _alfs(Alf(form="cat", senses=[_sense()]))
    corpus_counts = {"cat": 100, "_total": 100}
    result = compile_qc_coverage(_labeled([]), alfs, corpus_counts, set())
    assert result["pct_senses_covered_est"] == 0.0


# --- first uncovered ---


def test_first_uncovered_rank_basic():
    alfs = _alfs(
        Alf(form="cat", senses=[_sense()]),
        Alf(form="the", senses=[]),
        Alf(form="xyz", senses=[]),
    )
    corpus_counts = {"cat": 300, "the": 200, "xyz": 100}
    result = compile_qc_coverage(_labeled([]), alfs, corpus_counts, {"the"})
    assert result["first_uncovered_rank"] == 3
    assert result["first_uncovered_form"] == "xyz"


def test_first_uncovered_rank_none_when_all_covered():
    alfs = _alfs(
        Alf(form="cat", senses=[_sense()]),
        Alf(form="dog", senses=[_sense()]),
    )
    corpus_counts = {"cat": 200, "dog": 100}
    result = compile_qc_coverage(_labeled([]), alfs, corpus_counts, set())
    assert result["first_uncovered_rank"] is None


def test_first_uncovered_blocklist_checks_lowercase():
    # blocklist uses lowercase; corpus has mixed-case form
    alfs = _alfs(Alf(form="cat", senses=[_sense()]))
    corpus_counts = {"cat": 200, "The": 100}
    top_corpus_forms = {"cat": 200, "The": 100}
    result = compile_qc_coverage(
        _labeled([]),
        alfs,
        corpus_counts,
        {"the"},
        top_corpus_forms=top_corpus_forms,
    )
    assert result["first_uncovered_rank"] is None  # "The" → blocklisted via "the"


# --- buckets ---


def test_bucket_arrays_sum_to_shown_total():
    alfs = _alfs(
        Alf(form="cat", senses=[_sense()]),
        Alf(form="dog", senses=[]),
    )
    corpus_counts = {"cat": 800, "dog": 200}
    result = compile_qc_coverage(
        _labeled([]),
        alfs,
        corpus_counts,
        set(),
        n_buckets=2,
        min_bucket_count=1,
    )
    total = sum(result["bucket_counts_covered"]) + sum(
        result["bucket_counts_uncovered"]
    )
    assert total == 1000


def test_top_corpus_forms_shows_untracked_words():
    alfs = _alfs(Alf(form="cat", senses=[_sense()]))
    corpus_counts = {"cat": 100, "_total": 200}
    top_corpus_forms = {"the": 100, "cat": 100}
    result = compile_qc_coverage(
        _labeled([]),
        alfs,
        corpus_counts,
        set(),
        n_buckets=2,
        min_bucket_count=1,
        top_corpus_forms=top_corpus_forms,
    )
    assert sum(result["bucket_counts_uncovered"]) > 0


def test_empty_corpus_returns_zeros():
    alfs = _alfs(Alf(form="cat", senses=[_sense()]))
    result = compile_qc_coverage(_labeled([]), alfs, {}, set())
    assert result["pct_instances_covered"] == 0.0
    assert result["pct_senses_covered_est"] == 0.0
    assert result["first_uncovered_rank"] is None
    assert result["bucket_counts_covered"] == []
    assert result["bucket_counts_uncovered"] == []
