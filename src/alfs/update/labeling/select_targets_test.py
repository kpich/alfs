import numpy as np
import polars as pl

from alfs.update.labeling.select_targets import select_top_n


def _total_counts(form_counts: dict[str, int]) -> pl.DataFrame:
    return pl.DataFrame(
        {"form": list(form_counts.keys()), "total": list(form_counts.values())}
    )


def _n_labeled_counts(form_labeled: dict[str, int]) -> pl.DataFrame:
    if not form_labeled:
        return pl.DataFrame(
            {"form": [], "n_labeled": []},
            schema={"form": pl.String, "n_labeled": pl.Int64},
        )
    return pl.DataFrame(
        {"form": list(form_labeled.keys()), "n_labeled": list(form_labeled.values())}
    )


def test_basic_unlabeled_sort():
    tc = _total_counts({"A": 100, "B": 90})
    nlc = _n_labeled_counts({"A": 20})
    result = select_top_n(tc, nlc, top_n=2, rng=np.random.default_rng(0), min_count=1)
    # A has 80 unlabeled; B has 90 unlabeled → both have nonzero weight, both returned
    assert set(result) == {"A", "B"}


def test_all_labeled_form_loses():
    tc = _total_counts({"A": 50, "B": 40})
    nlc = _n_labeled_counts({"A": 50})
    result = select_top_n(tc, nlc, top_n=2, rng=np.random.default_rng(0), min_count=1)
    # A has 0 unlabeled → score=0 always; B has 40 unlabeled → B first
    assert result[0] == "B"


def test_no_labeled_data():
    tc = _total_counts({"A": 100, "B": 90})
    nlc = _n_labeled_counts({})
    result = select_top_n(tc, nlc, top_n=2, rng=np.random.default_rng(0), min_count=1)
    # No labeled data → both cold-start with bad_rate=0.5; both should be returned
    assert set(result) == {"A", "B"}


def test_top_n_caps_result():
    tc = _total_counts({"a": 5, "b": 4, "c": 3, "d": 2, "e": 1})
    nlc = _n_labeled_counts({})
    result = select_top_n(tc, nlc, top_n=3, rng=np.random.default_rng(0), min_count=1)
    assert len(result) == 3


def test_non_letter_forms_filtered():
    tc = _total_counts({"!!": 100, "ok": 90})
    nlc = _n_labeled_counts({})
    result = select_top_n(tc, nlc, top_n=10, rng=np.random.default_rng(0), min_count=1)
    assert "!!" not in result
    assert "ok" in result


def test_rating_zero_treated_as_unlabeled():
    tc = _total_counts({"A": 10})
    nlc = _n_labeled_counts({"A": 10})
    result = select_top_n(tc, nlc, top_n=1, rng=np.random.default_rng(0), min_count=1)
    # All 10 occurrences labeled → unlabeled=0 → excluded
    assert result == []


def test_high_volume_wins():
    # X: few unlabeled; Y: many unlabeled regardless of rating quality
    tc = _total_counts({"X": 20, "Y": 200})
    nlc = _n_labeled_counts({"X": 4, "Y": 50})
    result = select_top_n(tc, nlc, top_n=2, rng=np.random.default_rng(0), min_count=1)
    # X: 16 unlabeled; Y: 150 unlabeled → Y has much higher weight, selected first
    assert result[0] == "Y"


def test_cold_start_form_gets_half_rate():
    # Form with no labels has all occurrences unlabeled → nonzero weight → selected
    tc = _total_counts({"coldword": 20})
    nlc = _n_labeled_counts({})
    result = select_top_n(tc, nlc, top_n=1, rng=np.random.default_rng(0), min_count=1)
    # All 20 occurrences unlabeled → weight=20 → form is selected
    assert result == ["coldword"]


def test_all_excellent_ratings_deprioritized():
    # "excellent" has more unlabeled instances than "unknown" despite high label
    # coverage
    tc = _total_counts({"excellent": 50, "unknown": 10})
    nlc = _n_labeled_counts({"excellent": 30})
    result = select_top_n(tc, nlc, top_n=2, rng=np.random.default_rng(0), min_count=1)
    # "excellent": 20 unlabeled; "unknown": 10 unlabeled → both selected (top_n=2)
    assert set(result) == {"excellent", "unknown"}


def test_redirect_forms_excluded():
    tc = _total_counts({"The": 100, "the": 90})
    nlc = _n_labeled_counts({})
    result = select_top_n(
        tc,
        nlc,
        top_n=10,
        rng=np.random.default_rng(0),
        redirect_forms={"The"},
        min_count=1,
    )
    assert "The" not in result
    assert "the" in result


def test_min_count_excludes_rare_forms():
    """Forms below min_count are excluded from selection."""
    total = _total_counts({"apple": 10, "rare": 2})
    labeled = _n_labeled_counts({})
    rng = np.random.default_rng(0)
    result = select_top_n(total, labeled, top_n=10, rng=rng, min_count=5)
    assert "rare" not in result
    assert "apple" in result


def test_min_count_1_includes_all():
    """min_count=1 passes all forms with at least one occurrence."""
    total = _total_counts({"a": 1, "b": 3})
    labeled = _n_labeled_counts({})
    rng = np.random.default_rng(0)
    result = select_top_n(total, labeled, top_n=10, rng=rng, min_count=1)
    assert set(result) == {"a", "b"}


def test_log_weight_reduces_head_dominance():
    """A high-volume form does not completely crowd out a mid-volume form."""
    total = _total_counts({"common": 10_000, "mid": 100})
    labeled = _n_labeled_counts({})
    # With raw weights, mid is selected with probability ~1 %; with log weights ~67 %.
    # Run 200 trials and assert mid is chosen at least once.
    chosen = set()
    for seed in range(200):
        chosen |= set(
            select_top_n(
                total, labeled, top_n=1, rng=np.random.default_rng(seed), min_count=1
            )
        )
    assert "mid" in chosen
