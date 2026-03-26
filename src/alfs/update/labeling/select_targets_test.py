import numpy as np
import polars as pl

from alfs.update.labeling.select_targets import select_top_n


def _total_counts(form_counts: dict[str, int]) -> pl.DataFrame:
    return pl.DataFrame(
        {"form": list(form_counts.keys()), "total": list(form_counts.values())}
    )


def _labeled_counts(
    n_labeled: dict[str, int], n_covered: dict[str, int] | None = None
) -> pl.DataFrame:
    """Build labeled_counts DataFrame with columns ["form", "n_labeled", "n_covered"].

    n_covered defaults to equal n_labeled (all labeled instances are covered).
    """
    if n_covered is None:
        n_covered = n_labeled
    forms = list(n_labeled.keys())
    return (
        pl.DataFrame(
            {
                "form": forms,
                "n_labeled": [n_labeled[f] for f in forms],
                "n_covered": [n_covered.get(f, 0) for f in forms],
            },
            schema={"form": pl.String, "n_labeled": pl.Int64, "n_covered": pl.Int64},
        )
        if forms
        else pl.DataFrame(
            schema={"form": pl.String, "n_labeled": pl.Int64, "n_covered": pl.Int64}
        )
    )


def test_cold_start_selects_form():
    # No labels → bad_rate=1.0 → need_work=total → nonzero weight → selected
    tc = _total_counts({"coldword": 20})
    lc = _labeled_counts({})
    result = select_top_n(tc, lc, top_n=1, rng=np.random.default_rng(0), min_count=1)
    assert result == ["coldword"]


def test_cold_start_two_forms_both_returned():
    tc = _total_counts({"A": 100, "B": 90})
    lc = _labeled_counts({})
    result = select_top_n(tc, lc, top_n=2, rng=np.random.default_rng(0), min_count=1)
    assert set(result) == {"A", "B"}


def test_all_labeled_good_excluded_without_smoothing():
    # A: all 50 labeled and covered → bad_rate=0 → weight=0 → excluded
    # B: no labels → bad_rate=1.0 → weight=sqrt(40) → selected
    tc = _total_counts({"A": 50, "B": 40})
    lc = _labeled_counts({"A": 50})
    result = select_top_n(tc, lc, top_n=2, rng=np.random.default_rng(0), min_count=1)
    assert "A" not in result
    assert result == ["B"]


def test_all_labeled_good_included_with_smoothing():
    # A: 10 labeled all covered → bad_rate=alpha/(10+alpha) > 0 with smoothing
    # B: no labels → bad_rate=1.0; both have nonzero weight
    tc = _total_counts({"A": 10, "B": 10})
    lc = _labeled_counts({"A": 10})
    result = select_top_n(
        tc, lc, top_n=2, rng=np.random.default_rng(0), min_count=1, smoothing_alpha=1.0
    )
    assert set(result) == {"A", "B"}


def test_bad_rate_extrapolated_over_corpus():
    # A: 500 labeled, 250 covered → bad_rate=0.5 → need_work=1000*0.5=500
    # B: 500 labeled, 500 covered → bad_rate=0 → need_work=0 → excluded
    tc = _total_counts({"A": 1000, "B": 1000})
    lc = _labeled_counts(n_labeled={"A": 500, "B": 500}, n_covered={"A": 250, "B": 500})
    result = select_top_n(tc, lc, top_n=2, rng=np.random.default_rng(0), min_count=1)
    assert result == ["A"]


def test_high_bad_rate_wins_over_low():
    # X: 100 labeled, 90 covered → bad_rate=0.1 → need_work=20*0.1=2
    # Y: 100 labeled, 0 covered → bad_rate=1.0 → need_work=200*1.0=200
    tc = _total_counts({"X": 20, "Y": 200})
    lc = _labeled_counts(n_labeled={"X": 100, "Y": 100}, n_covered={"X": 90, "Y": 0})
    result = select_top_n(tc, lc, top_n=2, rng=np.random.default_rng(0), min_count=1)
    assert result[0] == "Y"


def test_top_n_caps_result():
    tc = _total_counts({"a": 5, "b": 4, "c": 3, "d": 2, "e": 1})
    lc = _labeled_counts({})
    result = select_top_n(tc, lc, top_n=3, rng=np.random.default_rng(0), min_count=1)
    assert len(result) == 3


def test_non_letter_forms_filtered():
    tc = _total_counts({"!!": 100, "ok": 90})
    lc = _labeled_counts({})
    result = select_top_n(tc, lc, top_n=10, rng=np.random.default_rng(0), min_count=1)
    assert "!!" not in result
    assert "ok" in result


def test_redirect_forms_included():
    tc = _total_counts({"The": 100, "the": 90})
    lc = _labeled_counts({})
    result = select_top_n(tc, lc, top_n=10, rng=np.random.default_rng(0), min_count=1)
    assert "The" in result
    assert "the" in result


def test_min_count_excludes_rare_forms():
    total = _total_counts({"apple": 10, "rare": 2})
    lc = _labeled_counts({})
    result = select_top_n(
        total, lc, top_n=10, rng=np.random.default_rng(0), min_count=5
    )
    assert "rare" not in result
    assert "apple" in result


def test_min_count_1_includes_all():
    total = _total_counts({"a": 1, "b": 3})
    lc = _labeled_counts({})
    result = select_top_n(
        total, lc, top_n=10, rng=np.random.default_rng(0), min_count=1
    )
    assert set(result) == {"a", "b"}


def test_sqrt_weight_reduces_head_dominance():
    """A high-volume form does not completely crowd out a mid-volume form."""
    total = _total_counts({"common": 10_000, "mid": 100})
    lc = _labeled_counts({})
    # With raw weights, mid is selected ~1% of the time; with sqrt weights ~9%.
    # Run 200 trials and assert mid is chosen at least once.
    chosen = set()
    for seed in range(200):
        chosen |= set(
            select_top_n(
                total, lc, top_n=1, rng=np.random.default_rng(seed), min_count=1
            )
        )
    assert "mid" in chosen
