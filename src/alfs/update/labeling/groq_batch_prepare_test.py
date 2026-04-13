"""Unit tests for groq_batch_prepare."""

import polars as pl

from alfs.data_models.alf import Alf, Sense
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore
from alfs.update.labeling.groq_batch_prepare import (
    allocate_proportional,
    run,
    sense_weight,
    split_labeled_pairs,
)


def _store(tmp_path, *entries: Alf) -> SenseStore:
    store = SenseStore(tmp_path / "senses.db")
    for entry in entries:
        store.write(entry)
    return store


# --- sense_weight ---


def test_sense_weight_no_quality_labels_equals_sense_count(tmp_path):
    """With no quality=2 labels, each sense contributes 1/sqrt(1) = 1.0."""
    entry = Alf(
        form="run",
        senses=[Sense(definition="to move quickly"), Sense(definition="to operate")],
    )
    store = _store(tmp_path, entry)
    assert sense_weight(entry, store, {}) == 2.0


def test_sense_weight_no_senses_returns_zero(tmp_path):
    entry = Alf(form="Run", senses=[])
    store = _store(tmp_path, entry)
    assert sense_weight(entry, store, {}) == 0.0


def test_sense_weight_case_variant_includes_lowercase_senses(tmp_path):
    """A case variant with no own senses still sees the canonical form's senses."""
    canonical = Alf(
        form="run",
        senses=[Sense(definition="to move quickly"), Sense(definition="to operate")],
    )
    alias = Alf(form="Run", senses=[])
    store = _store(tmp_path, canonical, alias)
    assert sense_weight(alias, store, {}) == 2.0


def test_sense_weight_decreases_with_quality_labels(tmp_path):
    """A form with quality=2 labels on all senses has lower weight than one without."""
    s1 = Sense(definition="sense one")
    s2 = Sense(definition="sense two")
    entry = Alf(form="run", senses=[s1, s2])
    store = _store(tmp_path, entry)
    w_no_labels = sense_weight(entry, store, {})
    quality_counts = {"run": {s1.id: 10, s2.id: 10}}
    w_with_labels = sense_weight(entry, store, quality_counts)
    assert w_with_labels < w_no_labels


def test_sense_weight_partial_coverage_intermediate(tmp_path):
    """A form where only one sense has labels has weight between zero-coverage and
    full-coverage."""
    s1 = Sense(definition="sense one")
    s2 = Sense(definition="sense two")
    entry = Alf(form="run", senses=[s1, s2])
    store = _store(tmp_path, entry)
    w_none = sense_weight(entry, store, {})
    w_all = sense_weight(entry, store, {"run": {s1.id: 10, s2.id: 10}})
    w_half = sense_weight(entry, store, {"run": {s1.id: 10}})
    assert w_all < w_half < w_none


# --- allocate_proportional ---


def test_allocate_proportional_to_weights() -> None:
    """With equal corpus, allocation is proportional to weight."""
    result = allocate_proportional(
        sense_weights={"a": 1.0, "b": 2.0},
        good_labeled={},
        corpus_counts={"a": 1_000_000, "b": 1_000_000},
        budget=300,
    )
    # b should get ~2x a (int truncation may cause ±1 difference)
    assert abs(result["b"] - 2 * result["a"]) <= 1
    assert sum(result.values()) <= 300


def test_allocate_proportional_cap_by_available() -> None:
    """Scarce form is capped at available; remaining budget goes to ample form."""
    result = allocate_proportional(
        sense_weights={"scarce": 1.0, "ample": 1.0},
        good_labeled={"scarce": 990},
        corpus_counts={"scarce": 1_000, "ample": 1_000_000},
        budget=500,
    )
    assert result.get("scarce", 0) <= 10
    assert result.get("ample", 0) > result.get("scarce", 0)
    assert sum(result.values()) <= 500


def test_allocate_proportional_budget_met() -> None:
    """Sum of allocations is close to budget (may be slightly under due to int
    truncation)."""
    budget = 600
    result = allocate_proportional(
        sense_weights={"a": 1.0, "b": 2.0, "c": 3.0},
        good_labeled={},
        corpus_counts={"a": 1_000_000, "b": 1_000_000, "c": 1_000_000},
        budget=budget,
    )
    total = sum(result.values())
    assert total <= budget
    assert total >= budget - len(result)


def test_allocate_proportional_insufficient_corpus_returns_all_available() -> None:
    """When total available < budget, return all available instances."""
    result = allocate_proportional(
        sense_weights={"a": 1.0, "b": 1.0},
        good_labeled={"a": 90, "b": 90},
        corpus_counts={"a": 100, "b": 100},
        budget=10_000,
    )
    assert result.get("a", 0) == 10
    assert result.get("b", 0) == 10


def test_allocate_proportional_zero_weight_excluded() -> None:
    """Forms with weight=0 receive no allocation."""
    result = allocate_proportional(
        sense_weights={"a": 0.0, "b": 1.0},
        good_labeled={},
        corpus_counts={"a": 1_000_000, "b": 1_000_000},
        budget=100,
    )
    assert "a" not in result
    assert "b" in result


def test_allocate_proportional_below_min_count_excluded() -> None:
    """Forms with fewer than min_count corpus occurrences are excluded."""
    result = allocate_proportional(
        sense_weights={"rare": 1.0, "common": 1.0},
        good_labeled={},
        corpus_counts={"rare": 3, "common": 1_000_000},
        budget=100,
        min_count=5,
    )
    assert "rare" not in result
    assert "common" in result


def test_allocate_proportional_saturated_form_excluded() -> None:
    """Forms with available=0 (all corpus instances well-labeled) get no allocation."""
    result = allocate_proportional(
        sense_weights={"full": 1.0, "empty": 1.0},
        good_labeled={"full": 1_000_000},
        corpus_counts={"full": 1_000_000, "empty": 1_000_000},
        budget=100,
    )
    assert "full" not in result
    assert "empty" in result


def test_allocate_proportional_bad_labels_do_not_block_resampling() -> None:
    """Instances with rating 0 (bad) are available for resampling (not in
    good_labeled)."""
    result = allocate_proportional(
        sense_weights={"noisy": 1.0},
        good_labeled={"noisy": 0},
        corpus_counts={"noisy": 1_000_000},
        budget=200,
    )
    assert result.get("noisy", 0) > 0


def test_allocate_proportional_empty_inputs_returns_empty() -> None:
    result = allocate_proportional(
        sense_weights={},
        good_labeled={},
        corpus_counts={},
        budget=100,
    )
    assert result == {}


# --- split_labeled_pairs ---


def _make_labeled_df(
    rows: list[tuple[str, str, int, str | None]],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "form": [r[0] for r in rows],
            "doc_id": [r[1] for r in rows],
            "byte_offset": [r[2] for r in rows],
            "updated_at": [r[3] for r in rows],
        }
    )


def test_split_labeled_pairs_stale_goes_to_stale() -> None:
    df = _make_labeled_df([("run", "doc1", 100, "2024-01-01 00:00:00")])
    max_sense_ts = {"run": "2024-06-01 00:00:00"}
    good, stale = split_labeled_pairs(df, max_sense_ts)
    assert ("doc1", 100) not in good["run"]
    assert ("doc1", 100) in stale["run"]


def test_split_labeled_pairs_fresh_goes_to_good() -> None:
    df = _make_labeled_df([("run", "doc1", 100, "2024-09-01 00:00:00")])
    max_sense_ts = {"run": "2024-06-01 00:00:00"}
    good, stale = split_labeled_pairs(df, max_sense_ts)
    assert ("doc1", 100) in good["run"]
    assert ("doc1", 100) not in stale.get("run", [])


def test_split_labeled_pairs_no_sense_ts_goes_to_good() -> None:
    df = _make_labeled_df([("run", "doc1", 100, "2024-01-01 00:00:00")])
    good, stale = split_labeled_pairs(df, max_sense_ts={})
    assert ("doc1", 100) in good["run"]
    assert len(stale.get("run", [])) == 0


def test_split_labeled_pairs_null_labeled_ts_goes_to_good() -> None:
    df = _make_labeled_df([("run", "doc1", 100, None)])
    max_sense_ts = {"run": "2024-06-01 00:00:00"}
    good, stale = split_labeled_pairs(df, max_sense_ts)
    assert ("doc1", 100) in good["run"]
    assert len(stale.get("run", [])) == 0


def test_split_labeled_pairs_mixed() -> None:
    df = _make_labeled_df(
        [
            ("run", "doc1", 100, "2024-01-01 00:00:00"),  # stale
            ("run", "doc2", 200, "2024-09-01 00:00:00"),  # fresh
            ("run", "doc3", 300, "2024-01-01 00:00:00"),  # stale
        ]
    )
    max_sense_ts = {"run": "2024-06-01 00:00:00"}
    good, stale = split_labeled_pairs(df, max_sense_ts)
    assert ("doc2", 200) in good["run"]
    assert ("doc1", 100) in stale["run"]
    assert ("doc3", 300) in stale["run"]
    assert len(stale["run"]) == 2


# --- run (integration) ---


def test_run_case_variants_share_lowercase_pool(tmp_path) -> None:
    """Case variants share the same lowercase occurrence pool; budget is not doubled."""
    senses = [
        Sense(definition="a domesticated canine"),
        Sense(definition="informal: ugly person"),
    ]
    canonical = Alf(form="dog", senses=senses)
    uppercase = Alf(form="DOG", senses=[])

    store = SenseStore(tmp_path / "senses.db")
    store.write(canonical)
    store.write(uppercase)

    OccurrenceStore(tmp_path / "labeled.db")

    n = 40
    rows = [{"form": "dog", "doc_id": f"doc{i}", "byte_offset": 0} for i in range(n)]
    occ_df = pl.DataFrame(rows)
    prefix_dir = tmp_path / "by_prefix" / "d"
    prefix_dir.mkdir(parents=True)
    occ_df.write_parquet(str(prefix_dir / "occurrences.parquet"))

    all_doc_ids = [f"doc{i}" for i in range(n)]
    docs_df = pl.DataFrame(
        {"doc_id": all_doc_ids, "text": ["the dog ran away"] * len(all_doc_ids)}
    )
    docs_path = tmp_path / "docs.parquet"
    docs_df.write_parquet(str(docs_path))

    out_dir = tmp_path / "out"
    budget = 10

    pairs = run(
        senses_db=tmp_path / "senses.db",
        labeled_db=tmp_path / "labeled.db",
        seg_data_dir=tmp_path / "by_prefix",
        docs=docs_path,
        output_dir=out_dir,
        n=budget,
        seed=0,
    )
    assert len(pairs) == 1
    batch_path, _ = pairs[0]
    total = sum(1 for _ in batch_path.read_text().splitlines() if _)
    assert total <= budget + 1
