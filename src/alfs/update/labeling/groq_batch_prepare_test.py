"""Unit tests for groq_batch_prepare.allocate_instances and effective_sense_count."""

from alfs.data_models.alf import Alf, Sense
from alfs.data_models.sense_store import SenseStore
from alfs.update.labeling.groq_batch_prepare import (
    allocate_instances,
    effective_sense_count,
)


def _store(tmp_path, *entries: Alf) -> SenseStore:
    store = SenseStore(tmp_path / "senses.db")
    for entry in entries:
        store.write(entry)
    return store


def test_effective_sense_count_redirect_uses_canonical_senses(tmp_path):
    canonical = Alf(
        form="run",
        senses=[Sense(definition="to move quickly"), Sense(definition="to operate")],
    )
    alias = Alf(form="Run", senses=[], redirect="run")
    store = _store(tmp_path, canonical, alias)
    assert effective_sense_count(alias, store) == 2


def test_effective_sense_count_redirect_missing_target_returns_zero(tmp_path):
    alias = Alf(form="Run", senses=[], redirect="run")
    store = _store(tmp_path, alias)
    assert effective_sense_count(alias, store) == 0


def test_allocate_proportional_to_senses() -> None:
    """With no existing labels, allocation is proportional to effective sense count."""
    result = allocate_instances(
        effective_senses={"a": 1, "b": 2},
        existing_labeled={},
        good_labeled={},
        corpus_counts={"a": 1_000_000, "b": 1_000_000},
        budget=300,
    )
    # b should get ~2x a (int truncation may cause ±1 difference)
    assert abs(result["b"] - 2 * result["a"]) <= 1
    assert sum(result.values()) <= 300


def test_existing_labels_reduce_allocation() -> None:
    """Words with many existing labels get fewer new instances."""
    result = allocate_instances(
        effective_senses={"a": 1, "b": 1},
        existing_labeled={"a": 0, "b": 100},
        good_labeled={"a": 0, "b": 100},
        corpus_counts={"a": 1_000_000, "b": 1_000_000},
        budget=100,
    )
    assert result.get("a", 0) > result.get("b", 0)


def test_cap_by_available() -> None:
    """Forms with few unlabeled instances are capped; remaining budget goes
    elsewhere."""
    result = allocate_instances(
        effective_senses={"scarce": 1, "ample": 1},
        existing_labeled={"scarce": 990, "ample": 0},
        good_labeled={"scarce": 990, "ample": 0},
        corpus_counts={"scarce": 1_000, "ample": 1_000_000},
        budget=500,
    )
    assert result.get("scarce", 0) <= 10
    assert result.get("ample", 0) > result.get("scarce", 0)
    assert sum(result.values()) <= 500


def test_budget_met_when_sufficient_corpus() -> None:
    """Sum of allocations is close to budget (may be slightly under due to int
    truncation)."""
    budget = 600
    result = allocate_instances(
        effective_senses={"a": 1, "b": 2, "c": 3},
        existing_labeled={},
        good_labeled={},
        corpus_counts={"a": 1_000_000, "b": 1_000_000, "c": 1_000_000},
        budget=budget,
    )
    total = sum(result.values())
    assert total <= budget
    # At most 1 lost per form due to int truncation
    assert total >= budget - len(result)


def test_insufficient_corpus_returns_all_available() -> None:
    """When total available < budget, return all available instances."""
    result = allocate_instances(
        effective_senses={"a": 1, "b": 1},
        existing_labeled={"a": 90, "b": 90},
        good_labeled={"a": 90, "b": 90},
        corpus_counts={"a": 100, "b": 100},
        budget=10_000,
    )
    assert result.get("a", 0) == 10
    assert result.get("b", 0) == 10


def test_zero_effective_senses_excluded() -> None:
    """Forms with 0 effective senses receive no allocation."""
    result = allocate_instances(
        effective_senses={"a": 0, "b": 1},
        existing_labeled={},
        good_labeled={},
        corpus_counts={"a": 1_000_000, "b": 1_000_000},
        budget=100,
    )
    assert "a" not in result
    assert "b" in result


def test_below_min_count_excluded() -> None:
    """Forms with fewer than min_count corpus occurrences are excluded."""
    result = allocate_instances(
        effective_senses={"rare": 1, "common": 1},
        existing_labeled={},
        good_labeled={},
        corpus_counts={"rare": 3, "common": 1_000_000},
        budget=100,
        min_count=5,
    )
    assert "rare" not in result
    assert "common" in result


def test_fully_saturated_form_excluded() -> None:
    """Forms with available=0 (all corpus instances well-labeled) get no allocation."""
    result = allocate_instances(
        effective_senses={"full": 1, "empty": 1},
        existing_labeled={"full": 1_000_000},
        good_labeled={"full": 1_000_000},
        corpus_counts={"full": 1_000_000, "empty": 1_000_000},
        budget=100,
    )
    assert "full" not in result
    assert "empty" in result


def test_bad_labels_do_not_block_resampling() -> None:
    """Instances with rating 0 (bad) are available for resampling (not in
    good_labeled)."""
    # form has 50 total labels but 50 are bad (rating 0) so good_labeled=0
    result = allocate_instances(
        effective_senses={"noisy": 1},
        existing_labeled={"noisy": 50},
        good_labeled={"noisy": 0},
        corpus_counts={"noisy": 1_000_000},
        budget=200,
    )
    # available = 1_000_000 - 0 = 1_000_000; need = max(0, k*1 - 50)
    # With enough budget and corpus, noisy should get a positive allocation
    assert result.get("noisy", 0) > 0


def test_empty_inputs_returns_empty() -> None:
    """No eligible forms → empty allocation."""
    result = allocate_instances(
        effective_senses={},
        existing_labeled={},
        good_labeled={},
        corpus_counts={},
        budget=100,
    )
    assert result == {}
