"""Unit tests for groq_batch_prepare.allocate_instances and effective_sense_count."""

import polars as pl

from alfs.data_models.alf import Alf, Sense
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.sense_store import SenseStore
from alfs.update.labeling.groq_batch_prepare import (
    allocate_instances,
    effective_sense_count,
    run,
    split_labeled_pairs,
)


def _store(tmp_path, *entries: Alf) -> SenseStore:
    store = SenseStore(tmp_path / "senses.db")
    for entry in entries:
        store.write(entry)
    return store


def test_effective_sense_count_case_variant_sees_lowercase_senses(tmp_path):
    canonical = Alf(
        form="run",
        senses=[Sense(definition="to move quickly"), Sense(definition="to operate")],
    )
    alias = Alf(form="Run", senses=[])
    store = _store(tmp_path, canonical, alias)
    assert effective_sense_count(alias, store) == 2


def test_effective_sense_count_no_senses_returns_zero(tmp_path):
    alias = Alf(form="Run", senses=[])
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


def _make_labeled_df(
    rows: list[tuple[str, str, int, str | None]],
) -> pl.DataFrame:
    """Build a minimal good_labeled_df with columns form, doc_id, byte_offset,
    updated_at."""
    return pl.DataFrame(
        {
            "form": [r[0] for r in rows],
            "doc_id": [r[1] for r in rows],
            "byte_offset": [r[2] for r in rows],
            "updated_at": [r[3] for r in rows],
        }
    )


def test_split_labeled_pairs_stale_goes_to_stale() -> None:
    """Occurrence labeled before a new sense is added → stale."""
    df = _make_labeled_df([("run", "doc1", 100, "2024-01-01 00:00:00")])
    max_sense_ts = {"run": "2024-06-01 00:00:00"}
    good, stale = split_labeled_pairs(df, max_sense_ts)
    assert ("doc1", 100) not in good["run"]
    assert ("doc1", 100) in stale["run"]


def test_split_labeled_pairs_fresh_goes_to_good() -> None:
    """Occurrence labeled after the latest sense → good (excluded from re-sampling)."""
    df = _make_labeled_df([("run", "doc1", 100, "2024-09-01 00:00:00")])
    max_sense_ts = {"run": "2024-06-01 00:00:00"}
    good, stale = split_labeled_pairs(df, max_sense_ts)
    assert ("doc1", 100) in good["run"]
    assert ("doc1", 100) not in stale.get("run", [])


def test_split_labeled_pairs_no_sense_ts_goes_to_good() -> None:
    """Form with no sense timestamp → occurrence treated as non-stale (good)."""
    df = _make_labeled_df([("run", "doc1", 100, "2024-01-01 00:00:00")])
    good, stale = split_labeled_pairs(df, max_sense_ts={})
    assert ("doc1", 100) in good["run"]
    assert len(stale.get("run", [])) == 0


def test_split_labeled_pairs_null_labeled_ts_goes_to_good() -> None:
    """Occurrence with NULL updated_at → treated as non-stale (good)."""
    df = _make_labeled_df([("run", "doc1", 100, None)])
    max_sense_ts = {"run": "2024-06-01 00:00:00"}
    good, stale = split_labeled_pairs(df, max_sense_ts)
    assert ("doc1", 100) in good["run"]
    assert len(stale.get("run", [])) == 0


def test_run_case_variants_share_lowercase_pool(tmp_path) -> None:
    """Case variants share the same lowercase occurrence pool; budget is not doubled.

    "DOG" and "dog" are separate entries but look up the same lowercase parquet rows.
    The seen_lower dedup ensures "dog" occurrences are only sampled once.
    """
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

    # 40 "dog" occurrences (parquets are lowercase-normalized after aggregation)
    n = 40
    rows = [{"form": "dog", "doc_id": f"doc{i}", "byte_offset": 0} for i in range(n)]
    occ_df = pl.DataFrame(rows)
    prefix_dir = tmp_path / "by_prefix" / "d"
    prefix_dir.mkdir(parents=True)
    occ_df.write_parquet(str(prefix_dir / "occurrences.parquet"))

    # docs.parquet with enough text for context extraction
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

    # Should be close to budget (≤ budget + 1 for int truncation), not doubled
    assert total <= budget + 1


def test_split_labeled_pairs_mixed() -> None:
    """Mix of stale and fresh occurrences for the same form."""
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
