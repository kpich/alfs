from alfs.data_models.doc import Doc
from alfs.etl.dedup import exact_dedup, ngram_dedup
from alfs.etl.ngram_cache import NgramCache


def _make_doc(doc_id: str, text: str) -> Doc:
    return Doc(doc_id=doc_id, text=text)


def test_exact_dedup_filters_known_id() -> None:
    doc = _make_doc("known", "some text")
    existing_ids: set[str] = {"known"}
    result = exact_dedup([doc], existing_ids)
    assert result == []


def test_exact_dedup_passes_new_id() -> None:
    doc = _make_doc("new", "some text")
    existing_ids: set[str] = set()
    result = exact_dedup([doc], existing_ids)
    assert result == [doc]
    assert "new" in existing_ids


def test_exact_dedup_updates_existing_ids_to_prevent_duplicates_within_batch() -> None:
    doc1 = _make_doc("dup", "text one")
    doc2 = _make_doc("dup", "text two")
    existing_ids: set[str] = set()
    result = exact_dedup([doc1, doc2], existing_ids)
    assert len(result) == 1
    assert result[0] is doc1


def test_ngram_dedup_filters_near_duplicate() -> None:
    # Build a long-enough text (8+ words) so ngrams are generated
    base_text = "the quick brown fox jumps over the lazy dog and more words here"
    cache = NgramCache()
    cache.add_doc(base_text)
    # Near-duplicate: same text with minor change at end
    near_dup = _make_doc("nd", base_text + " extra")
    result = ngram_dedup([near_dup], cache)
    assert result == []


def test_ngram_dedup_updates_cache() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    cache = NgramCache()
    doc = _make_doc("d1", text)
    ngram_dedup([doc], cache)
    # After adding, a near-dup should be caught
    near_dup = _make_doc("d2", text + " lambda")
    result = ngram_dedup([near_dup], cache)
    assert result == []
