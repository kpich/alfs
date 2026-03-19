import pytest

from alfs.etl.ngram_cache import NgramCache

# A sentence long enough to have 8-grams (needs >= 8 words)
SHORT = "the quick brown fox jumps over the lazy dog"
# Near-identical to SHORT — shares most 8-grams
NEAR_DUP = "the quick brown fox jumps over the lazy cat"
# Completely unrelated text
UNRELATED = "sodium chloride dissolves in water to form a saline solution"


def _make_long(base: str, n: int = 20) -> str:
    """Repeat a sentence to create a text with many 8-grams."""
    words = base.split() * n
    return " ".join(words)


def test_empty_cache_never_flags_duplicate():
    cache = NgramCache()
    assert not cache.is_near_duplicate(SHORT)


def test_exact_text_is_near_duplicate_after_add():
    cache = NgramCache()
    text = _make_long(SHORT)
    cache.add_doc(text)
    assert cache.is_near_duplicate(text)


def test_near_duplicate_detected():
    cache = NgramCache()
    text = _make_long(SHORT, n=30)
    # Build a variant that differs only in the last word of every copy
    variant = _make_long(NEAR_DUP, n=30)
    cache.add_doc(text)
    assert cache.is_near_duplicate(variant)


def test_unrelated_text_not_flagged():
    cache = NgramCache()
    cache.add_doc(_make_long(SHORT, n=30))
    assert not cache.is_near_duplicate(_make_long(UNRELATED, n=30))


def test_short_text_below_8_words_never_flagged():
    cache = NgramCache()
    short = "only seven words here so no grams"  # 7 words → 0 8-grams
    cache.add_doc(_make_long(SHORT, n=30))
    assert not cache.is_near_duplicate(short)


def test_add_docs_bulk_same_as_add_doc_individually():
    text1 = _make_long(SHORT, n=20)
    text2 = _make_long(UNRELATED, n=20)

    c1 = NgramCache()
    c1.add_doc(text1)
    c1.add_doc(text2)

    c2 = NgramCache()
    c2.add_docs([text1, text2])

    assert c1._hashes == c2._hashes


def test_hash_gram_is_deterministic():
    cache = NgramCache()
    gram = "the quick brown fox jumps over the lazy"
    assert cache._hash_gram(gram) == cache._hash_gram(gram)


def test_word_8grams_length():
    cache = NgramCache()
    words = list(range(10))
    text = " ".join(str(w) for w in words)
    grams = cache._word_8grams(text)
    assert len(grams) == 3  # 10 - 8 + 1 = 3


def test_word_8grams_empty_and_short():
    cache = NgramCache()
    assert cache._word_8grams("") == []
    assert cache._word_8grams("only seven words here so no grams") == []


def test_save_and_load_roundtrip(tmp_path):
    cache = NgramCache()
    cache.add_doc(_make_long(SHORT, n=20))
    original_hashes = set(cache._hashes)

    path = tmp_path / "cache.npy"
    cache.save(path)

    loaded = NgramCache.load(path)
    assert loaded._hashes == original_hashes


def test_loaded_cache_detects_duplicate(tmp_path):
    text = _make_long(SHORT, n=30)
    cache = NgramCache()
    cache.add_doc(text)
    path = tmp_path / "cache.npy"
    cache.save(path)

    loaded = NgramCache.load(path)
    assert loaded.is_near_duplicate(text)
    assert not loaded.is_near_duplicate(_make_long(UNRELATED, n=30))


def test_threshold_respected():
    """A text that shares exactly 0 grams with cache should not be flagged."""
    cache = NgramCache()
    cache.add_doc(_make_long(SHORT, n=30))
    # Nothing from UNRELATED will be in the cache
    # threshold=0.0 means any hit counts — still False if truly no overlap
    # We can't guarantee zero overlap with a hash, but with distinct vocabularies
    # real hit rate should be well below 5%
    assert not cache.is_near_duplicate(_make_long(UNRELATED, n=30))


@pytest.mark.parametrize("n_docs", [1, 5, 50])
def test_cache_grows_with_docs(n_docs):
    cache = NgramCache()
    for i in range(n_docs):
        cache.add_doc(
            _make_long(
                f"unique word set number {i} alpha beta gamma delta epsilon zeta", n=5
            )
        )
    # Cache should be non-empty after adding docs with enough words
    assert len(cache._hashes) > 0
