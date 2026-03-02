import alfs.seg.segment_docs as sd
from alfs.seg.segment_docs import iter_chunks


def test_iter_chunks_short_text(monkeypatch):
    monkeypatch.setattr(sd, "CHUNK_SIZE", 20)
    text = "hello"
    result = list(iter_chunks(text))
    assert result == [("hello", 0)]


def test_iter_chunks_splits_at_whitespace(monkeypatch):
    monkeypatch.setattr(sd, "CHUNK_SIZE", 20)
    # 22 chars; spaces at indices 5 and 11; last space before pos 20 is at 11
    # so split happens after that space: first chunk ends at index 12
    text = "hello world foobar!!xy"
    result = list(iter_chunks(text))
    assert result[0] == ("hello world ", 0)
    assert result[1] == ("foobar!!xy", 12)


def test_iter_chunks_no_whitespace_fallback(monkeypatch):
    monkeypatch.setattr(sd, "CHUNK_SIZE", 20)
    text = "x" * 25
    result = list(iter_chunks(text))
    assert result[0] == ("x" * 20, 0)
    assert result[1] == ("x" * 5, 20)


def test_iter_chunks_empty(monkeypatch):
    monkeypatch.setattr(sd, "CHUNK_SIZE", 20)
    assert list(iter_chunks("")) == []


def test_iter_chunks_reassembly(monkeypatch):
    monkeypatch.setattr(sd, "CHUNK_SIZE", 20)
    text = "hello world foobar!!xy baz qux more text here and beyond"
    reassembled = "".join(chunk for chunk, _ in iter_chunks(text))
    assert reassembled == text


def test_iter_chunks_start_positions(monkeypatch):
    monkeypatch.setattr(sd, "CHUNK_SIZE", 20)
    text = "hello world foobar!!xy baz"
    result = list(iter_chunks(text))
    assert len(result) >= 2
    first_chunk, first_start = result[0]
    _, second_start = result[1]
    assert second_start == first_start + len(first_chunk)
