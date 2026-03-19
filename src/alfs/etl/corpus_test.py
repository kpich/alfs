from pathlib import Path

from alfs.data_models.doc import Doc
from alfs.etl.corpus import append_docs, get_doc_ids


def _make_doc(doc_id: str, text: str) -> Doc:
    return Doc(doc_id=doc_id, text=text)


def test_append_docs_creates_corpus_if_not_exists(tmp_path: Path) -> None:
    corpus = tmp_path / "docs.parquet"
    doc = _make_doc("abc123", "hello world")
    append_docs([doc], corpus)
    assert corpus.exists()
    assert get_doc_ids(corpus) == {"abc123"}


def test_append_docs_merges_with_existing(tmp_path: Path) -> None:
    corpus = tmp_path / "docs.parquet"
    doc1 = _make_doc("aaa", "first doc")
    doc2 = _make_doc("bbb", "second doc")
    append_docs([doc1], corpus)
    append_docs([doc2], corpus)
    assert get_doc_ids(corpus) == {"aaa", "bbb"}


def test_get_doc_ids_returns_correct_set(tmp_path: Path) -> None:
    corpus = tmp_path / "docs.parquet"
    docs = [_make_doc(f"id{i}", f"text {i}") for i in range(3)]
    append_docs(docs, corpus)
    assert get_doc_ids(corpus) == {"id0", "id1", "id2"}
