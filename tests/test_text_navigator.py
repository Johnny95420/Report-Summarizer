"""Tests for Tools.text_navigator — AgentDocumentReader core logic.

Tests cover navigation, bookmarks, dedup, error handling, and _make_result.
Heavy deps (Chroma, BM25, HuggingFaceEmbeddings) are mocked.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from Tools.reader_models import BaseReaderDocument, PDFReaderDocument
from Tools.text_navigator import AgentDocumentReader, Bookmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_base_doc(name="test-doc", num_pages=3) -> BaseReaderDocument:
    pages = [
        Document(page_content=f"Page {i} content for {name}", metadata={"page_id": i})
        for i in range(num_pages)
    ]
    return BaseReaderDocument(date="2026-01-01", name=name, outlines=[], pages=pages)


def _make_pdf_doc(name="pdf-doc", num_pages=3) -> PDFReaderDocument:
    pages = [
        Document(page_content=f"Page {i} content for {name}", metadata={"page_id": i})
        for i in range(num_pages)
    ]
    tables = [Document(page_content="table summary", metadata={"page_id": 0, "table": "<table>...</table>"})]
    return PDFReaderDocument(
        date="2026-01-01", name=name, outlines=[], pages=pages, highlights="key insight", tables=tables,
    )


def _write_doc_json(tmp_path, doc, filename="doc.json") -> str:
    path = os.path.join(str(tmp_path), filename)
    doc.save(path)
    return path


@pytest.fixture
def reader(tmp_path):
    """Return an AgentDocumentReader with a temp persist dir."""
    return AgentDocumentReader(persist_dir=str(tmp_path / "nav_store"))


@pytest.fixture
def mock_indexing():
    """Mock Chroma and BM25 so open_document doesn't need real embeddings."""
    fake_embeddings = MagicMock()
    with (
        patch("Tools.text_navigator.get_embedding_model", return_value=fake_embeddings),
        patch("Tools.text_navigator.Chroma") as mock_chroma,
        patch("Tools.text_navigator.BM25Retriever") as mock_bm25,
    ):
        mock_chroma.from_documents.return_value = MagicMock()
        mock_bm25.from_documents.return_value = MagicMock()
        yield mock_chroma, mock_bm25


# ---------------------------------------------------------------------------
# Navigation: open_document, go_to_page, next_page, prev_page, boundaries
# ---------------------------------------------------------------------------
class TestNavigation:
    def test_open_base_document(self, reader, tmp_path, mock_indexing):
        doc = _make_base_doc()
        path = _write_doc_json(tmp_path, doc)

        result = reader.open_document(path)
        assert "test-doc" in result
        assert "3 pages" in result
        assert reader._current_page == 0

    def test_open_pdf_document(self, reader, tmp_path, mock_indexing):
        doc = _make_pdf_doc()
        path = _write_doc_json(tmp_path, doc)

        result = reader.open_document(path)
        assert "pdf-doc" in result
        assert isinstance(reader._document, PDFReaderDocument)

    def test_go_to_page(self, reader, tmp_path, mock_indexing):
        doc = _make_base_doc()
        path = _write_doc_json(tmp_path, doc)
        reader.open_document(path)

        content = reader.go_to_page(2)
        assert "Page 2" in content
        assert reader._current_page == 2

    def test_next_page(self, reader, tmp_path, mock_indexing):
        doc = _make_base_doc()
        path = _write_doc_json(tmp_path, doc)
        reader.open_document(path)

        content = reader.next_page()
        assert "Page 1" in content
        assert reader._current_page == 1

    def test_prev_page(self, reader, tmp_path, mock_indexing):
        doc = _make_base_doc()
        path = _write_doc_json(tmp_path, doc)
        reader.open_document(path)
        reader.go_to_page(2)

        content = reader.prev_page()
        assert "Page 1" in content
        assert reader._current_page == 1

    def test_boundary_next_page_raises(self, reader, tmp_path, mock_indexing):
        doc = _make_base_doc(num_pages=2)
        path = _write_doc_json(tmp_path, doc)
        reader.open_document(path)
        reader.go_to_page(1)

        with pytest.raises(IndexError, match="out of range"):
            reader.next_page()

    def test_boundary_prev_page_raises(self, reader, tmp_path, mock_indexing):
        doc = _make_base_doc()
        path = _write_doc_json(tmp_path, doc)
        reader.open_document(path)

        with pytest.raises(IndexError, match="out of range"):
            reader.prev_page()


# ---------------------------------------------------------------------------
# Bookmarks
# ---------------------------------------------------------------------------
class TestBookmarks:
    def test_update_and_show_bookmarks(self, reader, tmp_path, mock_indexing):
        doc = _make_base_doc()
        path = _write_doc_json(tmp_path, doc)
        reader.open_document(path)
        reader.go_to_page(1)

        result = reader.update_bookmark("key_finding")
        assert "key_finding" in result
        assert reader._bookmarks["key_finding"] == Bookmark(path, "test-doc", 1)

        bookmarks = reader.show_bookmarks()
        assert bookmarks["key_finding"]["doc"] == "test-doc"
        assert bookmarks["key_finding"]["page_id"] == 1

    def test_show_bookmarks_empty(self, reader):
        assert reader.show_bookmarks() == {}

    def test_go_to_bookmark_same_doc(self, reader, tmp_path, mock_indexing):
        doc = _make_base_doc()
        path = _write_doc_json(tmp_path, doc)
        reader.open_document(path)
        reader.go_to_page(2)
        reader.update_bookmark("target")
        reader.go_to_page(0)

        content = reader.go_to_bookmark("target")
        assert "Page 2" in content
        assert reader._current_page == 2

    def test_go_to_bookmark_cross_document(self, reader, tmp_path, mock_indexing):
        doc_a = _make_base_doc(name="doc-A")
        doc_b = _make_base_doc(name="doc-B")
        path_a = _write_doc_json(tmp_path, doc_a, "a.json")
        path_b = _write_doc_json(tmp_path, doc_b, "b.json")

        reader.open_document(path_a)
        reader.go_to_page(1)
        reader.update_bookmark("from_a")

        reader.open_document(path_b)
        assert reader._current_path == path_b

        content = reader.go_to_bookmark("from_a")
        assert reader._current_path == path_a
        assert "Page 1" in content

    def test_go_to_bookmark_bad_label(self, reader, tmp_path, mock_indexing):
        doc = _make_base_doc()
        path = _write_doc_json(tmp_path, doc)
        reader.open_document(path)

        with pytest.raises(KeyError, match="not found"):
            reader.go_to_bookmark("nonexistent")

    def test_bookmarks_persist_after_document_switch(self, reader, tmp_path, mock_indexing):
        """Bookmarks must survive open_document() calls to a different path."""
        doc_a = _make_base_doc(name="doc-A")
        doc_b = _make_base_doc(name="doc-B")
        path_a = _write_doc_json(tmp_path, doc_a, "a.json")
        path_b = _write_doc_json(tmp_path, doc_b, "b.json")

        reader.open_document(path_a)
        reader.go_to_page(1)
        reader.update_bookmark("section_x")

        reader.open_document(path_b)

        assert "section_x" in reader._bookmarks
        assert reader._bookmarks["section_x"].doc_name == "doc-A"


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------
class TestDedup:
    def test_open_same_path_returns_cached(self, reader, tmp_path, mock_indexing):
        doc = _make_base_doc()
        path = _write_doc_json(tmp_path, doc)

        reader.open_document(path)
        reader.go_to_page(2)

        result = reader.open_document(path)
        assert "[Already open:" in result
        assert "page 2" in result
        assert reader._current_page == 2  # cursor not reset


# ---------------------------------------------------------------------------
# Error: open_document with nonexistent file
# ---------------------------------------------------------------------------
class TestErrors:
    def test_open_nonexistent_file(self, reader):
        with pytest.raises(FileNotFoundError):
            reader.open_document("/nonexistent/path.json")

    def test_open_document_state_reset_on_failure(self, reader, tmp_path, mock_indexing):
        """If indexing fails, state should be reset, not half-committed."""
        doc = _make_base_doc()
        path = _write_doc_json(tmp_path, doc)

        mock_chroma, _ = mock_indexing
        mock_chroma.from_documents.side_effect = RuntimeError("embedding failure")

        with pytest.raises(RuntimeError, match="Failed to open"):
            reader.open_document(path)

        assert reader._document is None
        assert reader._vectorstore is None
        assert reader._current_path is None

    def test_open_document_state_cleared_when_switching_to_bad_doc(self, reader, tmp_path, mock_indexing):
        """A previously-open good doc must not survive a failed switch to a bad doc (C3)."""
        doc = _make_base_doc()
        path_good = _write_doc_json(tmp_path, doc, "good.json")
        reader.open_document(path_good)
        assert reader._document is not None

        mock_chroma, _ = mock_indexing
        mock_chroma.from_documents.side_effect = RuntimeError("fail")
        doc2 = _make_base_doc(name="doc2")
        path_bad = _write_doc_json(tmp_path, doc2, "bad.json")

        with pytest.raises(RuntimeError):
            reader.open_document(path_bad)

        assert reader._document is None
        assert reader._current_path is None


# ---------------------------------------------------------------------------
# _make_result
# ---------------------------------------------------------------------------
class TestMakeResult:
    def test_valid_page_id(self, reader, tmp_path, mock_indexing):
        doc = _make_base_doc()
        path = _write_doc_json(tmp_path, doc)
        reader.open_document(path)

        result_doc = Document(page_content="match", metadata={"page_id": 1})
        sr = reader._make_result(result_doc, score=0.85)
        assert sr.page_id == 1
        assert sr.score == 0.85
        assert "Page 1" in sr.page_preview

    def test_missing_page_id_defaults_to_zero(self, reader, tmp_path, mock_indexing):
        doc = _make_base_doc()
        path = _write_doc_json(tmp_path, doc)
        reader.open_document(path)

        result_doc = Document(page_content="match", metadata={})
        sr = reader._make_result(result_doc, score=None)
        assert sr.page_id == 0

    def test_out_of_bounds_page_id(self, reader, tmp_path, mock_indexing):
        doc = _make_base_doc(num_pages=2)
        path = _write_doc_json(tmp_path, doc)
        reader.open_document(path)

        result_doc = Document(page_content="match", metadata={"page_id": 99})
        sr = reader._make_result(result_doc, score=0.5)
        assert sr.page_id == 99
        assert "error" in sr.page_preview
