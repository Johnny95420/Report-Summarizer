"""Tests for Tools.text_navigator — AgentDocumentReader core logic.

Tests cover navigation, bookmarks, dedup, error handling, and _make_result.
Heavy deps (Chroma, BM25, HuggingFaceEmbeddings) are mocked.
"""

import os
import threading
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from Tools.reader_models import BaseReaderDocument, PDFReaderDocument
from Tools.text_navigator import _DEFAULT_PERSIST_DIR, AgentDocumentReader, Bookmark


# ---------------------------------------------------------------------------
# Config loading: _DEFAULT_PERSIST_DIR reads from retriever_config.yaml
# ---------------------------------------------------------------------------
class TestConfigLoading:
    def test_default_persist_dir_is_string(self):
        """_DEFAULT_PERSIST_DIR must be a non-empty string."""
        assert isinstance(_DEFAULT_PERSIST_DIR, str)
        assert _DEFAULT_PERSIST_DIR

    def test_default_persist_dir_contains_navigator_tmp(self):
        """_DEFAULT_PERSIST_DIR should contain the config value 'navigator_tmp' (from FAKE_CONFIG)."""
        assert "navigator_tmp" in _DEFAULT_PERSIST_DIR

    def test_default_persist_dir_is_absolute_path(self):
        """_DEFAULT_PERSIST_DIR should be an absolute path (anchored to project root)."""
        import os

        assert os.path.isabs(_DEFAULT_PERSIST_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_base_doc(name="test-doc", num_pages=3) -> BaseReaderDocument:
    pages = [Document(page_content=f"Page {i} content for {name}", metadata={"page_id": i}) for i in range(num_pages)]
    return BaseReaderDocument(date="2026-01-01", name=name, outlines=[], pages=pages)


def _make_pdf_doc(name="pdf-doc", num_pages=3) -> PDFReaderDocument:
    pages = [Document(page_content=f"Page {i} content for {name}", metadata={"page_id": i}) for i in range(num_pages)]
    tables = [Document(page_content="table summary", metadata={"page_id": 0, "table": "<table>...</table>"})]
    return PDFReaderDocument(
        date="2026-01-01",
        name=name,
        outlines=[],
        pages=pages,
        highlights="key insight",
        tables=tables,
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
        with pytest.raises(RuntimeError, match="Failed to open"):
            reader.open_document("/nonexistent/path.json")

    def test_open_document_logs_error_on_failure(self, reader, caplog):
        """open_document logs logger.error with [open_document] prefix when file is missing."""
        import logging

        with caplog.at_level(logging.ERROR, logger="TextNavigator"):
            with pytest.raises(RuntimeError):
                reader.open_document("/nonexistent/path.json")
        assert "[open_document] Failed to open" in caplog.text

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


# ---------------------------------------------------------------------------
# Chroma cache validation (C1)
# ---------------------------------------------------------------------------
class TestChromaCacheValidation:
    def test_stale_cache_triggers_rebuild(self, reader, tmp_path):
        """When Chroma cache count != page count, cache should be rebuilt (C1 fix)."""
        doc = _make_base_doc(num_pages=5)
        path = _write_doc_json(tmp_path, doc)

        fake_embeddings = MagicMock()
        fake_collection = MagicMock()
        fake_collection.count.return_value = 3  # stale: 3 != 5

        fake_vectorstore = MagicMock()
        fake_vectorstore._collection = fake_collection

        with (
            patch("Tools.text_navigator.get_embedding_model", return_value=fake_embeddings),
            patch("Tools.text_navigator.Chroma") as mock_chroma,
            patch("Tools.text_navigator.BM25Retriever") as mock_bm25,
            patch("os.path.exists", side_effect=lambda p: p == path or "nav_store" in str(p)),
            patch("shutil.rmtree") as mock_rmtree,
            patch("chromadb.api.client.SharedSystemClient.clear_system_cache") as mock_clear,
        ):
            mock_chroma.return_value = fake_vectorstore
            mock_chroma.from_documents.return_value = MagicMock()
            mock_bm25.from_documents.return_value = MagicMock()

            reader.open_document(path)

            # Should have rebuilt: clear_system_cache → rmtree → from_documents
            mock_clear.assert_called_once()
            mock_rmtree.assert_called_once()
            mock_chroma.from_documents.assert_called_once()

    def test_stale_cache_clears_system_cache_before_rmtree(self, reader, tmp_path):
        """clear_system_cache must be called before shutil.rmtree on stale rebuild."""
        doc = _make_base_doc(num_pages=5)
        path = _write_doc_json(tmp_path, doc)

        fake_embeddings = MagicMock()
        fake_collection = MagicMock()
        fake_collection.count.return_value = 3  # stale

        fake_vectorstore = MagicMock()
        fake_vectorstore._collection = fake_collection

        call_order = []

        with (
            patch("Tools.text_navigator.get_embedding_model", return_value=fake_embeddings),
            patch("Tools.text_navigator.Chroma") as mock_chroma,
            patch("Tools.text_navigator.BM25Retriever") as mock_bm25,
            patch("os.path.exists", side_effect=lambda p: p == path or "nav_store" in str(p)),
            patch("shutil.rmtree", side_effect=lambda *a, **kw: call_order.append("rmtree")),
            patch(
                "chromadb.api.client.SharedSystemClient.clear_system_cache",
                side_effect=lambda: call_order.append("clear"),
            ),
        ):
            mock_chroma.return_value = fake_vectorstore
            mock_chroma.from_documents.return_value = MagicMock()
            mock_bm25.from_documents.return_value = MagicMock()

            reader.open_document(path)

        assert call_order == ["clear", "rmtree"], f"Expected clear_system_cache before rmtree, got: {call_order}"

    def test_valid_cache_not_rebuilt(self, reader, tmp_path):
        """When Chroma cache count matches page count, no rebuild needed."""
        doc = _make_base_doc(num_pages=3)
        path = _write_doc_json(tmp_path, doc)

        fake_embeddings = MagicMock()
        fake_collection = MagicMock()
        fake_collection.count.return_value = 3  # matches

        fake_vectorstore = MagicMock()
        fake_vectorstore._collection = fake_collection

        with (
            patch("Tools.text_navigator.get_embedding_model", return_value=fake_embeddings),
            patch("Tools.text_navigator.Chroma") as mock_chroma,
            patch("Tools.text_navigator.BM25Retriever") as mock_bm25,
            patch("os.path.exists", side_effect=lambda p: p == path or "nav_store" in str(p)),
        ):
            mock_chroma.return_value = fake_vectorstore
            mock_bm25.from_documents.return_value = MagicMock()

            reader.open_document(path)

            # Should NOT have rebuilt
            mock_chroma.from_documents.assert_not_called()


# ---------------------------------------------------------------------------
# Chroma cleanup on document switch (C7)
# ---------------------------------------------------------------------------
class TestChromaCleanup:
    def test_vectorstore_reference_released_on_switch(self, reader, tmp_path, mock_indexing):
        """Old vectorstore reference is released (set to None) when switching documents.
        SQLite cache on disk is preserved — delete_collection() is NOT called."""
        doc_a = _make_base_doc(name="doc-A")
        doc_b = _make_base_doc(name="doc-B")
        path_a = _write_doc_json(tmp_path, doc_a, "a.json")
        path_b = _write_doc_json(tmp_path, doc_b, "b.json")

        reader.open_document(path_a)
        old_vectorstore = reader._vectorstore

        reader.open_document(path_b)
        # delete_collection must NOT be called — preserves disk cache
        old_vectorstore.delete_collection.assert_not_called()
        assert reader._vectorstore is not None  # new vectorstore assigned

    def test_close_document_releases_vectorstore(self, reader, tmp_path, mock_indexing):
        """close_document sets _vectorstore to None without calling delete_collection."""
        doc = _make_base_doc()
        path = _write_doc_json(tmp_path, doc)
        reader.open_document(path)
        vectorstore = reader._vectorstore

        reader.close_document()
        vectorstore.delete_collection.assert_not_called()
        assert reader._vectorstore is None

    def test_close_document_preserves_bookmarks(self, reader, tmp_path, mock_indexing):
        """close_document preserves bookmarks — they survive for the instance lifetime."""
        doc = _make_base_doc()
        path = _write_doc_json(tmp_path, doc)
        reader.open_document(path)
        reader.update_bookmark("chap1")
        assert reader._bookmarks

        reader.close_document()
        assert "chap1" in reader._bookmarks


# ---------------------------------------------------------------------------
# get_embedding_model error handling (C9)
# ---------------------------------------------------------------------------
class TestGetEmbeddingModelErrors:
    def test_embedding_model_load_failure(self):
        """Embedding model load failure should raise RuntimeError (C9 fix)."""
        import Tools.text_navigator as tn

        # Reset singleton
        old_state = tn._embedding_state
        tn._embedding_state = None

        try:
            with (
                patch("Tools.text_navigator.HuggingFaceEmbeddings", side_effect=OSError("disk full")),
                pytest.raises(RuntimeError, match="Failed to load embedding model"),
            ):
                tn.get_embedding_model("bad-model")
        finally:
            tn._embedding_state = old_state

    def test_concurrent_requests_return_matching_model_name(self):
        """Concurrent callers must never receive an embedding for the wrong model name."""
        import Tools.text_navigator as tn

        old_state = tn._embedding_state
        tn._embedding_state = None
        mismatches: list[tuple[str, str]] = []
        names = ("model-a", "model-b")
        start = threading.Barrier(8)

        class _FakeEmb:
            def __init__(self, model_name: str):
                self.model_name = model_name

        def _worker(worker_id: int):
            start.wait()
            for i in range(200):
                requested = names[(worker_id + i) % 2]
                emb = tn.get_embedding_model(requested)
                if emb.model_name != requested:
                    mismatches.append((requested, emb.model_name))

        try:
            with patch("Tools.text_navigator.HuggingFaceEmbeddings", side_effect=_FakeEmb):
                threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
        finally:
            tn._embedding_state = old_state

        assert mismatches == []

    def test_cache_hit_skips_initialization(self):
        """Fast path must NOT call HuggingFaceEmbeddings() again on cache hit."""
        import Tools.text_navigator as tn

        old_state = tn._embedding_state
        tn._embedding_state = None
        try:
            with patch("Tools.text_navigator.HuggingFaceEmbeddings") as mock_emb:
                mock_emb.return_value = MagicMock(model_name="shared-model")
                emb1 = tn.get_embedding_model("shared-model")
                assert mock_emb.call_count == 1
                emb2 = tn.get_embedding_model("shared-model")
                assert mock_emb.call_count == 1, "Cache hit must not reinitialize"
                assert emb1 is emb2
        finally:
            tn._embedding_state = old_state

    def test_model_name_mismatch_forces_rebuild(self):
        """Requesting a different model_name must build a new HuggingFaceEmbeddings instance."""
        import Tools.text_navigator as tn

        old_state = tn._embedding_state
        tn._embedding_state = None
        try:
            class _FakeEmb:
                def __init__(self, model_name: str):
                    self.model_name = model_name

            with patch("Tools.text_navigator.HuggingFaceEmbeddings", side_effect=_FakeEmb) as mock_emb:
                emb1 = tn.get_embedding_model("model-a")
                assert emb1.model_name == "model-a"
                assert mock_emb.call_count == 1
                emb2 = tn.get_embedding_model("model-b")
                assert emb2.model_name == "model-b"
                assert mock_emb.call_count == 2, "Different model name must trigger rebuild"
                assert emb1 is not emb2
        finally:
            tn._embedding_state = old_state


# ---------------------------------------------------------------------------
# Path validation (C5)
# ---------------------------------------------------------------------------
class TestPathValidation:
    def test_rejects_non_json_path(self, reader):
        """open_document should reject paths not ending in .json (C5 fix)."""
        with pytest.raises(ValueError, match=r"\.json"):
            reader.open_document("/etc/passwd")

    def test_rejects_txt_path(self, reader):
        """open_document should reject .txt paths (C5 fix)."""
        with pytest.raises(ValueError, match=r"\.json"):
            reader.open_document("/tmp/doc.txt")

    def test_rejects_path_without_extension(self, reader):
        """open_document should reject paths with no extension (C5 fix)."""
        with pytest.raises(ValueError, match=r"\.json"):
            reader.open_document("/tmp/doc")

    def test_accepts_json_path(self, reader, tmp_path, mock_indexing):
        """open_document should accept .json paths (C5 fix)."""
        doc = _make_base_doc()
        path = _write_doc_json(tmp_path, doc)
        result = reader.open_document(path)
        assert "test-doc" in result


# ---------------------------------------------------------------------------
# Chroma _collection.count() failure triggers rebuild (C7)
# ---------------------------------------------------------------------------
class TestChromaCountFailure:
    def test_attribute_error_triggers_rebuild(self, reader, tmp_path):
        """When _collection.count() raises AttributeError, should rebuild (C7 fix)."""
        doc = _make_base_doc(num_pages=3)
        path = _write_doc_json(tmp_path, doc)

        fake_embeddings = MagicMock()
        fake_collection = MagicMock()
        fake_collection.count.side_effect = AttributeError("no _collection attr")

        fake_vectorstore = MagicMock()
        fake_vectorstore._collection = fake_collection

        with (
            patch("Tools.text_navigator.get_embedding_model", return_value=fake_embeddings),
            patch("Tools.text_navigator.Chroma") as mock_chroma,
            patch("Tools.text_navigator.BM25Retriever") as mock_bm25,
            patch("os.path.exists", side_effect=lambda p: p == path or "nav_store" in str(p)),
            patch("shutil.rmtree") as mock_rmtree,
        ):
            mock_chroma.return_value = fake_vectorstore
            mock_chroma.from_documents.return_value = MagicMock()
            mock_bm25.from_documents.return_value = MagicMock()

            reader.open_document(path)

            mock_rmtree.assert_called_once()
            mock_chroma.from_documents.assert_called_once()

    def test_generic_exception_triggers_rebuild(self, reader, tmp_path):
        """When _collection.count() raises any exception, should rebuild (C7 fix)."""
        doc = _make_base_doc(num_pages=3)
        path = _write_doc_json(tmp_path, doc)

        fake_embeddings = MagicMock()
        fake_collection = MagicMock()
        fake_collection.count.side_effect = RuntimeError("Chroma internal error")

        fake_vectorstore = MagicMock()
        fake_vectorstore._collection = fake_collection

        with (
            patch("Tools.text_navigator.get_embedding_model", return_value=fake_embeddings),
            patch("Tools.text_navigator.Chroma") as mock_chroma,
            patch("Tools.text_navigator.BM25Retriever") as mock_bm25,
            patch("os.path.exists", side_effect=lambda p: p == path or "nav_store" in str(p)),
            patch("shutil.rmtree") as mock_rmtree,
        ):
            mock_chroma.return_value = fake_vectorstore
            mock_chroma.from_documents.return_value = MagicMock()
            mock_bm25.from_documents.return_value = MagicMock()

            reader.open_document(path)

            mock_rmtree.assert_called_once()
            mock_chroma.from_documents.assert_called_once()


# ---------------------------------------------------------------------------
# char_count in get_metadata (I5)
# ---------------------------------------------------------------------------
class TestCharCount:
    def test_get_metadata_has_char_count(self, reader, tmp_path, mock_indexing):
        """get_metadata should return char_count not word_count (I5 fix)."""
        doc = _make_base_doc()
        path = _write_doc_json(tmp_path, doc)
        reader.open_document(path)

        metadata = reader.get_metadata()
        assert "char_count" in metadata
        assert "word_count" not in metadata
        expected = sum(len(p.page_content) for p in doc.pages)
        assert metadata["char_count"] == expected

    def test_char_count_for_cjk_content(self, reader, tmp_path, mock_indexing):
        """char_count should correctly count CJK characters (I5 fix)."""
        pages = [
            Document(page_content="台積電2024年資本支出", metadata={"page_id": 0}),
            Document(page_content="預計約300億美元", metadata={"page_id": 1}),
        ]
        doc = MagicMock()
        doc.date = "2024-01-01"
        doc.name = "test"
        doc.outlines = []
        doc.pages = pages

        # Inject mocked document directly
        reader._document = doc
        reader._current_path = "/fake/path.json"

        metadata = reader.get_metadata()
        assert metadata["char_count"] == len("台積電2024年資本支出") + len("預計約300億美元")
