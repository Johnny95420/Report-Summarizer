"""Integration tests for AgentDocumentReader Chroma SQLite cache behaviour.

These tests use a real Chroma SQLite backend (no mock) with a lightweight
DeterministicFakeEmbedding to avoid loading a GPU model.

Markers
-------
pytest.mark.integration — run alongside the unit suite; no external services needed.
"""

import json
import os
from unittest.mock import patch

import pytest
from chromadb.api.client import SharedSystemClient
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings.fake import DeterministicFakeEmbedding

import Tools.text_navigator as tn
from Tools.reader_models import sanitize_name
from Tools.text_navigator import AgentDocumentReader


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: Chroma SQLite cache integration tests")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_FAKE_EMB = DeterministicFakeEmbedding(size=64)


def _write_doc_json(path: str, name: str, num_pages: int) -> None:
    """Write a minimal BaseReaderDocument JSON to *path* with *num_pages* pages."""
    pages = [
        {"page_content": f"Page {i} of {name}", "metadata": {"page_id": i}}
        for i in range(num_pages)
    ]
    data = {"date": "2026-01-01", "name": name, "outlines": [], "pages": pages}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _chroma_dir(persist_dir: str, name: str) -> str:
    return os.path.join(persist_dir, sanitize_name(name))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestChromaCachePreservedAfterClose:
    """Cache on disk is NOT deleted when close_document() is called."""

    def test_cache_exists_after_close(self, tmp_path):
        doc_path = str(tmp_path / "doc.json")
        persist_dir = str(tmp_path / "chroma_store")
        _write_doc_json(doc_path, "report-A", num_pages=3)

        reader = AgentDocumentReader(persist_dir=persist_dir, embedding_model_name="fake")
        original = tn.get_embedding_model
        tn.get_embedding_model = lambda model_name=None: _FAKE_EMB
        try:
            reader.open_document(doc_path)
            cache_dir = _chroma_dir(persist_dir, "report-A")
            assert os.path.isdir(cache_dir), "Chroma cache dir should exist after open"

            reader.close_document()
            assert os.path.isdir(cache_dir), "Chroma cache dir must survive close_document()"
        finally:
            tn.get_embedding_model = original

    def test_cache_reused_on_reopen(self, tmp_path):
        """Second open_document on same file reuses SQLite cache (from_documents called once)."""
        doc_path = str(tmp_path / "doc.json")
        persist_dir = str(tmp_path / "chroma_store")
        _write_doc_json(doc_path, "report-B", num_pages=3)

        from_documents_calls = []
        original_from_documents = Chroma.from_documents.__func__  # unbound

        def spy_from_documents(cls, documents, embedding, persist_directory, **kwargs):
            from_documents_calls.append(persist_directory)
            return original_from_documents(cls, documents, embedding, persist_directory=persist_directory, **kwargs)

        tn.get_embedding_model = lambda model_name=None: _FAKE_EMB
        try:
            with patch.object(Chroma, "from_documents", classmethod(spy_from_documents)):
                reader = AgentDocumentReader(persist_dir=persist_dir, embedding_model_name="fake")
                reader.open_document(doc_path)
                assert len(from_documents_calls) == 1, "First open must call from_documents once"

                reader.close_document()

                reader2 = AgentDocumentReader(persist_dir=persist_dir, embedding_model_name="fake")
                reader2.open_document(doc_path)
                # Cache hit: from_documents should NOT be called again
                assert len(from_documents_calls) == 1, "Second open should reuse cache"
                reader2.close_document()
        finally:
            tn.get_embedding_model = tn.__class__.get_embedding_model if hasattr(tn.__class__, "get_embedding_model") else tn.get_embedding_model


@pytest.mark.integration
class TestChromaCacheRebuiltOnPageCountChange:
    """Cache is rebuilt when the page count in the JSON file changes."""

    def test_rebuild_when_page_count_changes(self, tmp_path):
        doc_path = str(tmp_path / "doc.json")
        persist_dir = str(tmp_path / "chroma_store")
        _write_doc_json(doc_path, "report-C", num_pages=2)

        tn.get_embedding_model = lambda model_name=None: _FAKE_EMB
        try:
            with patch.object(Chroma, "from_documents", wraps=Chroma.from_documents) as mock_fd:
                # First open: 2 pages → builds cache
                reader = AgentDocumentReader(persist_dir=persist_dir, embedding_model_name="fake")
                reader.open_document(doc_path)
                assert mock_fd.call_count == 1
                reader.close_document()

                # Overwrite with 4 pages — page count changed
                _write_doc_json(doc_path, "report-C", num_pages=4)

                # open_document internally calls SharedSystemClient.clear_system_cache()
                # before shutil.rmtree when rebuilding (see text_navigator.py).
                # Second open: stale cache detected (2 != 4) → rebuild
                reader2 = AgentDocumentReader(persist_dir=persist_dir, embedding_model_name="fake")
                reader2.open_document(doc_path)
                assert mock_fd.call_count == 2, "Stale cache must trigger from_documents rebuild"

                cache_dir = _chroma_dir(persist_dir, "report-C")
                rebuilt_store = Chroma(
                    persist_directory=cache_dir,
                    embedding_function=_FAKE_EMB,
                )
                assert rebuilt_store._collection.count() == 4
                reader2.close_document()
        finally:
            SharedSystemClient.clear_system_cache()
