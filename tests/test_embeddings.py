"""Tests for Utils/embeddings.py — embedding singleton."""

import threading
from unittest.mock import MagicMock, patch

import pytest


class TestGetEmbeddingModel:
    def test_embedding_model_load_failure(self):
        """Load failure must raise RuntimeError with a descriptive message."""
        import Utils.embeddings as emb

        old_state = emb._embedding_state
        emb._embedding_state = None
        try:
            with (
                patch("Utils.embeddings.HuggingFaceEmbeddings", side_effect=OSError("disk full")),
                pytest.raises(RuntimeError, match="Failed to load embedding model"),
            ):
                emb.get_embedding_model("bad-model")
        finally:
            emb._embedding_state = old_state

    def test_cache_hit_skips_initialization(self):
        """Fast path must NOT call HuggingFaceEmbeddings() again on cache hit."""
        import Utils.embeddings as emb

        old_state = emb._embedding_state
        emb._embedding_state = None
        try:
            with patch("Utils.embeddings.HuggingFaceEmbeddings") as mock_emb:
                mock_emb.return_value = MagicMock(model_name="shared-model")
                emb1 = emb.get_embedding_model("shared-model")
                assert mock_emb.call_count == 1
                emb2 = emb.get_embedding_model("shared-model")
                assert mock_emb.call_count == 1, "Cache hit must not reinitialize"
                assert emb1 is emb2
        finally:
            emb._embedding_state = old_state

    def test_model_name_mismatch_forces_rebuild(self):
        """Requesting a different model_name must build a new HuggingFaceEmbeddings instance."""
        import Utils.embeddings as emb

        old_state = emb._embedding_state
        emb._embedding_state = None
        try:
            class _FakeEmb:
                def __init__(self, model_name):
                    self.model_name = model_name

            with patch("Utils.embeddings.HuggingFaceEmbeddings", side_effect=_FakeEmb) as mock_emb:
                emb1 = emb.get_embedding_model("model-a")
                assert emb1.model_name == "model-a"
                assert mock_emb.call_count == 1
                emb2 = emb.get_embedding_model("model-b")
                assert emb2.model_name == "model-b"
                assert mock_emb.call_count == 2, "Different model name must trigger rebuild"
                assert emb1 is not emb2
        finally:
            emb._embedding_state = old_state

    def test_concurrent_requests_return_matching_model_name(self):
        """Concurrent callers must never receive an embedding for the wrong model name."""
        import Utils.embeddings as emb

        old_state = emb._embedding_state
        emb._embedding_state = None
        mismatches: list = []
        names = ("model-a", "model-b")
        start = threading.Barrier(8)

        class _FakeEmb:
            def __init__(self, model_name):
                self.model_name = model_name

        def _worker(worker_id):
            start.wait()
            for i in range(200):
                requested = names[(worker_id + i) % 2]
                e = emb.get_embedding_model(requested)
                if e.model_name != requested:
                    mismatches.append((requested, e.model_name))

        try:
            with patch("Utils.embeddings.HuggingFaceEmbeddings", side_effect=_FakeEmb):
                threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
        finally:
            emb._embedding_state = old_state

        assert mismatches == []
