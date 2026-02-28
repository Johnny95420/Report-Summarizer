"""Shared embedding model singleton.

Provides get_embedding_model() with double-checked locking so the
HuggingFaceEmbeddings instance is initialized at most once per model name
across all modules (text_navigator, agentic_search, etc.).
"""
import logging
import pathlib
import threading

import omegaconf
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger("Embeddings")

_cfg = omegaconf.OmegaConf.load(pathlib.Path(__file__).parent.parent / "retriever_config.yaml")
_DEFAULT_EMBEDDING_MODEL: str = str(_cfg.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B"))

_embedding_state: tuple[str, HuggingFaceEmbeddings] | None = None
_embedding_lock = threading.Lock()


def get_embedding_model(model_name: str = _DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Return a cached HuggingFaceEmbeddings instance (double-checked locking singleton).

    Args:
        model_name: HuggingFace model identifier. Defaults to retriever_config.yaml value.

    Returns:
        Shared HuggingFaceEmbeddings instance for model_name.

    Raises:
        RuntimeError: If the model cannot be loaded.
    """
    global _embedding_state
    state = _embedding_state  # fast path — no lock
    if state is not None and state[0] == model_name:
        return state[1]
    with _embedding_lock:  # slow path
        state = _embedding_state
        if state is None or state[0] != model_name:
            try:
                new_model = HuggingFaceEmbeddings(model_name=model_name)
            except Exception as e:
                raise RuntimeError(f"Failed to load embedding model '{model_name}': {e}") from e
            _embedding_state = (model_name, new_model)
        return _embedding_state[1]
