"""Shared fixtures and helpers for the test suite.

Mocks heavy dependencies (OmegaConf, dotenv) so tests can run without
GPU, config files, API keys, or running services.
"""

import ast
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Project root — shared across all test modules
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Custom markers
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line("markers", "integration: integration tests that run without external services")


# ---------------------------------------------------------------------------
# Fake config that satisfies every module-level OmegaConf.load() call
# ---------------------------------------------------------------------------
FAKE_CONFIG = {
    # report_writer / agentic_search model config
    "PLANNER_MODEL_NAME": "test-model",
    "BACKUP_PLANNER_MODEL_NAME": "test-model",
    "VERIFY_MODEL_NAME": "test-model",
    "BACKUP_VERIFY_MODEL_NAME": "test-model",
    "MODEL_NAME": "test-model",
    "BACKUP_MODEL_NAME": "test-model",
    "WRITER_MODEL_NAME": "test-model",
    "BACKUP_WRITER_MODEL_NAME": "test-model",
    "CONCLUDE_MODEL_NAME": "test-model",
    "BACKUP_CONCLUDE_MODEL_NAME": "test-model",
    "LIGHT_MODEL_NAME": "test-model",
    "BACKUP_LIGHT_MODEL_NAME": "test-model",
    "REPORT_STRUCTURE": "default",
    # retriever config (raw_file_path=None skips data loading)
    "raw_file_path": None,
    "navigator_top_k": 5,
    "navigator_persist_dir": "navigator_tmp",
    "reader_tmp_dir": "reader_tmp",
}


class _FakeOmegaConf:
    """Minimal stand-in for omegaconf.OmegaConf."""

    @staticmethod
    def load(path):
        return FAKE_CONFIG


@pytest.fixture(autouse=True)
def _mock_omegaconf(monkeypatch):
    """Patch OmegaConf.load globally so module-level config reads succeed."""
    import omegaconf

    monkeypatch.setattr(omegaconf.OmegaConf, "load", _FakeOmegaConf.load)


@pytest.fixture(autouse=True)
def _mock_dotenv(monkeypatch):
    """Prevent load_dotenv from touching the real .env file."""
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: None)


@pytest.fixture(autouse=True)
def _mock_env_vars(monkeypatch):
    """Set dummy env vars required by module-level initialisations."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


# ---------------------------------------------------------------------------
# Shared AST helper
# ---------------------------------------------------------------------------
def find_function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    """Find a function definition by name in an AST (searches all depths)."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return node
    return None
