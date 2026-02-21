"""Tests for project configuration (pyproject.toml, config loading patterns)."""

import pytest

from .conftest import ROOT


class TestDependencies:
    def test_langchain_classic_in_pyproject(self):
        source = (ROOT / "pyproject.toml").read_text()
        assert "langchain-classic" in source, "langchain-classic not declared in pyproject.toml"


class TestConfigPaths:
    """All modules that load config should use pathlib for robust path resolution."""

    FILES_TO_CHECK = [
        "report_writer.py",
        "subagent/agentic_search.py",
        "subagent/document_qa.py",
        "retriever.py",
    ]

    @pytest.mark.parametrize("filename", FILES_TO_CHECK)
    def test_pathlib_used_for_config(self, filename):
        source = (ROOT / filename).read_text()
        assert "Path(__file__).parent" in source, f"{filename} does not use Path(__file__).parent for config loading"
