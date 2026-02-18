"""Tests for simple_report_writer.py"""

from pathlib import Path

from .conftest import ROOT


class TestImports:
    def test_uses_langchain_litellm(self):
        source = (ROOT / "simple_report_writer.py").read_text()
        assert "from langchain_litellm import ChatLiteLLM" in source, (
            "simple_report_writer.py should import ChatLiteLLM from langchain_litellm"
        )
