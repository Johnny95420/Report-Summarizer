"""Tests for Utils/pdf_processor.py and preprocess_files.py"""

import ast

from .conftest import ROOT, find_function


class TestChatLiteLLMImport:
    """ChatLiteLLM must be imported from langchain_litellm, not langchain_community."""

    def test_import_from_langchain_litellm(self):
        source = (ROOT / "Utils" / "pdf_processor.py").read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "langchain_community.chat_models", (
                    "ChatLiteLLM must be imported from langchain_litellm, "
                    "not the deprecated langchain_community.chat_models"
                )

        imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "langchain_litellm"
        ]
        names = [alias.name for imp in imports for alias in imp.names]
        assert "ChatLiteLLM" in names, "ChatLiteLLM must be imported from langchain_litellm"


class TestJsonDumpEnsureAscii:
    """json.dump calls must use ensure_ascii=False to write Chinese characters correctly."""

    def _get_dump_calls(self, tree: ast.Module) -> list[ast.Call]:
        calls = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "dump"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "json"
            ):
                calls.append(node)
        return calls

    def _has_ensure_ascii_false(self, call: ast.Call) -> bool:
        for kw in call.keywords:
            if kw.arg == "ensure_ascii" and isinstance(kw.value, ast.Constant):
                return kw.value.value is False
        return False

    def test_all_json_dumps_have_ensure_ascii_false(self):
        source = (ROOT / "Utils" / "pdf_processor.py").read_text()
        tree = ast.parse(source)

        dump_calls = self._get_dump_calls(tree)
        assert len(dump_calls) > 0, "Expected at least one json.dump call in pdf_processor.py"

        for call in dump_calls:
            assert self._has_ensure_ascii_false(call), (
                "All json.dump calls must have ensure_ascii=False to preserve Chinese characters"
            )


class TestPreprocessFilePaths:
    """preprocess_files.py must use /root/pdf_parser paths, not /pdf_parser."""

    def test_paths_use_root_prefix(self):
        source = (ROOT / "preprocess_files.py").read_text()

        assert "/pdf_parser/raw_pdf" not in source or "/root/pdf_parser/raw_pdf" in source, (
            "raw_pdf path must use /root/pdf_parser/raw_pdf, not /pdf_parser/raw_pdf"
        )
        assert "/pdf_parser/raw_md" not in source or "/root/pdf_parser/raw_md" in source, (
            "raw_md path must use /root/pdf_parser/raw_md, not /pdf_parser/raw_md"
        )
        assert "/root/pdf_parser/raw_pdf" in source, "raw_pdf path must be /root/pdf_parser/raw_pdf"
        assert "/root/pdf_parser/raw_md" in source, "raw_md path must be /root/pdf_parser/raw_md"
