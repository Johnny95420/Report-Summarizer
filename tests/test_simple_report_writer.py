"""Tests for simple_report_writer.py"""

import ast

from .conftest import ROOT


class TestImports:
    def test_uses_langchain_litellm(self):
        source = (ROOT / "simple_report_writer.py").read_text()
        assert "from langchain_litellm import ChatLiteLLM" in source, (
            "simple_report_writer.py should import ChatLiteLLM from langchain_litellm"
        )

    def test_import_ordering_clean(self):
        """load_dotenv() should come first, then stdlib, third-party, local imports
        should not be interleaved with assignments."""
        source = (ROOT / "simple_report_writer.py").read_text()
        tree = ast.parse(source)

        # Collect top-level import line numbers (excluding load_dotenv block)
        import_lines = []
        non_import_lines = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_lines.append(node.lineno)
            elif isinstance(node, ast.Assign):
                non_import_lines.append(node.lineno)

        # Verify no assignment appears between import statements
        if import_lines and non_import_lines:
            first_import = min(import_lines)
            last_import = max(import_lines)
            interleaved = [ln for ln in non_import_lines if first_import < ln < last_import]
            assert len(interleaved) == 0, f"Assignments on lines {interleaved} are interleaved between imports"
