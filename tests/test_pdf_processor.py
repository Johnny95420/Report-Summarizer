"""Tests for Utils/pdf_processor.py and preprocess_files.py"""

import ast

from .conftest import ROOT


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
            node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module == "langchain_litellm"
        ]
        names = [alias.name for imp in imports for alias in imp.names]
        assert "ChatLiteLLM" in names, "ChatLiteLLM must be imported from langchain_litellm"

    def test_langchain_litellm_in_pyproject(self):
        pyproject = (ROOT / "pyproject.toml").read_text()
        assert "langchain-litellm" in pyproject, "langchain-litellm must be declared in pyproject.toml dependencies"


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


class TestParseBugFixes:
    """Regression tests for critical bugs fixed in PDFProcessor.parse()."""

    def _get_parse_method(self, tree: ast.Module) -> ast.AsyncFunctionDef:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "parse":
                return node
        raise AssertionError("parse() method not found in pdf_processor.py")

    def test_loop_variable_not_shadowed_by_file_handle(self):
        """parse() must not use the same name for the loop var and the open() handle."""
        source = (ROOT / "Utils" / "pdf_processor.py").read_text()
        tree = ast.parse(source)
        parse_fn = self._get_parse_method(tree)

        for_loops = [n for n in ast.walk(parse_fn) if isinstance(n, ast.For)]
        assert for_loops, "parse() must contain a for loop"
        loop_var = for_loops[0].target.id  # e.g. "file_path"

        with_items = [n for n in ast.walk(parse_fn) if isinstance(n, ast.withitem)]
        handle_names = [item.optional_vars.id for item in with_items if isinstance(item.optional_vars, ast.Name)]
        assert loop_var not in handle_names, (
            f"Loop variable '{loop_var}' must not be reused as a file handle — "
            "this shadows the path on the second iteration"
        )

    def test_tables_always_assigned_before_use(self):
        """tables must be assigned unconditionally so do_extract_table=False never raises NameError."""
        source = (ROOT / "Utils" / "pdf_processor.py").read_text()
        tree = ast.parse(source)
        parse_fn = self._get_parse_method(tree)

        # tables = ... must appear as a direct (non-nested-in-if) assign in the for-loop body
        for_loop = next(n for n in ast.walk(parse_fn) if isinstance(n, ast.For))
        direct_assigns = [
            n
            for n in for_loop.body
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "tables" for t in n.targets)
        ]
        assert direct_assigns, (
            "'tables' must be assigned unconditionally in parse() loop body "
            "(use ternary or initialize before if-block) to avoid NameError when do_extract_table=False"
        )

    def test_bare_except_logs_warning(self):
        """The retry except in financial_report_metadata_extraction must log a warning."""
        source = (ROOT / "Utils" / "pdf_processor.py").read_text()
        tree = ast.parse(source)

        extraction_fn = next(
            (
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.AsyncFunctionDef) and n.name == "financial_report_metadata_extraction"
            ),
            None,
        )
        assert extraction_fn is not None, "financial_report_metadata_extraction not found"

        try_nodes = [n for n in ast.walk(extraction_fn) if isinstance(n, ast.Try)]
        assert try_nodes, "No try/except found in financial_report_metadata_extraction"

        for try_node in try_nodes:
            for handler in try_node.handlers:
                # handler must capture the exception (as e) and call logger.warning
                assert handler.name is not None, "except clause must capture exception with 'as e' to enable logging"
                warning_calls = [
                    n
                    for n in ast.walk(handler)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "warning"
                ]
                assert warning_calls, "except handler must call logger.warning() — bare except swallows errors silently"


class TestPreprocessFilesCLI:
    """preprocess_files.py must be a typer CLI with no hardcoded paths."""

    def test_no_hardcoded_paths(self):
        source = (ROOT / "preprocess_files.py").read_text()
        assert "/pdf_parser/raw_pdf" not in source, "raw_pdf path must not be hardcoded"
        assert "/pdf_parser/raw_md" not in source, "raw_md path must not be hardcoded"

    def test_uses_typer(self):
        source = (ROOT / "preprocess_files.py").read_text()
        assert "typer" in source, "preprocess_files.py must use typer for CLI"

    def test_input_output_are_arguments(self):
        source = (ROOT / "preprocess_files.py").read_text()
        tree = ast.parse(source)
        func = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main"),
            None,
        )
        assert func is not None, "main() function not found in preprocess_files.py"
        arg_names = [a.arg for a in func.args.args]
        assert "input_dir" in arg_names, "main() must have input_dir argument"
        assert "output_dir" in arg_names, "main() must have output_dir argument"
