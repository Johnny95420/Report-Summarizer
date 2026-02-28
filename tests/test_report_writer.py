"""Tests for report_writer.py"""

import ast
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.runnables import RunnableConfig

from .conftest import ROOT, find_function


class TestSearchRelevanceDoc:
    """search_relevance_doc must deepcopy docs before expanding content."""

    def test_uses_deepcopy_and_appends_expanded_copy(self):
        source = (ROOT / "report_writer.py").read_text()
        tree = ast.parse(source)

        func = find_function(tree, "search_relevance_doc")
        assert func is not None, "search_relevance_doc not found"

        src = ast.unparse(func)
        assert "deepcopy" in src
        assert "info.append(return_res)" in src
        # The table branch legitimately appends res directly;
        # verify the non-table else-branch uses return_res via deepcopy
        assert "return_res = deepcopy(res)" in src or "return_res = deepcopy" in src


class TestPrepareSourceForWriting:
    """_prepare_source_for_writing must include follow_up_queries in every
    format() call, including inside the truncation while-loop."""

    def test_all_format_calls_include_follow_up_queries(self):
        source = (ROOT / "report_writer.py").read_text()
        tree = ast.parse(source)

        func = find_function(tree, "_prepare_source_for_writing")
        assert func is not None

        format_calls = []
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "format":
                    keywords = {kw.arg for kw in node.keywords}
                    format_calls.append(keywords)

        assert len(format_calls) >= 2, f"Expected >=2 format calls (initial + loop), got {len(format_calls)}"
        for idx, kw_set in enumerate(format_calls):
            assert "follow_up_queries" in kw_set, f"format() call #{idx} missing follow_up_queries keyword"


class TestLoggerLevel:
    def test_logger_level_is_info(self):
        source = (ROOT / "report_writer.py").read_text()
        assert "logger.setLevel(logging.INFO)" in source


class TestExplicitPromptImports:
    """All 8 prompt variables should be explicitly imported (no wildcard)."""

    EXPECTED_NAMES = [
        "report_planner_query_writer_instructions",
        "report_planner_instructions",
        "query_writer_instructions",
        "section_writer_instructions",
        "section_grader_instructions",
        "final_section_writer_instructions",
        "refine_section_instructions",
        "content_refinement_instructions",
    ]

    def test_all_prompt_vars_explicitly_imported(self):
        source = (ROOT / "report_writer.py").read_text()
        tree = ast.parse(source)

        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "Prompt" in node.module:
                    for alias in node.names:
                        imported_names.add(alias.name)

        for name in self.EXPECTED_NAMES:
            assert name in imported_names, f"{name} not explicitly imported"

    def test_no_wildcard_import_from_prompt(self):
        source = (ROOT / "report_writer.py").read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "Prompt" in node.module:
                    for alias in node.names:
                        assert alias.name != "*", "Wildcard import from Prompt found"


class TestTypos:
    def test_no_indutry_typo(self):
        source = (ROOT / "report_writer.py").read_text()
        assert "indutry" not in source.lower(), '"indutry" typo still present in report_writer.py'


class TestGenerateQuestionGuard:
    """generate_question must retry with a reminder when LLM returns empty tool_calls.

    Unlike _grade_section_content (where empty = route to generate_question),
    generate_question drives the entire downstream research flow — an empty
    question is meaningless, so we must retry rather than silently fall back.
    """

    def _make_state(self):
        from State.state import Section

        return {
            "section": Section(name="Test", description="Test section desc", research=True, content=""),
            "weakness": "",
            "question_history": [],
        }

    def test_empty_tool_calls_triggers_retry(self):
        """When first call returns empty tool_calls, call_llm should be called a second time."""
        from report_writer import generate_question

        empty_result = MagicMock()
        empty_result.tool_calls = []
        retry_result = MagicMock()
        retry_result.tool_calls = [{"args": {"question": "Retry question"}}]

        with patch("report_writer.call_llm", side_effect=[empty_result, retry_result]) as mock_llm:
            result = generate_question(self._make_state(), RunnableConfig({"configurable": {}}))

        assert mock_llm.call_count == 2
        assert result["current_question"].question == "Retry question"

    def test_retry_still_empty_falls_back_to_section_description(self):
        """When both calls return empty tool_calls, fall back to section description as the question."""
        from report_writer import generate_question

        empty_result = MagicMock()
        empty_result.tool_calls = []

        with patch("report_writer.call_llm", return_value=empty_result):
            result = generate_question(self._make_state(), RunnableConfig({"configurable": {}}))

        assert result["current_question"].question == "Test section desc"

    def test_normal_path_returns_question(self):
        """Normal path: should return the question generated by the LLM on the first call."""
        from report_writer import generate_question

        mock_result = MagicMock()
        mock_result.tool_calls = [{"args": {"question": "What is CPO technology?"}}]

        with patch("report_writer.call_llm", return_value=mock_result) as mock_llm:
            result = generate_question(self._make_state(), RunnableConfig({"configurable": {}}))

        assert mock_llm.call_count == 1
        assert result["current_question"].question == "What is CPO technology?"
        assert result["question_history"] == ["What is CPO technology?"]


class TestLangGraphDeprecations:
    """Ensure deprecated LangGraph APIs are not used."""

    def test_send_imported_from_langgraph_types(self):
        source = (ROOT / "report_writer.py").read_text()
        assert "from langgraph.constants import Send" not in source, (
            "Send should be imported from langgraph.types, not langgraph.constants"
        )

    def test_state_graph_uses_schema_kwargs(self):
        source = (ROOT / "report_writer.py").read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "StateGraph":
                    kw_names = {kw.arg for kw in node.keywords}
                    assert "input" not in kw_names, "StateGraph uses deprecated 'input' kwarg; use 'input_schema'"
                    assert "output" not in kw_names, "StateGraph uses deprecated 'output' kwarg; use 'output_schema'"
