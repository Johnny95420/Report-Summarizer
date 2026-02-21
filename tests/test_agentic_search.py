"""Tests for subagent.agentic_search — tool_calls guards (C8) and followed_up_queries default (I10)."""

import asyncio
from unittest.mock import MagicMock, patch

from langgraph.graph import END

from subagent.agentic_search import (
    check_search_quality_async,
    check_searching_results,
    compress_raw_content,
    perform_web_search,
    queries_rewriter,
)


# ---------------------------------------------------------------------------
# C8 — Guard #1: queries_rewriter falls back on parse failure
# ---------------------------------------------------------------------------
class TestQueriesRewriterGuard:
    def test_returns_original_on_empty_tool_calls(self):
        """When tool_calls is empty, should return original queries unchanged."""
        mock_result = MagicMock()
        mock_result.tool_calls = []
        with patch("subagent.agentic_search.call_llm", return_value=mock_result):
            result = queries_rewriter(["query A", "query B"])
        assert result == ["query A", "query B"]

    def test_returns_original_on_missing_args_key(self):
        """When tool_calls[0] has no 'queries' key in args, should return original."""
        mock_result = MagicMock()
        mock_result.tool_calls = [{"args": {"wrong_key": "value"}}]
        with patch("subagent.agentic_search.call_llm", return_value=mock_result):
            result = queries_rewriter(["original"])
        assert result == ["original"]

    def test_returns_rewritten_on_success(self):
        """Normal path: should return the rewritten queries."""
        mock_result = MagicMock()
        mock_result.tool_calls = [{"args": {"queries": ["rewritten A", "rewritten B"]}}]
        with patch("subagent.agentic_search.call_llm", return_value=mock_result):
            result = queries_rewriter(["query A", "query B"])
        assert result == ["rewritten A", "rewritten B"]


# ---------------------------------------------------------------------------
# C8 — Guard #2: check_search_quality_async returns 0 on parse failure
# ---------------------------------------------------------------------------
class TestCheckSearchQualityGuard:
    def test_returns_zero_on_empty_tool_calls(self):
        """When tool_calls is empty, should return score 0."""
        mock_result = MagicMock()
        mock_result.tool_calls = []
        with patch("subagent.agentic_search.call_llm_async", return_value=mock_result):
            score = asyncio.get_event_loop().run_until_complete(check_search_quality_async("query", "document text"))
        assert score == 0

    def test_returns_score_on_success(self):
        """Normal path: should return the parsed score."""
        mock_result = MagicMock()
        mock_result.tool_calls = [{"args": {"score": 4}}]
        with patch("subagent.agentic_search.call_llm_async", return_value=mock_result):
            score = asyncio.get_event_loop().run_until_complete(check_search_quality_async("query", "document text"))
        assert score == 4


# ---------------------------------------------------------------------------
# C8 — Guard #3: compress_raw_content falls back on parse failure
# ---------------------------------------------------------------------------
class TestCompressRawContentGuard:
    def test_uses_original_on_tool_calls_parse_failure(self):
        """When compressed_result.tool_calls has bad structure, use original content."""
        mock_compressed = MagicMock()
        # tool_calls with missing 'summary_content' key
        mock_compressed.tool_calls = [{"args": {"wrong_key": "value"}}]

        state = {
            "queries": ["test query"],
            "followed_up_queries": [],
            "filtered_web_results": [
                {"results": [{"title": "T", "content": "C", "raw_content": "RC", "url": "http://x"}]}
            ],
        }
        with patch("subagent.agentic_search.call_llm_async", return_value=mock_compressed):
            result = asyncio.get_event_loop().run_until_complete(compress_raw_content(state))

        # Should fall back to original result
        assert len(result["compressed_web_results"][0]["results"]) == 1
        assert result["compressed_web_results"][0]["results"][0]["raw_content"] == "RC"


# ---------------------------------------------------------------------------
# C8 — Guard #4: check_searching_results returns END on parse failure
# ---------------------------------------------------------------------------
class TestCheckSearchingResultsGuard:
    def test_routes_to_end_on_parse_failure(self):
        """When feedback tool_calls parse fails, should return Command(goto=END)."""
        mock_result = MagicMock()
        mock_result.tool_calls = []  # empty → IndexError
        state = {
            "queries": ["test"],
            "source_str": "some sources",
            "curr_num_iterations": 0,
            "max_num_iterations": 5,
        }
        with patch("subagent.agentic_search.call_llm", return_value=mock_result):
            cmd = check_searching_results(state)
        assert cmd.goto == END

    def test_routes_to_end_on_pass(self):
        """Normal path: grade='pass' should route to END."""
        mock_result = MagicMock()
        mock_result.tool_calls = [{"args": {"grade": "pass", "follow_up_queries": []}}]
        state = {
            "queries": ["test"],
            "source_str": "some sources",
            "curr_num_iterations": 0,
            "max_num_iterations": 5,
        }
        with patch("subagent.agentic_search.call_llm", return_value=mock_result):
            cmd = check_searching_results(state)
        assert cmd.goto == END

    def test_routes_to_search_on_fail(self):
        """Normal path: grade='fail' should route back to perform_web_search."""
        mock_result = MagicMock()
        mock_result.tool_calls = [{"args": {"grade": "fail", "follow_up_queries": ["follow up"]}}]
        state = {
            "queries": ["test"],
            "source_str": "some sources",
            "curr_num_iterations": 0,
            "max_num_iterations": 5,
        }
        with patch("subagent.agentic_search.call_llm", return_value=mock_result):
            cmd = check_searching_results(state)
        assert cmd.goto == "perform_web_search"
        assert cmd.update["followed_up_queries"] == ["follow up"]


# ---------------------------------------------------------------------------
# I10 — followed_up_queries default is [] not ""
# ---------------------------------------------------------------------------
class TestFollowedUpQueriesDefault:
    def test_perform_web_search_without_followed_up_queries(self):
        """Missing followed_up_queries key should default to [] not '' (I10 fix)."""
        state = {
            "queries": ["test query"],
            "url_memo": set(),
            "curr_num_iterations": 0,
            # followed_up_queries intentionally absent
        }
        with patch("subagent.agentic_search.selenium_api_search", return_value=[{"results": []}]) as mock_search:
            result = perform_web_search(state)
        assert result["curr_num_iterations"] == 1
        # Should have searched with original queries, not empty string
        mock_search.assert_called_once_with(["test query"], True)

    def test_perform_web_search_with_followed_up_queries(self):
        """When followed_up_queries is present and non-empty, use it instead of queries."""
        state = {
            "queries": ["original"],
            "followed_up_queries": ["follow up query"],
            "url_memo": set(),
            "curr_num_iterations": 0,
        }
        with patch("subagent.agentic_search.selenium_api_search", return_value=[{"results": []}]) as mock_search:
            perform_web_search(state)
        mock_search.assert_called_once_with(["follow up query"], True)
