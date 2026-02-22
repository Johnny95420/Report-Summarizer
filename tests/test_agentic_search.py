"""Tests for subagent.agentic_search — tool_calls guards (C8) and followed_up_queries default (I10)."""

import asyncio
import logging
from unittest.mock import MagicMock, patch

import pytest
from langgraph.graph import END

from subagent.agentic_search import (
    check_search_quality_async,
    check_searching_results,
    compress_raw_content,
    filter_and_format_results,
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
            score = asyncio.run(check_search_quality_async("query", "document text"))
        assert score == 0

    def test_returns_score_on_success(self):
        """Normal path: should return the parsed score."""
        mock_result = MagicMock()
        mock_result.tool_calls = [{"args": {"score": 4}}]
        with patch("subagent.agentic_search.call_llm_async", return_value=mock_result):
            score = asyncio.run(check_search_quality_async("query", "document text"))
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
            result = asyncio.run(compress_raw_content(state))

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

    def test_url_memo_returned_in_state(self):
        """url_memo must be included in the return dict so LangGraph persists it across iterations."""
        state = {
            "queries": ["q1"],
            "url_memo": set(),
            "curr_num_iterations": 0,
        }
        fake_results = [{"results": [{"url": "http://a.com", "title": "A", "content": "c", "raw_content": "r"}]}]
        with patch("subagent.agentic_search.selenium_api_search", return_value=fake_results):
            result = perform_web_search(state)
        assert "url_memo" in result
        assert "http://a.com" in result["url_memo"]

    def test_url_memo_deduplicates_across_iterations(self):
        """A URL seen in iteration 1 must be excluded in iteration 2."""
        seen_url = "http://dup.com"
        state_iter1 = {
            "queries": ["q1"],
            "url_memo": set(),
            "curr_num_iterations": 0,
        }
        fake_results = [{"results": [{"url": seen_url, "title": "T", "content": "c", "raw_content": "r"}]}]
        with patch("subagent.agentic_search.selenium_api_search", return_value=fake_results):
            result1 = perform_web_search(state_iter1)

        # Simulate LangGraph feeding back the returned url_memo as the next iteration's state
        state_iter2 = {
            "queries": ["q1"],
            "url_memo": result1["url_memo"],
            "curr_num_iterations": result1["curr_num_iterations"],
        }
        with patch("subagent.agentic_search.selenium_api_search", return_value=fake_results):
            result2 = perform_web_search(state_iter2)

        assert result2["web_results"][0]["results"] == [], "duplicate URL should be filtered out in second iteration"

    def test_url_memo_partial_dedup_within_batch(self):
        """url_memo pre-seeded with one URL: that URL is filtered, new URL passes through."""
        state = {
            "queries": ["q1"],
            "url_memo": {"http://seen.com"},
            "curr_num_iterations": 1,
        }
        fake_results = [{"results": [
            {"url": "http://seen.com", "title": "old", "content": "c", "raw_content": "r"},
            {"url": "http://new.com",  "title": "new", "content": "c", "raw_content": "r"},
        ]}]
        with patch("subagent.agentic_search.selenium_api_search", return_value=fake_results):
            result = perform_web_search(state)

        urls_in_results = [r["url"] for r in result["web_results"][0]["results"]]
        assert "http://seen.com" not in urls_in_results
        assert "http://new.com" in urls_in_results
        assert "http://seen.com" in result["url_memo"]
        assert "http://new.com" in result["url_memo"]


# ---------------------------------------------------------------------------
# Helpers shared by filter_and_format_results tests
# ---------------------------------------------------------------------------
def _run(coro):
    return asyncio.run(coro)


def _filter_state(results_per_query: list[list[dict]]) -> dict:
    return {
        "queries": [f"q{i}" for i in range(len(results_per_query))],
        "followed_up_queries": [],
        "web_results": [{"results": items} for items in results_per_query],
    }


# ---------------------------------------------------------------------------
# #17 — LLM failure during quality check must not silently drop all results
# ---------------------------------------------------------------------------
class TestQualityFilterLLMFailure:
    def test_llm_failure_includes_result_in_output(self):
        """#17: When call_llm_async raises during quality check, result must still appear in output."""
        state = _filter_state([[{"url": "http://x.com", "title": "T", "content": "C", "raw_content": "R"}]])
        with patch("subagent.agentic_search.call_llm_async", side_effect=Exception("LLM down")):
            result = _run(filter_and_format_results(state))
        assert len(result["filtered_web_results"][0]["results"]) == 1, (
            "result must be preserved when quality LLM fails, not silently dropped"
        )

    def test_all_results_included_when_entire_llm_fails(self):
        """
        Fail-open policy: when call_llm_async fails for every result,
        all results must be included rather than producing an empty corpus.
        """
        state = _filter_state([
            [
                {"url": "http://a.com", "title": "T", "content": "C", "raw_content": "R"},
                {"url": "http://b.com", "title": "T", "content": "C", "raw_content": "R"},
            ]
        ])
        with patch("subagent.agentic_search.call_llm_async", side_effect=Exception("LLM down")):
            result = _run(filter_and_format_results(state))
        assert len(result["filtered_web_results"][0]["results"]) == 2, (
            "both results must be preserved when quality LLM is entirely unavailable"
        )


# ---------------------------------------------------------------------------
# #18 — Diagnostic events (quality failures, compression failures) must be logged at ERROR
#        so they are visible under the module's ERROR-level logger
# ---------------------------------------------------------------------------
class TestLoggerLevel:
    def test_quality_check_failure_emits_error_log(self, caplog):
        """#18: Quality check failure must emit an ERROR log so it is not suppressed."""
        state = _filter_state([[{"url": "http://x.com", "title": "T", "content": "C", "raw_content": "R"}]])
        with patch("subagent.agentic_search.call_llm_async", side_effect=Exception("LLM down")):
            with caplog.at_level(logging.ERROR, logger="AgenticSearch"):
                _run(filter_and_format_results(state))
        error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert any("Quality check failed" in msg for msg in error_messages), (
            "ERROR log must be emitted when quality check fails so it is visible under ERROR-level logger"
        )


# ---------------------------------------------------------------------------
# #19 — Regression: per-task exception must not prevent other results processing
# ---------------------------------------------------------------------------
class TestScoreThresholdAndExceptionHandling:
    def test_failing_quality_for_one_query_does_not_drop_other_query_results(self):
        """#19 regression: A low/failed quality check for q0 must not prevent q1 results from appearing."""
        state = _filter_state([
            [{"url": "http://low.com", "title": "T", "content": "C", "raw_content": "R"}],
            [{"url": "http://high.com", "title": "T", "content": "C", "raw_content": "R"}],
        ])
        call_count = 0

        async def fake_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock = MagicMock()
            mock.tool_calls = [{"args": {"score": 1 if call_count == 1 else 5}}]
            return mock

        with patch("subagent.agentic_search.call_llm_async", side_effect=fake_llm):
            result = _run(filter_and_format_results(state))

        assert len(result["filtered_web_results"][0]["results"]) == 0, "score=1 should be filtered"
        assert len(result["filtered_web_results"][1]["results"]) == 1, "score=5 should pass"

    def test_gather_level_exception_drops_that_result_but_preserves_others(self):
        """
        #19 regression: if a coroutine propagates an exception into gather's return list
        (the isinstance branch), that result is dropped but others are still processed.
        This guards against future refactors that move the inner try/except out of the closure.
        """
        state = _filter_state([
            [{"url": "http://fail.com", "title": "T", "content": "C", "raw_content": "R"}],
            [{"url": "http://pass.com", "title": "T", "content": "C", "raw_content": "R"}],
        ])

        async def fake_gather(*coros, return_exceptions=False):
            # Inject a raw Exception as the first element (simulates propagated coroutine failure)
            results = [RuntimeError("simulated propagated exception")]
            for coro in list(coros)[1:]:
                results.append(await coro)
            return results

        mock_llm = MagicMock()
        mock_llm.tool_calls = [{"args": {"score": 5}}]

        with patch("subagent.agentic_search.call_llm_async", return_value=mock_llm):
            with patch("subagent.agentic_search.asyncio.gather", side_effect=fake_gather):
                result = _run(filter_and_format_results(state))

        # First result dropped (isinstance Exception branch hits continue)
        assert len(result["filtered_web_results"][0]["results"]) == 0
        # Second result still processes normally
        assert len(result["filtered_web_results"][1]["results"]) == 1


# ---------------------------------------------------------------------------
# Helper shared by compress_raw_content tests
# ---------------------------------------------------------------------------
def _compress_state(results_per_query: list[list[dict]]) -> dict:
    return {
        "queries": [f"q{i}" for i in range(len(results_per_query))],
        "followed_up_queries": [],
        "filtered_web_results": [{"results": items} for items in results_per_query],
    }


# ---------------------------------------------------------------------------
# compress_raw_content LLM failure — original content must be preserved
# ---------------------------------------------------------------------------
class TestCompressRawContentLLMFailure:
    def test_llm_exception_preserves_original_content(self):
        """
        When call_llm_async raises during compression, the original result
        must be preserved unchanged (compress_content_with_metadata returns None,
        outer loop falls through to else branch and appends original_result).
        """
        original = {"url": "http://x.com", "title": "T", "content": "C", "raw_content": "RC"}
        state = _compress_state([[original]])
        with patch("subagent.agentic_search.call_llm_async", side_effect=Exception("LLM down")):
            result = _run(compress_raw_content(state))
        assert len(result["compressed_web_results"][0]["results"]) == 1, (
            "original result must be preserved when compression LLM raises"
        )
        assert result["compressed_web_results"][0]["results"][0]["raw_content"] == "RC", (
            "raw_content must be the original, not replaced with empty summary"
        )

    def test_llm_exception_emits_error_log(self, caplog):
        """
        When call_llm_async raises during compression, an ERROR log must be emitted
        so it is visible under the module's ERROR-level logger.
        """
        state = _compress_state(
            [[{"url": "http://x.com", "title": "T", "content": "C", "raw_content": "RC"}]]
        )
        with patch("subagent.agentic_search.call_llm_async", side_effect=Exception("LLM down")):
            with caplog.at_level(logging.ERROR, logger="AgenticSearch"):
                _run(compress_raw_content(state))
        error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert any("Content compression failed" in msg for msg in error_messages), (
            "ERROR log must be emitted when compression LLM raises"
        )
