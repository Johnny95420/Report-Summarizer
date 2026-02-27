"""Tests for subagent.agentic_search — tool_calls guards (C8) and followed_up_queries default (I10)."""

import asyncio
import logging
from unittest.mock import MagicMock, patch

import pytest
from langgraph.graph import END

from subagent.agentic_search import (
    aggregate_final_results,
    check_search_quality_async,
    check_searching_results,
    compress_raw_content,
    filter_and_format_results,
    generate_queries_from_question,
    perform_web_search,
    queries_rewriter,
    synthesize_answer,
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
            "question": "What are Tesla's key revenue drivers?",
            "answer": "Some answer",
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
            "question": "What are Tesla's key revenue drivers?",
            "answer": "Tesla's main revenue driver is automotive sales.",
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
            "question": "What are Tesla's key revenue drivers?",
            "answer": "Tesla sells cars.",
            "curr_num_iterations": 0,
            "max_num_iterations": 5,
        }
        with patch("subagent.agentic_search.call_llm", return_value=mock_result):
            cmd = check_searching_results(state)
        assert cmd.goto == "perform_web_search"
        assert cmd.update["followed_up_queries"] == ["follow up"]

    def test_routes_to_end_when_iterations_exhausted(self):
        """When curr_num_iterations >= max_num_iterations, skip LLM and go to END."""
        state = {
            "question": "What are Tesla's key revenue drivers?",
            "answer": "Some answer",
            "curr_num_iterations": 3,
            "max_num_iterations": 3,
        }
        with patch("subagent.agentic_search.call_llm") as mock_llm:
            cmd = check_searching_results(state)
        mock_llm.assert_not_called()
        assert cmd.goto == END


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
            [[{"url": "http://x.com", "title": "T", "content": "C", "raw_content": "R" * 6000}]]
        )
        with patch("subagent.agentic_search.call_llm_async", side_effect=Exception("LLM down")):
            with caplog.at_level(logging.ERROR, logger="AgenticSearch"):
                _run(compress_raw_content(state))
        error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert any("Content compression failed" in msg for msg in error_messages), (
            "ERROR log must be emitted when compression LLM raises"
        )


# ---------------------------------------------------------------------------
# New nodes: generate_queries_from_question
# ---------------------------------------------------------------------------
class TestGenerateQueriesFromQuestion:
    def test_returns_queries_list(self):
        """Normal path: should return a list of search queries."""
        mock_result = MagicMock()
        mock_result.tool_calls = [{"args": {"queries": ["Tesla revenue Q1 2024", "Tesla FSD subscription 2024"]}}]
        state = {
            "question": (
                "Main Question: What are Tesla's key revenue drivers in Q1 2024?\n"
                "- Sub-question 1: What is the automotive vs. energy revenue split?\n"
                "- Sub-question 2: How did FSD subscription revenue perform?"
            )
        }
        with patch("subagent.agentic_search.call_llm", return_value=mock_result):
            result = generate_queries_from_question(state)
        assert result["queries"] == ["Tesla revenue Q1 2024", "Tesla FSD subscription 2024"]

    def test_falls_back_to_question_on_parse_failure(self):
        """When tool_calls parse fails, should fall back to the question itself as a query."""
        mock_result = MagicMock()
        mock_result.tool_calls = []  # empty → fallback
        question = "Main Question: What are Tesla's key revenue drivers?"
        state = {"question": question}
        with patch("subagent.agentic_search.call_llm", return_value=mock_result):
            result = generate_queries_from_question(state)
        assert result["queries"] == [question]


# ---------------------------------------------------------------------------
# New nodes: synthesize_answer
# ---------------------------------------------------------------------------
class TestSynthesizeAnswer:
    def test_returns_answer_on_first_iteration(self):
        """First iteration (empty previous answer): synthesizes answer from materials."""
        mock_result = MagicMock()
        mock_result.tool_calls = [{"args": {"answer": "Tesla's main revenue is automotive. [1]\n\n### Sources\n[1] Reuters — http://x.com"}}]
        state = {
            "question": "What are Tesla's key revenue drivers?",
            "answer": "",
            "materials": "## Source 1\nTesla Q1 2024 automotive revenue: $17.4B",
            "curr_num_iterations": 1,
        }
        with patch("subagent.agentic_search.call_llm", return_value=mock_result):
            result = synthesize_answer(state)
        assert "answer" in result
        assert len(result["answer"]) > 0

    def test_updates_answer_on_subsequent_iteration(self):
        """Subsequent iteration: synthesizes updated answer incorporating new materials."""
        mock_result = MagicMock()
        mock_result.tool_calls = [{"args": {"answer": "Updated comprehensive answer. [1][2]"}}]
        state = {
            "question": "What are Tesla's key revenue drivers?",
            "answer": "Initial answer from iteration 1. [1]",
            "materials": "New materials from iteration 2",
            "curr_num_iterations": 2,
        }
        with patch("subagent.agentic_search.call_llm", return_value=mock_result):
            result = synthesize_answer(state)
        assert result["answer"] == "Updated comprehensive answer. [1][2]"

    def test_falls_back_on_parse_failure(self):
        """When tool_calls parse fails, falls back to concatenating previous answer + materials."""
        mock_result = MagicMock()
        mock_result.tool_calls = []  # empty → fallback
        state = {
            "question": "What are Tesla's key revenue drivers?",
            "answer": "Previous answer",
            "materials": "New materials",
            "curr_num_iterations": 2,
        }
        with patch("subagent.agentic_search.call_llm", return_value=mock_result):
            result = synthesize_answer(state)
        assert "Previous answer" in result["answer"]
        assert "New materials" in result["answer"]


# ---------------------------------------------------------------------------
# aggregate_final_results: materials reset each round
# ---------------------------------------------------------------------------
class TestAggregateFinalResultsMaterialsReset:
    def test_materials_contains_only_current_iteration_results(self):
        """materials should be reset each round (not cumulative like old source_str)."""
        state = {
            "compressed_web_results": [{"results": [
                {"title": "Iteration 2 Article", "content": "c", "raw_content": "r", "url": "http://b.com"}
            ]}],
        }
        formatted = "Iteration 2 Article"
        with patch(
            "subagent.agentic_search.web_search_deduplicate_and_format_sources",
            return_value=formatted,
        ):
            result = aggregate_final_results(state)

        assert "materials" in result
        assert result["materials"] == formatted
        # Old source_str key must not appear
        assert "source_str" not in result


# ---------------------------------------------------------------------------
# compress_raw_content pass-through: short content skips LLM, long content calls it
# ---------------------------------------------------------------------------
def test_compress_passthrough_short_content():
    """Results with raw_content < 5000 chars are passed through without any LLM call."""
    short_result = {
        "title": "Short Article",
        "content": "brief",
        "url": "https://example.com/short",
        "raw_content": "x" * 100,  # well under 5000
    }
    state = {
        "queries": ["test query"],
        "followed_up_queries": [],
        "filtered_web_results": [{"results": [short_result]}],
    }
    with patch("subagent.agentic_search.call_llm_async") as mock_llm:
        result = asyncio.run(compress_raw_content(state))
    compressed = result["compressed_web_results"]
    # raw_content must be unchanged
    assert compressed[0]["results"][0]["raw_content"] == short_result["raw_content"]
    # No LLM call should have been made
    mock_llm.assert_not_called()


def test_compress_passthrough_none_raw_content():
    """Results with raw_content=None are passed through without crashing or being dropped."""
    none_result = {
        "title": "PDF Article",
        "content": "brief",
        "url": "https://example.com/file.pdf",
        "raw_content": None,
    }
    state = {
        "queries": ["test query"],
        "followed_up_queries": [],
        "filtered_web_results": [{"results": [none_result]}],
    }
    with patch("subagent.agentic_search.call_llm_async") as mock_llm:
        result = asyncio.run(compress_raw_content(state))
    compressed = result["compressed_web_results"]
    # Result must not be dropped
    assert len(compressed[0]["results"]) == 1
    # raw_content preserved as-is (None)
    assert compressed[0]["results"][0]["raw_content"] is None
    # No LLM call (None treated as empty → pass-through)
    mock_llm.assert_not_called()


def test_compress_llm_called_for_long_content():
    """Results with raw_content >= 5000 chars trigger the LLM compression path."""
    long_result = {
        "title": "Long Article",
        "content": "detailed",
        "url": "https://example.com/long",
        "raw_content": "y" * 6000,  # over 5000
    }
    state = {
        "queries": ["test query"],
        "followed_up_queries": [],
        "filtered_web_results": [{"results": [long_result]}],
    }
    mock_compressed = MagicMock()
    mock_compressed.tool_calls = [{"args": {"summary_content": "compressed"}}]
    with patch("subagent.agentic_search.call_llm_async", return_value=mock_compressed) as mock_llm:
        asyncio.run(compress_raw_content(state))
    # LLM must have been called exactly once
    mock_llm.assert_called_once()


# ---------------------------------------------------------------------------
# _format_sources_section pure function
# ---------------------------------------------------------------------------
class TestFormatSourcesSection:
    """Unit tests for _format_sources_section pure function."""

    def _make_registry(self, *urls):
        return [{"title": f"Title {i+1}", "url": url} for i, url in enumerate(urls)]

    def test_renumbers_sparse_citations(self):
        """[3][19] in text → renumbered [1][2]; Sources section uses new numbers."""
        from subagent.agentic_search import _format_sources_section
        registry = self._make_registry(
            "http://a.com", "http://b.com", "http://c.com",
            *[f"http://x{i}.com" for i in range(15)],  # 15 items → s.com is at index 18, citation [19]
            "http://s.com",
        )
        answer = "Claim A [3]. Claim B [19]."
        result = _format_sources_section(answer, registry)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" not in result.split("### Sources")[0]  # original [3] gone from body
        assert "[19]" not in result
        assert "http://c.com" in result
        assert "http://s.com" in result

    def test_no_citations_returns_original(self):
        """Answer with no [N] references is returned unchanged."""
        from subagent.agentic_search import _format_sources_section
        registry = self._make_registry("http://a.com")
        answer = "Answer with no citations."
        assert _format_sources_section(answer, registry) == answer

    def test_empty_registry_returns_original(self):
        """Empty registry → original answer returned unchanged."""
        from subagent.agentic_search import _format_sources_section
        answer = "Claim [1]."
        assert _format_sources_section(answer, []) == answer

    def test_out_of_range_index_skipped(self):
        """[N] beyond registry length → that entry skipped in Sources, no crash."""
        from subagent.agentic_search import _format_sources_section
        registry = self._make_registry("http://a.com")  # only index 1
        answer = "Claim [1]. Claim [5]."
        result = _format_sources_section(answer, registry)
        assert "http://a.com" in result
        assert "### Sources" in result
        # [5] is out of range — should not appear in Sources
        lines = result.split("\n")
        source_lines = [l for l in lines if l.startswith("- [")]
        assert len(source_lines) == 1

    def test_sources_section_format(self):
        """Sources section format: '- [N] Title — URL'."""
        from subagent.agentic_search import _format_sources_section
        registry = [{"title": "My Article", "url": "https://example.com/art"}]
        answer = "See [1]."
        result = _format_sources_section(answer, registry)
        assert "- [1] My Article — https://example.com/art" in result

    def test_empty_answer_returns_empty(self):
        """Empty answer string returns unchanged."""
        from subagent.agentic_search import _format_sources_section
        assert _format_sources_section("", [{"title": "T", "url": "http://x.com"}]) == ""
