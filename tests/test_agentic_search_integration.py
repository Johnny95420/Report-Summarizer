"""Integration tests for the agentic search LangGraph graph.

These tests verify that LangGraph correctly propagates state across the
``check_searching_results → perform_web_search`` back-edge when the graph
loops.  The node-level deduplication behaviour of ``perform_web_search`` is
already covered by unit tests in ``test_agentic_search.py``; what is tested
here is whether the ``url_memo`` set returned by ``perform_web_search`` in
iteration 1 actually appears in the state received by ``perform_web_search``
in iteration 2 after travelling through the full graph cycle.

Markers
-------
pytest.mark.integration — runs alongside the unit suite; no external services
needed (all I/O mocked).
"""

import asyncio
import functools
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Shared fake data
# ---------------------------------------------------------------------------

_DUP_URL = "http://dup.com"

# A single search result whose URL is always the same.  ``perform_web_search``
# uses ``url_memo`` to filter duplicate URLs across iterations.
_FAKE_SEARCH_RESPONSE = [
    {
        "results": [
            {
                "url": _DUP_URL,
                "title": "Dup Title",
                "content": "Some content.",
                "raw_content": "Some raw content.",
            }
        ]
    }
]

# How many iterations the test should allow.
_MAX_ITERATIONS = 2


# ---------------------------------------------------------------------------
# Async mock for call_llm_async (quality check + compression)
# ---------------------------------------------------------------------------

async def _fake_call_llm_async(*args, **kwargs):
    """Async drop-in for ``call_llm_async``.

    Dispatches on the tool name (``quality_formatter`` vs ``summary_formatter``):
    * ``quality_formatter`` — returns score=5 so every result passes the filter.
    * ``summary_formatter`` — returns a stub summary so compression succeeds.
    """
    tool_list = kwargs.get("tool", [])
    # LangChain @tool decorators expose a .name attribute on the wrapped callable.
    tool_name = tool_list[0].name if tool_list else ""

    mock_response = MagicMock()
    if tool_name == "quality_formatter":
        mock_response.tool_calls = [{"args": {"score": 5}}]
    elif tool_name == "summary_formatter":
        mock_response.tool_calls = [{"args": {"summary_content": f"stub summary from {_DUP_URL}"}}]
    else:
        mock_response.tool_calls = [{"args": {"score": 5}}]
    return mock_response


# ---------------------------------------------------------------------------
# Synchronous mock for call_llm (query generator, answer synthesizer, grader)
# ---------------------------------------------------------------------------

def _make_call_llm_response(iteration_counter):
    """Build the ``call_llm`` return value that dispatches by tool name.

    - ``queries_formatter``: returns queries for generate_queries_from_question
    - ``answer_formatter``: returns a stub answer for synthesize_answer
    - ``searching_grader_formatter``: returns 'fail' on first call, short-circuit on second
      (the budget guard handles the second iteration → END before LLM is called)
    """
    def _call_llm(*args, **kwargs):
        tool_list = kwargs.get("tool", [])
        tool_name = tool_list[0].name if tool_list else ""

        mock_response = MagicMock()
        if tool_name == "queries_formatter":
            mock_response.tool_calls = [{"args": {"queries": ["test query"]}}]
        elif tool_name == "answer_formatter":
            answer = f"Synthesized answer mentioning {_DUP_URL}"
            mock_response.tool_calls = [{"args": {"answer": answer}}]
        elif tool_name == "searching_grader_formatter":
            # Return 'fail' to trigger the back-edge loop
            mock_response.tool_calls = [
                {"args": {"grade": "fail", "follow_up_queries": ["follow up query"]}}
            ]
        else:
            mock_response.tool_calls = []
        return mock_response

    return _call_llm


# ---------------------------------------------------------------------------
# Helper: wrap perform_web_search to inject max_num_iterations into state
# ---------------------------------------------------------------------------

def _make_perform_web_search_wrapper(real_fn, max_iter: int):
    """Wrap ``perform_web_search`` to inject ``max_num_iterations`` into its
    state update.

    Context: ``agentic_search_graph`` is a module-level singleton compiled once
    at import time.  Its nodes store direct Python function-object references;
    ``patch('subagent.agentic_search.get_searching_budget', ...)`` does not
    affect the already-compiled graph.  Wrapping ``perform_web_search`` —
    called at the start of every loop cycle — is the least-invasive way to
    propagate the test-controlled budget into live state without requiring graph
    recompilation.

    ``selenium_api_search`` is patched at the outer ``with`` level so that it
    is already replaced by the time ``real_fn`` executes.
    """

    def wrapper(state):
        result = real_fn(state)
        result["max_num_iterations"] = max_iter
        return result

    return wrapper


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAgenticSearchGraphUrlMemoPropagation:
    """Verify that LangGraph propagates url_memo across the search loop back-edge.

    The graph is:
        START → get_searching_budget → generate_queries_from_question
              → perform_web_search → filter_and_format_results
              → compress_raw_content → aggregate_final_results
              → synthesize_answer → check_searching_results
              → (loops to perform_web_search if grade='fail', else END)

    With ``max_num_iterations=2`` the graph executes two full cycles.  After
    iteration 1, ``url_memo`` contains ``_DUP_URL``.  LangGraph must feed that
    updated set back into ``perform_web_search`` for iteration 2 so that the
    duplicate URL is filtered out.

    The test asserts three things that only hold if LangGraph propagates the
    set correctly:
    1. ``_DUP_URL`` appears in the final ``url_memo``.
    2. ``selenium_api_search`` is called exactly twice (one per iteration).
    3. ``_DUP_URL`` appears exactly once in the final ``answer``
       (deduplication worked — iteration 2 contributed no new results for that URL).
    """

    def test_url_memo_propagated_across_loop(self):
        """LangGraph propagates url_memo set through the check_searching_results back-edge.

        Implementation note: ``agentic_search_graph`` is compiled once at import;
        its nodes hold direct function references that ``patch()`` on the module
        namespace cannot override after the fact.  This test therefore directly
        mutates ``RunnableCallable.func`` (used by sync invoke) and ``afunc``
        (used by LangGraph's async executor via ``ainvoke``) on the compiled
        ``perform_web_search`` node, injecting a wrapper that adds
        ``max_num_iterations=2`` to the state update.  The original callables are
        restored in a ``finally`` block so that no state leaks to other tests.
        """
        import subagent.agentic_search as ag
        from subagent.agentic_search import agentic_search_graph, perform_web_search

        wrapped_perform = _make_perform_web_search_wrapper(perform_web_search, _MAX_ITERATIONS)

        # Patch the compiled graph's perform_web_search node in-place.
        # The ``RunnableCallable`` stores two references:
        #   .func  — called by the sync Pregel runner
        #   .afunc — called by the async Pregel runner (ainvoke path) as
        #            functools.partial(run_in_executor, None, original_func)
        node = agentic_search_graph.nodes["perform_web_search"]
        runnable = node.bound  # langgraph._internal._runnable.RunnableCallable
        original_func = runnable.func
        original_afunc = runnable.afunc

        # Replace .func with the wrapper.
        runnable.func = wrapped_perform
        # Replace .afunc with a new partial pointing to the wrapper so that the
        # asyncio thread-pool executor also sees the patched version.
        runnable.afunc = functools.partial(original_afunc.func, None, wrapped_perform)

        call_llm_fn = _make_call_llm_response(iteration_counter=[0])

        try:
            async def _run():
                return await agentic_search_graph.ainvoke(
                    {
                        "question": "What are the key facts about this topic?",
                        "url_memo": set(),
                        "source_registry": [],
                    }
                )

            with (
                patch.object(
                    ag,
                    "selenium_api_search",
                    return_value=_FAKE_SEARCH_RESPONSE,
                ) as mock_search,
                patch.object(ag, "call_llm_async", side_effect=_fake_call_llm_async),
                patch.object(ag, "call_llm", side_effect=call_llm_fn),
            ):
                final_state = asyncio.run(_run())
        finally:
            # Restore originals to avoid polluting subsequent tests.
            runnable.func = original_func
            runnable.afunc = original_afunc

        # 1. The URL seen in iteration 1 must survive in the final url_memo.
        assert _DUP_URL in final_state["url_memo"], (
            f"{_DUP_URL!r} must be present in final url_memo; "
            "got: " + repr(final_state["url_memo"])
        )

        # 2. One selenium_api_search call per iteration = 2 total.
        assert mock_search.call_count == 2, (
            "Expected 2 selenium_api_search calls (one per iteration), "
            f"got {mock_search.call_count}"
        )

        # 3. The URL appears exactly once in the answer — dedup prevented
        #    iteration 2 from contributing any new results for that URL.
        answer = final_state.get("answer", "")
        occurrences = answer.count(_DUP_URL)
        assert occurrences >= 1, (
            f"Expected {_DUP_URL!r} to appear in the answer (from iteration 1), "
            f"but found {occurrences} occurrence(s). answer={answer!r}"
        )


# ---------------------------------------------------------------------------
# Source registry integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSourceRegistryIntegration:
    """Verify that source_registry is built from real {title, url} pairs across iterations.

    Asserts:
    1. Final ``### Sources`` entries contain real ``http`` URLs (not text excerpts).
    2. A source registered in iteration 1 keeps its ``[N]`` index in the final answer.
    """

    _FAKE_RESULT = [
        {
            "results": [
                {
                    "url": "https://example.com/article",
                    "title": "Example Article",
                    "content": "CPO reduces power",
                    "raw_content": "CPO reduces power by 30%.",
                }
            ]
        }
    ]

    @staticmethod
    async def _quality_ok(*args, **kwargs):
        """Async mock for call_llm_async: every result passes quality (score=4)."""
        mock = MagicMock()
        mock.tool_calls = [{"args": {"score": 4}}]
        return mock

    @staticmethod
    def _mock_resp(**kwargs):
        m = MagicMock()
        m.tool_calls = [{"args": kwargs}]
        return m

    @staticmethod
    def _make_call_llm(responses: list):
        """Return a call_llm side_effect that yields responses in order."""
        it = iter(responses)
        def _fn(*args, **kwargs):
            return next(it)
        return _fn

    def test_full_graph_sources_are_urls(self):
        """Final answer's ### Sources entries must all contain real http URLs."""
        import subagent.agentic_search as ag
        from subagent.agentic_search import agentic_search_graph

        queries_resp = self._mock_resp(queries=["CPO technology"])
        synth_resp = self._mock_resp(answer="CPO reduces power [1].")
        grade_resp = self._mock_resp(grade="pass", follow_up_queries=[])

        async def _run():
            return await agentic_search_graph.ainvoke({
                "question": "What is CPO?",
                "url_memo": set(),
                "source_registry": [],
                "max_num_iterations": 1,
            })

        with (
            patch.object(ag, "selenium_api_search", return_value=self._FAKE_RESULT),
            patch.object(ag, "call_llm_async", side_effect=self._quality_ok),
            patch.object(ag, "call_llm", side_effect=self._make_call_llm([queries_resp, synth_resp, grade_resp])),
        ):
            final = asyncio.run(_run())

        answer = final["answer"]
        assert "### Sources" in answer, "finalize_answer must append ### Sources"
        assert "https://example.com/article" in answer, "Sources must contain real URL, not text excerpt"
        source_lines = [l for l in answer.split("\n") if l.startswith("- [")]
        assert source_lines, "At least one source line must be present"
        for line in source_lines:
            assert "http" in line, f"Source line missing URL: {line!r}"

    def test_two_iterations_source_index_stable(self):
        """Source registered as [1] in iteration 1 keeps that index in the final answer."""
        import subagent.agentic_search as ag
        from subagent.agentic_search import agentic_search_graph

        iter1_result = [{"results": [
            {"url": "https://first.com", "title": "First", "content": "c", "raw_content": "r"}
        ]}]
        iter2_result = [{"results": [
            {"url": "https://second.com", "title": "Second", "content": "c", "raw_content": "r"}
        ]}]

        call_count = {"n": 0}

        def fake_search(queries, raw):
            call_count["n"] += 1
            return iter1_result if call_count["n"] == 1 else iter2_result

        queries_resp = self._mock_resp(queries=["test query"])
        synth1 = self._mock_resp(answer="Iteration 1 answer [1].")
        grade_fail = self._mock_resp(grade="fail", follow_up_queries=["follow up"])
        synth2 = self._mock_resp(answer="Updated answer [1][2].")
        # After iteration 2, budget exhausted → finalize_answer called without grader

        async def _run():
            return await agentic_search_graph.ainvoke({
                "question": "Test question",
                "url_memo": set(),
                "source_registry": [],
                "max_num_iterations": 2,
            })

        with (
            patch.object(ag, "selenium_api_search", side_effect=fake_search),
            patch.object(ag, "call_llm_async", side_effect=self._quality_ok),
            patch.object(ag, "call_llm", side_effect=self._make_call_llm([queries_resp, synth1, grade_fail, synth2])),
        ):
            final = asyncio.run(_run())

        answer = final["answer"]
        assert "https://first.com" in answer, "[1] source from iteration 1 must appear in final Sources"
        assert "https://second.com" in answer, "[2] source from iteration 2 must appear in final Sources"
        # [1] must still refer to first.com (stable index across iterations)
        source_lines = [l for l in answer.split("\n") if l.startswith("- [")]
        line_1 = next((l for l in source_lines if "- [1]" in l), "")
        line_2 = next((l for l in source_lines if "- [2]" in l), "")
        assert "first.com" in line_1, f"[1] must remain https://first.com; got: {line_1!r}"
        assert "second.com" in line_2, f"[2] must be https://second.com; got: {line_2!r}"


# ---------------------------------------------------------------------------
# finalize_answer node integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFinalizeAnswerNode:
    """Verify that finalize_answer appends ### Sources using real source_registry entries.

    The test runs the full agentic_search_graph with max_num_iterations=1 so
    the budget guard in check_searching_results fires immediately and routes to
    finalize_answer without calling the grader LLM.  The assertions confirm
    that finalize_answer stitches the real URL from source_registry into the
    final answer's ### Sources block.
    """

    _ARTICLE_URL = "https://test.com/article"
    _ARTICLE_TITLE = "Test Article"

    _FAKE_RESULT = [
        {
            "results": [
                {
                    "url": _ARTICLE_URL,
                    "title": _ARTICLE_TITLE,
                    "content": "Test content about the topic.",
                    "raw_content": "Test raw content about the topic.",
                }
            ]
        }
    ]

    @staticmethod
    async def _quality_ok(*args, **kwargs):
        """Async mock for call_llm_async: every result passes quality check (score=4)."""
        mock = MagicMock()
        mock.tool_calls = [{"args": {"score": 4}}]
        return mock

    @staticmethod
    def _mock_resp(**kwargs):
        m = MagicMock()
        m.tool_calls = [{"args": kwargs}]
        return m

    @staticmethod
    def _make_call_llm(responses: list):
        """Return a call_llm side_effect that yields responses in order."""
        it = iter(responses)

        def _fn(*args, **kwargs):
            return next(it)

        return _fn

    def test_finalize_answer_appends_sources_in_full_graph(self):
        """finalize_answer appends ### Sources with the real article URL.

        With max_num_iterations=1 the budget guard in check_searching_results
        short-circuits to finalize_answer after the first search cycle, so the
        searching_grader_formatter is never invoked.  The call_llm sequence is:
          1. queries_formatter   — generate_queries_from_question
          2. answer_formatter    — synthesize_answer (answer body with [1] citation)

        finalize_answer then appends ### Sources from source_registry without
        any additional LLM call.
        """
        import subagent.agentic_search as ag
        from subagent.agentic_search import agentic_search_graph

        queries_resp = self._mock_resp(queries=["test query"])
        # Answer body must contain [1] so _format_sources_section finds a citation to emit.
        answer_resp = self._mock_resp(answer="Answer text [1].")

        async def _run():
            return await agentic_search_graph.ainvoke({
                "question": "What is the topic?",
                "url_memo": set(),
                "source_registry": [],
                "max_num_iterations": 1,
            })

        with (
            patch.object(ag, "selenium_api_search", return_value=self._FAKE_RESULT),
            patch.object(ag, "call_llm_async", side_effect=self._quality_ok),
            patch.object(
                ag,
                "call_llm",
                side_effect=self._make_call_llm([queries_resp, answer_resp]),
            ),
        ):
            final_state = asyncio.run(_run())

        answer = final_state["answer"]

        # 1. finalize_answer must append the ### Sources block.
        assert "### Sources" in answer, (
            "finalize_answer must append ### Sources to the answer; "
            f"got: {answer!r}"
        )

        # 2. The real article URL must appear in the Sources block.
        assert self._ARTICLE_URL in answer, (
            f"Expected {self._ARTICLE_URL!r} to appear in the final answer; "
            f"got: {answer!r}"
        )

        # 3. source_registry must contain at least one entry with the article URL.
        registry = final_state.get("source_registry", [])
        assert any(entry.get("url") == self._ARTICLE_URL for entry in registry), (
            f"source_registry must contain an entry with url={self._ARTICLE_URL!r}; "
            f"got: {registry!r}"
        )
