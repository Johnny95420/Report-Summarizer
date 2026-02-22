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
        mock_response.tool_calls = [{"args": {"summary_content": "stub summary"}}]
    else:
        mock_response.tool_calls = [{"args": {"score": 5}}]
    return mock_response


# ---------------------------------------------------------------------------
# Synchronous mock for call_llm (grader in check_searching_results)
# ---------------------------------------------------------------------------

def _make_call_llm_fail_response():
    """Build the ``call_llm`` return value that makes the grader return 'fail'.

    ``check_searching_results`` accesses ``feedback.tool_calls[0]['args']``.
    Returning grade='fail' causes the graph to loop back to perform_web_search.
    After iteration 2 the budget guard short-circuits before calling the LLM,
    so this mock is exercised exactly once.
    """
    mock_response = MagicMock()
    mock_response.tool_calls = [
        {"args": {"grade": "fail", "follow_up_queries": ["follow up query"]}}
    ]
    return mock_response


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
        START → get_searching_budget → perform_web_search
              → filter_and_format_results → compress_raw_content
              → aggregate_final_results → check_searching_results
              → (loops to perform_web_search if grade='fail', else END)

    With ``max_num_iterations=2`` the graph executes two full cycles.  After
    iteration 1, ``url_memo`` contains ``_DUP_URL``.  LangGraph must feed that
    updated set back into ``perform_web_search`` for iteration 2 so that the
    duplicate URL is filtered out and not added to ``web_results`` again.

    The test asserts three things that only hold if LangGraph propagates the
    set correctly:
    1. ``_DUP_URL`` appears in the final ``url_memo``.
    2. ``selenium_api_search`` is called exactly twice (one per iteration).
    3. ``_DUP_URL`` appears exactly once in the final ``source_str``
       (deduplication worked — iteration 2 contributed no new results).
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

        call_llm_response = _make_call_llm_fail_response()

        try:
            async def _run():
                return await agentic_search_graph.ainvoke(
                    {"queries": ["test query"], "url_memo": set()}
                )

            with (
                patch.object(
                    ag,
                    "selenium_api_search",
                    return_value=_FAKE_SEARCH_RESPONSE,
                ) as mock_search,
                patch.object(ag, "call_llm_async", side_effect=_fake_call_llm_async),
                patch.object(ag, "call_llm", return_value=call_llm_response),
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

        # 3. The URL appears exactly once in source_str — dedup prevented
        #    iteration 2 from contributing any new results for that URL.
        source_str = final_state.get("source_str", "")
        occurrences = source_str.count(_DUP_URL)
        assert occurrences == 1, (
            f"Expected {_DUP_URL!r} to appear exactly once in source_str "
            f"(dedup via url_memo), but found {occurrences} occurrence(s)."
        )
