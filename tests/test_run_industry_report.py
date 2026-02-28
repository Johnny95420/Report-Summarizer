"""Integration test for the industry report generation pipeline.

Runs the full LangGraph workflow end-to-end with all LLM calls and external
APIs mocked out, verifying that:
  - The graph builds and compiles without error
  - Planning → interrupt → resume → section writing → final report flow works
  - The final report contains content from every section
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock LLM helpers
# ---------------------------------------------------------------------------


def _make_llm_response(tool=None, **extra):
    """Build a fake LLM response whose shape matches the caller's expectation.

    The *tool* list (LangChain tool objects) is inspected by name to decide
    which response shape to return.
    """
    resp = MagicMock()

    if tool:
        tool_names = {t.name for t in tool}

        if "queries_formatter" in tool_names:
            resp.tool_calls = [{"args": {"thought": "mock", "queries": ["mock query 1"]}}]

        elif "question_formatter" in tool_names:
            resp.tool_calls = [
                {
                    "args": {
                        "question": (
                            "Main Question: What are the key aspects of this topic?\n"
                            "- Sub-question 1: What are the main drivers?\n"
                            "- Sub-question 2: What are the key risks?"
                        )
                    }
                }
            ]

        elif "section_formatter" in tool_names:
            resp.tool_calls = [
                {
                    "args": {
                        "name": "Introduction",
                        "description": "Report introduction and overview",
                        "research": False,
                        "content": "",
                    }
                },
                {
                    "args": {
                        "name": "Analysis",
                        "description": "Main analysis of the topic",
                        "research": True,
                        "content": "",
                    }
                },
            ]

        elif "feedback_formatter" in tool_names:
            resp.tool_calls = [{"args": {"grade": "pass", "weakness": ""}}]

        elif "content_refinement_formatter" in tool_names:
            resp.tool_calls = [{"args": {"refined_content": "Refined mock content."}}]

        else:
            resp.content = "Mock LLM output."
    else:
        # Plain text generation (section writing, final section writing)
        resp.content = "Mock generated section content."

    return resp


def _mock_call_llm(_model, _backup, prompt, *, tool=None, tool_choice=None):
    return _make_llm_response(tool=tool)


async def _mock_call_llm_async(_model, _backup, prompt, *, tool=None, tool_choice=None):
    return _make_llm_response(tool=tool)


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_industry_report_end_to_end():
    """Run the full report graph with mocked LLM / search and verify output."""

    patches = {
        # LLM
        "report_writer.call_llm": _mock_call_llm,
        "report_writer.call_llm_async": _mock_call_llm_async,
        # Token counting — always return a small number
        "report_writer.get_num_tokens": lambda *a, **kw: 100,
        # Web search used in _perform_planner_search
        "report_writer.call_search_engine": lambda *a, **kw: [],
        "report_writer.web_search_deduplicate_and_format_sources": lambda *a, **kw: "mock web results",
        # Agentic search subgraph used in orchestration
        "report_writer.agentic_search_graph": MagicMock(
            ainvoke=AsyncMock(return_value={"answer": "Mock agentic search answer with [1] citations.\n\n### Sources\n[1] Mock Source — http://mock.com"})
        ),
    }

    with patch.multiple("report_writer", **{k.split(".")[-1]: v for k, v in patches.items()}):
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        from langgraph.types import Command

        from report_writer import ReportGraphBuilder
        from State.state import ReportStateInput

        async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
            builder = ReportGraphBuilder(async_checkpointer=checkpointer)
            graph = builder.get_async_graph()

            config = {
                "configurable": {
                    "thread_id": "integration-test",
                    "planner_search_queries": 2,
                    "use_web": True,
                    "use_local_db": False,
                    "max_search_depth": 1,
                    "agentic_search_iterations": 1,
                    "agentic_search_queries": 3,
                    "refine_iteration": 0,
                    "report_structure": "default",
                    "recursion_limit": 100,
                }
            }

            input_data = ReportStateInput(topic="Test topic")

            # Phase 1: run until human_feedback interrupt
            interrupted = False
            async for event in graph.astream(input_data, config, stream_mode="updates"):
                if "__interrupt__" in event:
                    interrupted = True
                    break

            assert interrupted, "Expected graph to interrupt for human feedback"

            # Phase 2: resume with approval
            final_report = None
            async for event in graph.astream(Command(resume=True), config, stream_mode="updates"):
                if "compile_final_report" in event:
                    final_report = event["compile_final_report"].get("final_report")

            assert final_report is not None, "compile_final_report should produce final_report"
            assert len(final_report) > 0, "final_report should not be empty"
            # Both sections should have contributed content
            assert "Mock" in final_report or "Refined" in final_report
