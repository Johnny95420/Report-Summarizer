"""Tests for subagent.document_qa — route_response, extract_answer_node, _build_iteration_reminder."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from subagent.document_qa import (
    _build_iteration_reminder,
    _validate_inputs,
    agent_node,
    extract_answer_node,
    force_answer_node,
    route_response,
    run_document_qa,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_state(last_msg, iteration=5, budget=50):
    return {
        "messages": [HumanMessage(content="question"), last_msg],
        "iteration": iteration,
        "budget": budget,
        "question": "test",
        "answer": "",
        "file_paths": [{"name": "doc", "path": "/tmp/doc.json"}],
    }


def _ai_with_tool_calls(tool_calls):
    """Create an AIMessage with tool_calls attribute."""
    msg = AIMessage(content="thinking...")
    msg.tool_calls = tool_calls
    return msg


# ---------------------------------------------------------------------------
# route_response: all 4 branches
# ---------------------------------------------------------------------------
class TestRouteResponse:
    def test_submit_answer_detected(self):
        msg = _ai_with_tool_calls([{"name": "submit_answer", "args": {"answer": "ans"}, "id": "1"}])
        state = _make_state(msg)
        assert route_response(state) == "extract_answer"

    def test_budget_exhausted(self):
        msg = _ai_with_tool_calls([{"name": "go_to_page", "args": {"page": 0}, "id": "2"}])
        state = _make_state(msg, iteration=50, budget=50)
        # submit_answer not in tool_calls, so check budget next
        # But actually the msg has tool_calls with go_to_page, not submit_answer
        # iteration >= budget → force_answer
        assert route_response(state) == "force_answer"

    def test_has_tool_calls(self):
        msg = _ai_with_tool_calls([{"name": "go_to_page", "args": {"page": 0}, "id": "3"}])
        state = _make_state(msg, iteration=5, budget=50)
        assert route_response(state) == "tools"

    def test_text_only_planning(self):
        msg = AIMessage(content="Let me plan...")
        state = _make_state(msg, iteration=5, budget=50)
        assert route_response(state) == "agent"

    def test_non_ai_message(self):
        msg = HumanMessage(content="user msg")
        state = _make_state(msg)
        assert route_response(state) == "agent"

    def test_no_tool_calls_attribute(self):
        """AIMessage without tool_calls attr should not crash (C6 fix)."""
        msg = AIMessage(content="response without tools")
        # Ensure no tool_calls attribute at all
        if hasattr(msg, "tool_calls"):
            delattr(msg, "tool_calls")
        state = _make_state(msg)
        # Should fall through to "agent" (text-only)
        result = route_response(state)
        assert result in ("agent",)

    def test_consecutive_errors_forces_answer(self):
        """3+ consecutive LLM errors should route to force_answer (C2 fix)."""
        msg = AIMessage(content="[LLM failed 3 times]")
        state = _make_state(msg, iteration=5, budget=50)
        state["consecutive_errors"] = 3
        assert route_response(state) == "force_answer"

    def test_consecutive_errors_below_threshold(self):
        """< 3 consecutive errors should not force answer."""
        msg = AIMessage(content="[LLM error]")
        state = _make_state(msg, iteration=5, budget=50)
        state["consecutive_errors"] = 2
        # No tool calls, no submit_answer → should be "agent"
        assert route_response(state) == "agent"


# ---------------------------------------------------------------------------
# extract_answer_node
# ---------------------------------------------------------------------------
class TestExtractAnswerNode:
    def test_primary_path_submit_answer(self):
        msg = _ai_with_tool_calls([{"name": "submit_answer", "args": {"answer": "Final answer"}, "id": "a1"}])
        state = _make_state(msg)
        result = extract_answer_node(state)
        assert result["answer"] == "Final answer"
        assert len(result["messages"]) == 1  # ToolMessage

    def test_fallback_text_content(self):
        msg = AIMessage(content="Fallback text answer")
        state = _make_state(msg)
        result = extract_answer_node(state)
        assert result["answer"] == "Fallback text answer"
        assert result["messages"] == []

    def test_fallback_list_content(self):
        msg = AIMessage(content=[{"type": "text", "text": "block answer"}])
        state = _make_state(msg)
        result = extract_answer_node(state)
        assert "block answer" in result["answer"]

    def test_empty_answer_raises(self):
        msg = AIMessage(content="")
        state = _make_state(msg)
        with pytest.raises(RuntimeError, match="Failed to extract any answer"):
            extract_answer_node(state)

    def test_only_first_submit_answer_used(self):
        """If LLM emits two submit_answer calls, only the first is used."""
        msg = _ai_with_tool_calls(
            [
                {"name": "submit_answer", "args": {"answer": "First answer"}, "id": "a1"},
                {"name": "submit_answer", "args": {"answer": "Second answer"}, "id": "a2"},
            ]
        )
        state = _make_state(msg)
        result = extract_answer_node(state)
        assert result["answer"] == "First answer"


# ---------------------------------------------------------------------------
# _build_iteration_reminder
# ---------------------------------------------------------------------------
class TestBuildIterationReminder:
    def test_urgent_when_remaining_lte_5(self):
        result = _build_iteration_reminder(iteration=46, budget=50)
        assert "URGENT" in result

    def test_urgent_takes_priority_over_progress_check(self):
        """When remaining <= 5 AND iteration % 10 == 0, URGENT should win (I4 fix)."""
        # iteration=50, budget=55 → remaining=5, and 50 % 10 == 0
        result = _build_iteration_reminder(iteration=50, budget=55)
        assert "URGENT" in result

    def test_progress_check_at_mod_10(self):
        result = _build_iteration_reminder(iteration=20, budget=50)
        assert "Progress Check" in result

    def test_normal_reminder(self):
        result = _build_iteration_reminder(iteration=7, budget=50)
        assert "7/50" in result
        assert "URGENT" not in result
        assert "Progress Check" not in result

    def test_remaining_zero(self):
        result = _build_iteration_reminder(iteration=50, budget=50)
        assert "URGENT" in result
        assert "0 calls remaining" in result

    def test_progress_check_wins_when_remaining_exactly_6(self):
        """remaining=6 is outside urgent zone; iteration % 10 == 0 → Progress Check."""
        result = _build_iteration_reminder(iteration=50, budget=56)
        assert "Progress Check" in result
        assert "URGENT" not in result


# ---------------------------------------------------------------------------
# _validate_inputs
# ---------------------------------------------------------------------------
class TestValidateInputs:
    def test_valid(self):
        _validate_inputs([{"name": "a", "path": "/a"}], "question?", 10)

    def test_empty_file_paths(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_inputs([], "question?", 10)

    def test_missing_keys(self):
        with pytest.raises(ValueError, match="'name' and 'path'"):
            _validate_inputs([{"name": "a"}], "question?", 10)

    def test_empty_question(self):
        with pytest.raises(ValueError, match="non-empty string"):
            _validate_inputs([{"name": "a", "path": "/a"}], "", 10)

    def test_zero_budget(self):
        with pytest.raises(ValueError, match="budget must be > 0"):
            _validate_inputs([{"name": "a", "path": "/a"}], "question?", 0)

    def test_whitespace_only_question(self):
        with pytest.raises(ValueError, match="non-empty string"):
            _validate_inputs([{"name": "a", "path": "/a"}], "   ", 10)

    def test_missing_name_key(self):
        with pytest.raises(ValueError, match="'name' and 'path'"):
            _validate_inputs([{"path": "/a"}], "question?", 10)


# ---------------------------------------------------------------------------
# agent_node: consecutive_errors tracking (C2)
# ---------------------------------------------------------------------------
class TestAgentNodeConsecutiveErrors:
    def test_increments_on_failure(self):
        """agent_node should increment consecutive_errors when LLM fails."""
        state = _make_state(AIMessage(content=""), iteration=0, budget=50)
        state["consecutive_errors"] = 1
        config = {"configurable": {"tools": []}}
        with patch("subagent.document_qa.call_llm", side_effect=Exception("fail")):
            result = agent_node(state, config)
        assert result["consecutive_errors"] == 2

    def test_resets_on_success(self):
        """agent_node should reset consecutive_errors to 0 on LLM success."""
        state = _make_state(AIMessage(content=""), iteration=0, budget=50)
        state["consecutive_errors"] = 2
        config = {"configurable": {"tools": []}}
        with patch("subagent.document_qa.call_llm", return_value=AIMessage(content="ok")):
            result = agent_node(state, config)
        assert result["consecutive_errors"] == 0

    def test_forces_message_at_threshold(self):
        """At 3 consecutive errors, agent_node should emit a 'forcing answer' message."""
        state = _make_state(AIMessage(content=""), iteration=0, budget=50)
        state["consecutive_errors"] = 2  # will become 3 after this failure
        config = {"configurable": {"tools": []}}
        with patch("subagent.document_qa.call_llm", side_effect=Exception("fail")):
            result = agent_node(state, config)
        assert result["consecutive_errors"] == 3
        assert "Forcing answer" in result["messages"][0].content


# ---------------------------------------------------------------------------
# force_answer_node: RuntimeError on LLM failure (C3)
# ---------------------------------------------------------------------------
class TestForceAnswerNode:
    def test_llm_failure_raises_runtime_error(self):
        """force_answer_node should raise RuntimeError when LLM fails (C3 fix)."""
        state = _make_state(AIMessage(content=""), iteration=50, budget=50)
        with (
            patch("subagent.document_qa.call_llm", side_effect=Exception("network error")),
            pytest.raises(RuntimeError, match="Force answer LLM call failed"),
        ):
            force_answer_node(state)

    def test_success_returns_messages(self):
        """force_answer_node should return LLM response on success."""
        state = _make_state(AIMessage(content=""), iteration=50, budget=50)
        mock_response = AIMessage(content="forced answer")
        mock_response.tool_calls = [{"name": "submit_answer", "args": {"answer": "ans"}, "id": "1"}]
        with patch("subagent.document_qa.call_llm", return_value=mock_response):
            result = force_answer_node(state)
        assert len(result["messages"]) == 1


# ---------------------------------------------------------------------------
# run_document_qa: answer validation (C5)
# ---------------------------------------------------------------------------
class TestRunDocumentQA:
    def test_reraises_runtime_error_from_graph(self):
        """run_document_qa should re-raise RuntimeError from graph.invoke (C5 fix)."""
        with (
            patch("subagent.document_qa.build_document_qa_graph") as mock_build,
            patch("subagent.document_qa.AgentDocumentReader"),
        ):
            mock_graph = MagicMock()
            mock_graph.invoke.side_effect = RuntimeError("graph crash")
            mock_build.return_value = mock_graph
            with pytest.raises(RuntimeError, match="Document QA failed"):
                run_document_qa([{"name": "a", "path": "/a"}], "question?", 10)

    def test_raises_on_empty_answer(self):
        """run_document_qa should raise RuntimeError when answer is whitespace-only (C5 fix)."""
        with (
            patch("subagent.document_qa.build_document_qa_graph") as mock_build,
            patch("subagent.document_qa.AgentDocumentReader"),
        ):
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {"answer": "   "}
            mock_build.return_value = mock_graph
            with pytest.raises(RuntimeError, match="empty answer"):
                run_document_qa([{"name": "a", "path": "/a"}], "question?", 10)

    def test_returns_valid_answer(self):
        """run_document_qa should return the answer when it's valid."""
        with (
            patch("subagent.document_qa.build_document_qa_graph") as mock_build,
            patch("subagent.document_qa.AgentDocumentReader"),
        ):
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {"answer": "Valid answer here"}
            mock_build.return_value = mock_graph
            result = run_document_qa([{"name": "a", "path": "/a"}], "question?", 10)
        assert result == "Valid answer here"
