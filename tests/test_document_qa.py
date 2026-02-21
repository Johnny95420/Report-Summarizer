"""Tests for subagent.document_qa — route_response, extract_answer_node, _build_iteration_reminder."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from subagent.document_qa import (
    _build_iteration_reminder,
    _validate_inputs,
    extract_answer_node,
    route_response,
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

    def test_empty_answer(self):
        msg = AIMessage(content="")
        state = _make_state(msg)
        result = extract_answer_node(state)
        assert result["answer"] == ""

    def test_only_first_submit_answer_used(self):
        """If LLM emits two submit_answer calls, only the first is used."""
        msg = _ai_with_tool_calls([
            {"name": "submit_answer", "args": {"answer": "First answer"}, "id": "a1"},
            {"name": "submit_answer", "args": {"answer": "Second answer"}, "id": "a2"},
        ])
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
