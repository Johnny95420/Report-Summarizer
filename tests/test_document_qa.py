"""Tests for subagent.document_qa — route_response, extract_answer_node, _build_iteration_reminder."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from subagent.document_qa import (
    _build_iteration_reminder,
    _prepare_qa_session,
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

    def test_empty_answer_returns_sentinel(self):
        """extract_answer_node should return sentinel on empty content (C6 fix)."""
        msg = AIMessage(content="")
        state = _make_state(msg)
        result = extract_answer_node(state)
        assert result["answer"] == "[Unable to extract answer from documents]"

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
# force_answer_node: degraded answer on LLM failure (C1)
# ---------------------------------------------------------------------------
class TestForceAnswerNode:
    def test_llm_failure_returns_degraded_answer(self):
        """force_answer_node should return synthetic submit_answer on LLM failure (C1 fix)."""
        state = _make_state(AIMessage(content=""), iteration=50, budget=50)
        with patch("subagent.document_qa.call_llm", side_effect=Exception("network error")):
            result = force_answer_node(state)
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert isinstance(msg, AIMessage)
        assert msg.tool_calls
        assert msg.tool_calls[0]["name"] == "submit_answer"
        assert msg.tool_calls[0]["args"]["answer"]  # non-empty fallback answer

    def test_success_returns_messages(self):
        """force_answer_node should return LLM response on success."""
        state = _make_state(AIMessage(content=""), iteration=50, budget=50)
        mock_response = AIMessage(content="forced answer")
        mock_response.tool_calls = [{"name": "submit_answer", "args": {"answer": "ans"}, "id": "1"}]
        with patch("subagent.document_qa.call_llm", return_value=mock_response):
            result = force_answer_node(state)
        assert len(result["messages"]) == 1


# ---------------------------------------------------------------------------
# run_document_qa: answer validation and error propagation
# ---------------------------------------------------------------------------
class TestRunDocumentQA:
    def test_graph_error_propagates(self):
        """run_document_qa should propagate errors from graph.invoke."""
        with (
            patch("subagent.document_qa.build_document_qa_graph") as mock_build,
            patch("subagent.document_qa.AgentDocumentReader"),
        ):
            mock_graph = MagicMock()
            mock_graph.invoke.side_effect = RuntimeError("graph crash")
            mock_build.return_value = mock_graph
            with pytest.raises(RuntimeError, match="graph crash"):
                run_document_qa([{"name": "a", "path": "/a"}], "question?", 10)

    def test_raises_on_empty_answer(self):
        """run_document_qa should raise RuntimeError when answer is whitespace-only."""
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


# ---------------------------------------------------------------------------
# run_document_qa: navigator cleanup (C2)
# ---------------------------------------------------------------------------
class TestRunDocumentQACleanup:
    def test_navigator_closed_on_success(self):
        """run_document_qa should close navigator even on success (C2 fix)."""
        with (
            patch("subagent.document_qa.build_document_qa_graph") as mock_build,
            patch("subagent.document_qa.AgentDocumentReader") as mock_navigator_cls,
        ):
            mock_navigator = MagicMock()
            mock_navigator_cls.return_value = mock_navigator
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {"answer": "Valid answer"}
            mock_build.return_value = mock_graph

            run_document_qa([{"name": "a", "path": "/a"}], "question?", 10)
            mock_navigator.close_document.assert_called_once()

    def test_navigator_closed_on_failure(self):
        """run_document_qa should close navigator even when graph fails (C2 fix)."""
        with (
            patch("subagent.document_qa.build_document_qa_graph") as mock_build,
            patch("subagent.document_qa.AgentDocumentReader") as mock_navigator_cls,
        ):
            mock_navigator = MagicMock()
            mock_navigator_cls.return_value = mock_navigator
            mock_graph = MagicMock()
            mock_graph.invoke.side_effect = RuntimeError("crash")
            mock_build.return_value = mock_graph

            with pytest.raises(RuntimeError):
                run_document_qa([{"name": "a", "path": "/a"}], "question?", 10)
            mock_navigator.close_document.assert_called_once()


# ---------------------------------------------------------------------------
# consecutive_text_only: degenerate loop detection (C4)
# ---------------------------------------------------------------------------
class TestConsecutiveTextOnly:
    def test_increments_on_text_only_response(self):
        """agent_node should increment consecutive_text_only when no tool calls (C4 fix)."""
        state = _make_state(AIMessage(content=""), iteration=0, budget=50)
        state["consecutive_text_only"] = 1
        state["consecutive_errors"] = 0
        config = {"configurable": {"tools": []}}
        with patch("subagent.document_qa.call_llm", return_value=AIMessage(content="planning...")):
            result = agent_node(state, config)
        assert result["consecutive_text_only"] == 2

    def test_resets_on_tool_call(self):
        """agent_node should reset consecutive_text_only when tool calls present (C4 fix)."""
        state = _make_state(AIMessage(content=""), iteration=0, budget=50)
        state["consecutive_text_only"] = 2
        state["consecutive_errors"] = 0
        config = {"configurable": {"tools": []}}
        mock_response = AIMessage(content="")
        mock_response.tool_calls = [{"name": "go_to_page", "args": {"page": 0}, "id": "x"}]
        with patch("subagent.document_qa.call_llm", return_value=mock_response):
            result = agent_node(state, config)
        assert result["consecutive_text_only"] == 0

    def test_route_to_force_answer_at_threshold(self):
        """route_response should route to force_answer when consecutive_text_only >= 3 (C4 fix)."""
        msg = AIMessage(content="still planning...")
        state = _make_state(msg, iteration=5, budget=50)
        state["consecutive_text_only"] = 3
        assert route_response(state) == "force_answer"

    def test_route_stays_agent_below_threshold(self):
        """route_response should stay in agent when consecutive_text_only < 3."""
        msg = AIMessage(content="planning...")
        state = _make_state(msg, iteration=5, budget=50)
        state["consecutive_text_only"] = 2
        assert route_response(state) == "agent"


# ---------------------------------------------------------------------------
# _prepare_qa_session: shared setup helper (I9)
# ---------------------------------------------------------------------------
class TestPrepareQASession:
    def _call(self, file_paths=None, question="What is X?", budget=10):
        if file_paths is None:
            file_paths = [{"name": "doc", "path": "/tmp/doc.json"}]
        with (
            patch("subagent.document_qa.AgentDocumentReader") as mock_nav_cls,
            patch("subagent.document_qa.build_document_qa_graph") as mock_build,
        ):
            mock_nav_cls.return_value.get_tools.return_value = []
            mock_build.return_value = MagicMock()
            return _prepare_qa_session(file_paths, question, budget)

    def test_initial_state_has_all_required_keys(self):
        _, _, initial_state, _ = self._call()
        required = {"file_paths", "question", "answer", "iteration", "budget",
                    "consecutive_errors", "consecutive_text_only", "messages"}
        assert required.issubset(initial_state.keys())

    def test_consecutive_text_only_initialized_to_0(self):
        _, _, initial_state, _ = self._call()
        assert initial_state["consecutive_text_only"] == 0

    def test_consecutive_errors_initialized_to_0(self):
        _, _, initial_state, _ = self._call()
        assert initial_state["consecutive_errors"] == 0

    def test_answer_initialized_to_empty_string(self):
        _, _, initial_state, _ = self._call()
        assert initial_state["answer"] == ""

    def test_iteration_initialized_to_0(self):
        _, _, initial_state, _ = self._call()
        assert initial_state["iteration"] == 0

    def test_budget_matches_input(self):
        _, _, initial_state, _ = self._call(budget=25)
        assert initial_state["budget"] == 25

    def test_invoke_config_has_configurable_with_tools(self):
        _, _, _, invoke_config = self._call()
        assert "configurable" in invoke_config
        assert "tools" in invoke_config["configurable"]

    def test_invoke_config_tools_includes_submit_answer(self):
        _, _, _, invoke_config = self._call()
        tool_names = [t.name for t in invoke_config["configurable"]["tools"]]
        assert "submit_answer" in tool_names
