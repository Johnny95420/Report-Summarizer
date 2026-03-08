# tests/test_auth_source_integration.py
"""Integration tests for the auth_source_search LangGraph graph.

All LLM calls and API calls are mocked — no external services required.
Tests run the full graph via asyncio.run(graph.ainvoke(...))
using the same execution model that run_auth_source_search uses.
"""

import asyncio
import json
import os
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_llm_response(tool_name: str, **args):
    resp = MagicMock()
    resp.tool_calls = [{"args": args}]
    return resp


def _llm_dispatcher(model, backup, prompt, *, tool=None, tool_choice=None):
    """Route call_llm to the right mock response based on tool name."""
    tool_names = {t.name for t in (tool or [])}
    if "sub_goal_formatter" in tool_names:
        return _make_llm_response("sub_goal_formatter", sub_goal="台積電 N3 良率 2024")
    if "download_queries_formatter" in tool_names:
        return _make_llm_response("download_queries_formatter", provider_a="台積電 N3", provider_b=None)
    if "reflect_download_formatter" in tool_names:
        return _make_llm_response("reflect_download_formatter", grade="pass", download_weakness="")
    if "reflect_qa_formatter" in tool_names:
        return _make_llm_response("reflect_qa_formatter", grade="pass", qa_weakness="")
    if "synthesis_formatter" in tool_names:
        return _make_llm_response("synthesis_formatter", merged_answer="合併答案")
    if "outer_reflect_formatter" in tool_names:
        return _make_llm_response("outer_reflect_formatter", grade="pass", hint="")
    if "document_selection_formatter" in tool_names:
        return _make_llm_response("document_selection_formatter", selected_names=["MockReport"])
    return MagicMock()


def _make_fake_provider_a_result(tmp_path, name="MockReport"):
    """Create a minimal BaseReaderDocument JSON file and return the download result string."""
    from langchain_core.documents import Document

    from Tools.reader_models import BaseReaderDocument

    doc = BaseReaderDocument(
        date="2024-01-01",
        name=name,
        outlines=[],
        pages=[Document(page_content=f"Content for {name}", metadata={"page_id": 0})],
    )
    path = str(tmp_path / f"{name}.json")
    doc.save(path)
    return json.dumps({"name": name, "path": path, "source": "provider_a"})


def _run_graph(
    tmp_path=None,
    initial_overrides=None,
    llm_fn=None,
    ia_fn=None,
    qa_answer="Mock QA answer",
):
    """Run the auth source graph synchronously with all I/O mocked."""
    import subagent.auth_source_search as asc
    from subagent.auth_source_search import build_auth_source_graph
    from Tools.text_navigator import AgentDocumentReader

    graph = build_auth_source_graph()

    initial_state = {
        "question": "台積電N3良率如何?",
        "max_pairs": 1,
        "max_download_reflections": 1,
        "max_qa_reflections": 1,
        "qa_budget": 5,
        "sub_goal": "",
        "sub_goal_history": [],
        "download_queries": {},
        "download_weakness": "",
        "download_reflection_count": 0,
        "downloaded_reports": [],
        "selected_reports": [],
        "curr_answer": "",
        "qa_weakness": "",
        "qa_reflection_count": 0,
        "navigator_state_path": "",
        "answer": "",
        "pair_count": 0,
    }
    if initial_overrides:
        initial_state.update(initial_overrides)

    nav = MagicMock(spec=AgentDocumentReader)
    nav.get_tools.return_value = []
    nav._bookmarks = {}
    nav._current_path = None

    fake_qa_graph = MagicMock()
    fake_qa_graph.invoke.return_value = {"answer": qa_answer}

    run_dir = str(tmp_path / "test_run") if tmp_path else "/tmp/test_run"
    config = {
        "configurable": {
            "shared_navigator": nav,
            "shared_pdf_converter": None,
            "run_dir": run_dir,
        }
    }
    default_ia = ia_fn or (lambda q, **kw: json.dumps({"error": "not_implemented", "source": "provider_a"}))

    with (
        patch.object(asc, "call_llm", side_effect=llm_fn or _llm_dispatcher),
        patch.object(asc, "build_document_qa_graph", return_value=fake_qa_graph),
        patch.object(asc, "download_provider_a_report", side_effect=default_ia),
        patch.object(
            asc,
            "download_provider_b_report",
            side_effect=lambda q, **kw: json.dumps({"error": "no_results", "source": "provider_b"}),
        ),
    ):
        return asyncio.run(graph.ainvoke(initial_state, config))


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAuthSourceGraphIntegration:
    def test_full_graph_single_pair_produces_answer(self, tmp_path):
        """Full graph with valid downloads produces a non-empty answer."""
        result = _run_graph(
            tmp_path=tmp_path,
            ia_fn=lambda q, **kw: _make_fake_provider_a_result(tmp_path),
            qa_answer="台積電N3良率達到業界水準。",
        )
        assert result["answer"] != ""
        assert result["pair_count"] == 1

    def test_download_failure_proceeds_to_qa(self, tmp_path):
        """All downloads fail -> qa_agent runs with empty file list; no exception raised."""
        result = _run_graph(
            tmp_path=tmp_path,
            ia_fn=lambda q, **kw: json.dumps({"error": "connection_failed", "source": "provider_a"}),
            qa_answer="",
        )
        assert result["pair_count"] == 1

    def test_reflect_download_retry_then_pass(self, tmp_path):
        """reflect_download fails once then passes; download_reflection_count reaches 1."""
        call_count = {"n": 0}

        def llm_retry(model, backup, prompt, *, tool=None, tool_choice=None):
            tool_names = {t.name for t in (tool or [])}
            if "reflect_download_formatter" in tool_names:
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return _make_llm_response("reflect_download_formatter", grade="fail", download_weakness="缺少數據")
                return _make_llm_response("reflect_download_formatter", grade="pass", download_weakness="")
            return _llm_dispatcher(model, backup, prompt, tool=tool, tool_choice=tool_choice)

        result = _run_graph(
            tmp_path=tmp_path,
            initial_overrides={"max_download_reflections": 2},
            ia_fn=lambda q, **kw: _make_fake_provider_a_result(tmp_path),
            llm_fn=llm_retry,
            qa_answer="答案在此",
        )
        assert call_count["n"] == 2
        assert result["answer"] != ""

    def test_max_pairs_stops_graph(self, tmp_path):
        """outer_reflect always fails; graph stops at max_pairs and returns current answer."""
        call_count = {"outer": 0}

        def always_fail_outer(model, backup, prompt, *, tool=None, tool_choice=None):
            tool_names = {t.name for t in (tool or [])}
            if "outer_reflect_formatter" in tool_names:
                call_count["outer"] += 1
                return _make_llm_response("outer_reflect_formatter", grade="fail", hint="still need more")
            return _llm_dispatcher(model, backup, prompt, tool=tool, tool_choice=tool_choice)

        result = _run_graph(
            tmp_path=tmp_path,
            initial_overrides={"max_pairs": 2},
            ia_fn=lambda q, **kw: _make_fake_provider_a_result(tmp_path),
            llm_fn=always_fail_outer,
            qa_answer="部分答案",
        )
        assert result["pair_count"] == 2

    def test_navigator_state_persisted_after_qa(self, tmp_path):
        """JSON sidecar is written to navigator_state_path after qa_agent completes."""
        nav_path = str(tmp_path / "nav_state.json")
        _run_graph(
            tmp_path=tmp_path,
            initial_overrides={"navigator_state_path": nav_path},
            ia_fn=lambda q, **kw: _make_fake_provider_a_result(tmp_path),
            qa_answer="答案在此",
        )
        assert os.path.exists(nav_path)

    def test_cookie_not_set_both_sources_no_crash(self, tmp_path):
        """Both tools return cookie_not_set; graph completes without raising."""
        ia_no_cookie = json.dumps({"error": "cookie_not_set", "source": "provider_a"})
        result = _run_graph(
            tmp_path=tmp_path,
            ia_fn=lambda q, **kw: ia_no_cookie,
            qa_answer="",
        )
        assert "answer" in result


@pytest.mark.integration
class TestRunAuthSourceSearchCleanup:
    def test_run_dir_cleaned_up_on_error(self, tmp_path):
        """run_auth_source_search removes per-run dir even when ainvoke raises."""
        from unittest.mock import AsyncMock

        import subagent.auth_source_search as asc
        from subagent.auth_source_search import run_auth_source_search

        with (
            patch.object(asc, "auth_source_graph") as mock_graph,
            patch("subagent.auth_source_search._READER_TMP_DIR", str(tmp_path)),
            patch("Tools.text_navigator.AgentDocumentReader") as mock_reader_cls,
        ):
            mock_reader_cls.return_value = MagicMock()
            mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("graph failed"))
            with pytest.raises(RuntimeError, match="graph failed"):
                asyncio.run(run_auth_source_search("test question"))

        # Per-run dirs should have been cleaned up
        remaining = list(tmp_path.glob("auth_run_*"))
        assert remaining == [], f"Stale run dirs not cleaned: {remaining}"
