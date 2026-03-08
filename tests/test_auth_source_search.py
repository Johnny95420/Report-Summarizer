"""Unit tests for the auth_source_search subagent (grows across milestones)."""

from unittest.mock import MagicMock, patch


def test_pdf_processor_accepts_existing_converter():
    """When converter is passed, PDFProcessor skips create_model_dict."""
    with patch("Utils.pdf_processor.create_model_dict") as mock_create:
        from Utils.pdf_processor import PDFProcessor

        fake_converter = MagicMock()
        proc = PDFProcessor(
            files=[],
            target_folder="/tmp/test_proc",
            converter=fake_converter,
        )
        mock_create.assert_not_called()
        assert proc.converter is fake_converter


def test_pdf_processor_creates_converter_when_none():
    """When converter is None (default), PDFProcessor calls create_model_dict."""
    fake_dict = MagicMock()
    with (
        patch("Utils.pdf_processor.create_model_dict", return_value=fake_dict) as mock_create,
        patch("Utils.pdf_processor.PdfConverter") as mock_pdfconv,
    ):
        from Utils.pdf_processor import PDFProcessor

        PDFProcessor(files=[], target_folder="/tmp/test_proc2")
        mock_create.assert_called_once()
        mock_pdfconv.assert_called_once_with(artifact_dict=fake_dict)


def test_auth_report_state_has_required_keys():
    from State.auth_source_state import AuthReportState

    required = {
        "question",
        "max_pairs",
        "max_download_reflections",
        "max_qa_reflections",
        "qa_budget",
        "sub_goal",
        "sub_goal_history",
        "download_queries",
        "download_weakness",
        "download_reflection_count",
        "downloaded_reports",
        "selected_reports",
        "curr_answer",
        "qa_weakness",
        "qa_reflection_count",
        "navigator_state_path",
        "answer",
        "pair_count",
    }
    assert required.issubset(set(AuthReportState.__annotations__))


def test_sub_goal_history_is_annotated_append_only():
    """sub_goal_history must use Annotated[list, operator.add] so LangGraph appends."""
    import typing

    from State.auth_source_state import AuthReportState

    hints = typing.get_type_hints(AuthReportState, include_extras=True)
    ann = hints["sub_goal_history"]
    assert hasattr(ann, "__metadata__"), "sub_goal_history must be Annotated"


def test_formatter_tools_have_correct_names():
    from Tools.auth_source_tools import (
        document_selection_formatter,
        download_queries_formatter,
        outer_reflect_formatter,
        reflect_download_formatter,
        reflect_qa_formatter,
        sub_goal_formatter,
        synthesis_formatter,
    )

    assert sub_goal_formatter.name == "sub_goal_formatter"
    assert download_queries_formatter.name == "download_queries_formatter"
    assert document_selection_formatter.name == "document_selection_formatter"
    assert reflect_download_formatter.name == "reflect_download_formatter"
    assert reflect_qa_formatter.name == "reflect_qa_formatter"
    assert synthesis_formatter.name == "synthesis_formatter"
    assert outer_reflect_formatter.name == "outer_reflect_formatter"


def test_document_selection_formatter_returns_json_list():
    import json

    from Tools.auth_source_tools import document_selection_formatter

    raw = document_selection_formatter.invoke({"selected_names": ["ReportA", "ReportB"]})
    assert json.loads(raw) == ["ReportA", "ReportB"]

    raw_empty = document_selection_formatter.invoke({"selected_names": []})
    assert json.loads(raw_empty) == []


def test_download_tools_have_required_params():
    import inspect

    from Tools.auth_source_tools import download_investanchor_report, download_yuanta_report

    for fn in (download_investanchor_report, download_yuanta_report):
        params = inspect.signature(fn).parameters
        assert "_cookie_provider" in params, f"{fn.__name__} missing _cookie_provider"
        assert "_run_dir" in params, f"{fn.__name__} missing _run_dir"

    yuanta_params = inspect.signature(download_yuanta_report).parameters
    assert "_converter" in yuanta_params, "download_yuanta_report missing _converter"


def test_download_investanchor_no_cookie_returns_error():
    import json

    from Tools.auth_source_tools import download_investanchor_report

    result = json.loads(download_investanchor_report("test query", _cookie_provider=lambda: ""))
    assert result["error"] == "cookie_not_set"
    assert result["source"] == "investanchor"


def test_download_yuanta_no_cookie_returns_error():
    import json

    from Tools.auth_source_tools import download_yuanta_report

    result = json.loads(download_yuanta_report("test query", _cookie_provider=lambda: ""))
    assert result["error"] == "cookie_not_set"
    assert result["source"] == "yuanta"


def test_download_yuanta_cache_hit_skips_processing(tmp_path):
    """When _doc.json already exists for a report title+date, skip PDF processing entirely."""
    import json
    from unittest.mock import MagicMock, patch

    from Tools.auth_source_tools import download_yuanta_report
    from Tools.reader_models import sanitize_name

    title = "AI伺服器供應鏈分析"
    date = "2026-01-15"
    cached_name = sanitize_name(f"{title}_{date}") + "_doc.json"
    cached_path = tmp_path / cached_name
    cached_path.write_text("{}")

    run_dir = tmp_path / "run_test"
    run_dir.mkdir()

    api_data = [
        {
            "title": title,
            "date": date,
            "url": "https://example.com",
            "pdf_path": "/tmp/fake.pdf",
            "pdf_error": None,
        }
    ]
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"success": True, "data": api_data, "metadata": {"downloaded": 1}}
    mock_resp.raise_for_status.return_value = None

    with (
        patch("Tools.auth_source_tools.requests.post", return_value=mock_resp),
        patch("Tools.auth_source_tools._READER_TMP_DIR", str(tmp_path)),
    ):
        result = json.loads(
            download_yuanta_report(
                "AI",
                _cookie_provider=lambda: "cookie_val",
                _run_dir=str(run_dir),
            )
        )

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["source"] == "yuanta"
    # path should be a symlink in run_dir
    from pathlib import Path

    assert Path(result[0]["path"]).parent == run_dir


def test_download_investanchor_deterministic_naming(tmp_path):
    """InvestAnchor uses deterministic filename, not UUID."""
    import json
    from unittest.mock import MagicMock, patch

    from Tools.auth_source_tools import download_investanchor_report

    run_dir = tmp_path / "run_test"
    run_dir.mkdir()

    articles = [
        {
            "title": "光通訊分析",
            "date": "2026-01-01",
            "content": "# 內容\n分析...",
            "url": "https://example.com/1",
        },
    ]
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"success": True, "data": articles}
    mock_resp.raise_for_status.return_value = None

    with (
        patch("Tools.auth_source_tools.requests.get", return_value=mock_resp),
        patch("Tools.auth_source_tools._READER_TMP_DIR", str(tmp_path)),
    ):
        result1 = json.loads(
            download_investanchor_report(
                "光通訊",
                _cookie_provider=lambda: "cookie_val",
                _run_dir=str(run_dir),
            )
        )

    from pathlib import Path

    assert Path(result1["path"]).exists()
    # Call again -- should produce same global filename (deterministic)
    with (
        patch("Tools.auth_source_tools.requests.get", return_value=mock_resp),
        patch("Tools.auth_source_tools._READER_TMP_DIR", str(tmp_path)),
    ):
        result2 = json.loads(
            download_investanchor_report(
                "光通訊",
                _cookie_provider=lambda: "cookie_val",
                _run_dir=str(run_dir),
            )
        )
    assert result1["name"] == result2["name"]


def test_prompts_have_required_format_keys():
    from Prompt.auth_source_prompt import (
        generate_download_queries_instruction,
        outer_reflect_instruction,
        plan_sub_goal_instruction,
        reflect_download_instruction,
        reflect_qa_instruction,
        synthesize_pair_answer_instruction,
    )

    assert "{question}" in plan_sub_goal_instruction
    assert "{sub_goal_history}" in plan_sub_goal_instruction
    assert "{answer}" in plan_sub_goal_instruction

    assert "{sub_goal}" in generate_download_queries_instruction
    assert "{download_weakness}" in generate_download_queries_instruction
    assert "{already_downloaded}" in generate_download_queries_instruction

    assert "{sub_goal}" in reflect_download_instruction
    assert "{reports_summary}" in reflect_download_instruction

    assert "{sub_goal}" in reflect_qa_instruction
    assert "{curr_answer}" in reflect_qa_instruction

    assert "{sub_goal}" in synthesize_pair_answer_instruction
    assert "{curr_answer}" in synthesize_pair_answer_instruction
    assert "{accumulated_answer}" in synthesize_pair_answer_instruction

    assert "{question}" in outer_reflect_instruction
    assert "{answer}" in outer_reflect_instruction


# -- _sanitize_qa_answer --


def test_sanitize_qa_answer_valid():
    from subagent.auth_source_search import _sanitize_qa_answer

    answer = "台積電2024年第三季營收為 6,508 億元，年增 36%。"
    assert _sanitize_qa_answer(answer) == answer


def test_sanitize_qa_answer_empty():
    from subagent.auth_source_search import _sanitize_qa_answer

    assert _sanitize_qa_answer("") == ""
    assert _sanitize_qa_answer("   ") == ""


def test_sanitize_qa_answer_sentinel_unable():
    from subagent.auth_source_search import _sanitize_qa_answer

    assert _sanitize_qa_answer("[Unable to extract answer from documents]") == ""


def test_sanitize_qa_answer_sentinel_llm_failed():
    from subagent.auth_source_search import _sanitize_qa_answer

    assert _sanitize_qa_answer("[LLM failed 3 times consecutively...]") == ""


def test_sanitize_qa_answer_sentinel_budget():
    from subagent.auth_source_search import _sanitize_qa_answer

    assert _sanitize_qa_answer("[BUDGET EXHAUSTED — FINAL CALL]") == ""


def test_sanitize_qa_answer_sentinel_answer_submitted():
    from subagent.auth_source_search import _sanitize_qa_answer

    assert _sanitize_qa_answer("[Answer submitted. Task complete.]") == ""


def test_sanitize_qa_answer_planning_text():
    from subagent.auth_source_search import _sanitize_qa_answer

    planning = "[] 1. open_document\n[] 2. semantic_search\n[] 3. submit_answer"
    assert _sanitize_qa_answer(planning) == ""


def test_sanitize_qa_answer_done_checkboxes():
    from subagent.auth_source_search import _sanitize_qa_answer

    done = "[x] 1. opened\n[x] 2. searched"
    assert _sanitize_qa_answer(done) == ""


# -- Navigator save / restore --


def test_save_navigator_state(tmp_path):
    """Save writes bookmarks as lists + last_open_path."""
    import json
    from unittest.mock import MagicMock

    from subagent.auth_source_search import _save_navigator_state
    from Tools.text_navigator import Bookmark

    nav = MagicMock()
    nav._bookmarks = {"B1": Bookmark("/doc/a.json", "DocA", 3)}
    nav._current_path = "/doc/a.json"

    path = str(tmp_path / "state.json")
    _save_navigator_state(nav, path)

    with open(path) as f:
        saved = json.load(f)

    assert saved["last_open_path"] == "/doc/a.json"
    assert saved["bookmarks"]["B1"] == ["/doc/a.json", "DocA", 3]


def test_restore_navigator_state(tmp_path):
    """Restore converts bookmark lists back to Bookmark NamedTuples."""
    import json
    from unittest.mock import MagicMock, patch

    from subagent.auth_source_search import _restore_navigator_state
    from Tools.text_navigator import Bookmark

    path = str(tmp_path / "state.json")
    state = {
        "bookmarks": {"B1": ["/doc/a.json", "DocA", 3]},
        "last_open_path": "/doc/a.json",
    }
    with open(path, "w") as f:
        json.dump(state, f)

    nav = MagicMock()
    nav._bookmarks = {}

    with patch("os.path.exists", side_effect=lambda p: p in (path, "/doc/a.json")):
        _restore_navigator_state(nav, path)

    assert "B1" in nav._bookmarks
    assert nav._bookmarks["B1"] == Bookmark("/doc/a.json", "DocA", 3)
    nav.open_document.assert_called_once_with("/doc/a.json")


def test_restore_navigator_state_missing_file(tmp_path):
    """Missing state file -> no error, navigator unchanged."""
    from unittest.mock import MagicMock

    from subagent.auth_source_search import _restore_navigator_state

    nav = MagicMock()
    _restore_navigator_state(nav, str(tmp_path / "nonexistent.json"))
    nav.open_document.assert_not_called()


# -- execute_downloads_node --


def _base_state(**overrides):
    """Minimal AuthReportState dict for testing individual nodes."""
    state = {
        "question": "Test question",
        "max_pairs": 3,
        "max_download_reflections": 1,
        "max_qa_reflections": 1,
        "qa_budget": 10,
        "sub_goal": "Test sub-goal",
        "sub_goal_history": [],
        "download_queries": {"investanchor": "台積電 2024", "yuanta": None},
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
    state.update(overrides)
    return state


def _make_config(**overrides):
    cfg = {"configurable": {"shared_pdf_converter": None, "run_dir": "/tmp/test_run"}}
    cfg["configurable"].update(overrides)
    return cfg


def test_execute_downloads_adds_new_report():
    import json
    from unittest.mock import patch

    from subagent.auth_source_search import execute_downloads_node

    fake = json.dumps({"name": "ReportA", "path": "/tmp/a.json", "source": "investanchor"})
    with (
        patch("subagent.auth_source_search.download_investanchor_report", return_value=fake),
        patch(
            "subagent.auth_source_search.download_yuanta_report",
            return_value=json.dumps({"error": "no_results", "source": "yuanta"}),
        ),
    ):
        result = execute_downloads_node(_base_state(), _make_config())

    assert len(result["downloaded_reports"]) == 1
    assert result["downloaded_reports"][0]["name"] == "ReportA"


def test_execute_downloads_yuanta_returns_list():
    import json
    from unittest.mock import patch

    from subagent.auth_source_search import execute_downloads_node

    ia_err = json.dumps({"error": "no_results", "source": "investanchor"})
    yuanta_list = json.dumps(
        [
            {"name": "報告A", "path": "/tmp/a_doc.json", "source": "yuanta"},
            {"name": "報告B", "path": "/tmp/b_doc.json", "source": "yuanta"},
        ]
    )
    state = _base_state(download_queries={"investanchor": None, "yuanta": "AI"})
    with (
        patch("subagent.auth_source_search.download_investanchor_report", return_value=ia_err),
        patch("subagent.auth_source_search.download_yuanta_report", return_value=yuanta_list),
    ):
        result = execute_downloads_node(state, _make_config())

    assert len(result["downloaded_reports"]) == 2


def test_execute_downloads_dedup_skips_existing():
    import json
    from unittest.mock import patch

    from subagent.auth_source_search import execute_downloads_node

    existing = [{"name": "ReportA", "path": "/tmp/a.json", "source": "investanchor"}]
    fake = json.dumps({"name": "ReportA", "path": "/tmp/a.json", "source": "investanchor"})
    with (
        patch("subagent.auth_source_search.download_investanchor_report", return_value=fake),
        patch(
            "subagent.auth_source_search.download_yuanta_report",
            return_value=json.dumps({"error": "no_results", "source": "yuanta"}),
        ),
    ):
        result = execute_downloads_node(_base_state(downloaded_reports=existing), _make_config())

    assert len(result["downloaded_reports"]) == 1


def test_execute_downloads_passes_converter_and_run_dir():
    import json
    from unittest.mock import MagicMock, patch

    from subagent.auth_source_search import execute_downloads_node

    mock_converter = MagicMock()
    yuanta_result = json.dumps([{"name": "R", "path": "/p.json", "source": "yuanta"}])
    state = _base_state(download_queries={"investanchor": None, "yuanta": "AI"})

    with patch("subagent.auth_source_search.download_yuanta_report", return_value=yuanta_result) as mock_fn:
        execute_downloads_node(state, _make_config(shared_pdf_converter=mock_converter, run_dir="/tmp/my_run"))

    mock_fn.assert_called_once()
    _, kwargs = mock_fn.call_args
    assert kwargs.get("_converter") is mock_converter
    assert kwargs.get("_run_dir") == "/tmp/my_run"


def test_execute_downloads_skips_none_query():
    from unittest.mock import patch

    from subagent.auth_source_search import execute_downloads_node

    state = _base_state(download_queries={"investanchor": None, "yuanta": None})
    with (
        patch("subagent.auth_source_search.download_investanchor_report") as mock_ia,
        patch("subagent.auth_source_search.download_yuanta_report") as mock_yn,
    ):
        execute_downloads_node(state, _make_config())

    mock_ia.assert_not_called()
    mock_yn.assert_not_called()


def _make_llm_response(tool_name: str, **args):
    from unittest.mock import MagicMock

    resp = MagicMock()
    resp.tool_calls = [{"args": args}]
    return resp


# -- plan_sub_goal --


def test_plan_sub_goal_first_call():
    from unittest.mock import patch

    from subagent.auth_source_search import plan_sub_goal_node

    mock_resp = _make_llm_response("sub_goal_formatter", sub_goal="台積電 N3 良率")
    with patch("subagent.auth_source_search.call_llm", return_value=mock_resp):
        result = plan_sub_goal_node(_base_state())

    assert result["sub_goal"] == "台積電 N3 良率"
    assert result["download_reflection_count"] == 0
    assert result["qa_reflection_count"] == 0
    assert result["curr_answer"] == ""
    assert "台積電 N3 良率" in result["sub_goal_history"]


def test_plan_sub_goal_subsequent_resets_counters():
    """On every call (first or subsequent), counters reset to 0."""
    from unittest.mock import patch

    from subagent.auth_source_search import plan_sub_goal_node

    mock_resp = _make_llm_response("sub_goal_formatter", sub_goal="新子目標")
    state = _base_state(
        sub_goal_history=["前一個子目標"],
        download_reflection_count=1,
        qa_reflection_count=1,
    )
    with patch("subagent.auth_source_search.call_llm", return_value=mock_resp):
        result = plan_sub_goal_node(state)

    assert result["download_reflection_count"] == 0
    assert result["qa_reflection_count"] == 0
    assert result["curr_answer"] == ""


# -- generate_download_queries --


def test_generate_download_queries_returns_dict():
    from unittest.mock import patch

    from subagent.auth_source_search import generate_download_queries_node

    mock_resp = _make_llm_response("download_queries_formatter", investanchor="台積電 N3 2024", yuanta=None)
    with patch("subagent.auth_source_search.call_llm", return_value=mock_resp):
        result = generate_download_queries_node(_base_state())

    assert result["download_queries"]["investanchor"] == "台積電 N3 2024"
    assert result["download_queries"]["yuanta"] is None


def test_generate_download_queries_includes_weakness_in_prompt():
    """On retry, download_weakness must appear in the prompt sent to call_llm."""
    from unittest.mock import patch

    from subagent.auth_source_search import generate_download_queries_node

    captured = {}

    def capture_call(model, backup, prompt, **kwargs):
        # prompt is a list of messages — serialize to check content
        captured["prompt"] = str(prompt)
        return _make_llm_response("download_queries_formatter", investanchor="refined", yuanta=None)

    state = _base_state(download_weakness="前次搜尋缺少季度資料")
    with patch("subagent.auth_source_search.call_llm", side_effect=capture_call):
        generate_download_queries_node(state)

    assert "前次搜尋缺少季度資料" in captured["prompt"]


# -- reflect_download --


def test_reflect_download_pass_routes_to_qa_agent():
    from unittest.mock import patch

    from subagent.auth_source_search import reflect_download_node

    mock_resp = _make_llm_response("reflect_download_formatter", grade="pass", download_weakness="")
    state = _base_state(downloaded_reports=[{"name": "R", "path": "/p.json", "source": "investanchor"}])
    with patch("subagent.auth_source_search.call_llm", return_value=mock_resp):
        cmd = reflect_download_node(state)

    assert cmd.goto == "qa_agent"


def test_reflect_download_fail_routes_to_generate_queries():
    from unittest.mock import patch

    from subagent.auth_source_search import reflect_download_node

    mock_resp = _make_llm_response("reflect_download_formatter", grade="fail", download_weakness="需要更多季度資料")
    with patch("subagent.auth_source_search.call_llm", return_value=mock_resp):
        cmd = reflect_download_node(_base_state())

    assert cmd.goto == "generate_download_queries"
    assert cmd.update["download_weakness"] == "需要更多季度資料"
    assert cmd.update["download_reflection_count"] == 1


def test_reflect_download_force_pass_at_max():
    """When count >= max, route to qa_agent without calling LLM."""
    from unittest.mock import patch

    from subagent.auth_source_search import reflect_download_node

    state = _base_state(download_reflection_count=1, max_download_reflections=1)
    with patch("subagent.auth_source_search.call_llm") as mock_llm:
        cmd = reflect_download_node(state)

    mock_llm.assert_not_called()
    assert cmd.goto == "qa_agent"


# -- reflect_qa --


def test_reflect_qa_pass_routes_to_synthesize():
    from unittest.mock import patch

    from subagent.auth_source_search import reflect_qa_node

    mock_resp = _make_llm_response("reflect_qa_formatter", grade="pass", qa_weakness="")
    state = _base_state(curr_answer="台積電的N3良率已達到業界水準。")
    with patch("subagent.auth_source_search.call_llm", return_value=mock_resp):
        cmd = reflect_qa_node(state)

    assert cmd.goto == "synthesize_pair_answer"


def test_reflect_qa_fail_routes_to_qa_agent():
    from unittest.mock import patch

    from subagent.auth_source_search import reflect_qa_node

    mock_resp = _make_llm_response("reflect_qa_formatter", grade="fail", qa_weakness="缺少具體數字")
    with patch("subagent.auth_source_search.call_llm", return_value=mock_resp):
        cmd = reflect_qa_node(_base_state(curr_answer=""))

    assert cmd.goto == "qa_agent"
    assert cmd.update["qa_weakness"] == "缺少具體數字"
    assert cmd.update["qa_reflection_count"] == 1


def test_reflect_qa_force_pass_at_max():
    from unittest.mock import patch

    from subagent.auth_source_search import reflect_qa_node

    state = _base_state(qa_reflection_count=1, max_qa_reflections=1)
    with patch("subagent.auth_source_search.call_llm") as mock_llm:
        cmd = reflect_qa_node(state)

    mock_llm.assert_not_called()
    assert cmd.goto == "synthesize_pair_answer"


# -- outer_reflect --


def test_outer_reflect_pass_routes_to_end():
    from unittest.mock import patch

    from langgraph.graph import END

    from subagent.auth_source_search import outer_reflect_node

    mock_resp = _make_llm_response("outer_reflect_formatter", grade="pass", hint="")
    state = _base_state(answer="完整答案在此", pair_count=1, max_pairs=3)
    with patch("subagent.auth_source_search.call_llm", return_value=mock_resp):
        cmd = outer_reflect_node(state)

    assert cmd.goto == END


def test_outer_reflect_fail_routes_to_plan_sub_goal():
    from unittest.mock import patch

    from subagent.auth_source_search import outer_reflect_node

    mock_resp = _make_llm_response("outer_reflect_formatter", grade="fail", hint="需研究競爭對手")
    state = _base_state(answer="部分答案", pair_count=1, max_pairs=3)
    with patch("subagent.auth_source_search.call_llm", return_value=mock_resp):
        cmd = outer_reflect_node(state)

    assert cmd.goto == "plan_sub_goal"


def test_outer_reflect_force_end_at_max_pairs():
    from unittest.mock import patch

    from langgraph.graph import END

    from subagent.auth_source_search import outer_reflect_node

    state = _base_state(pair_count=3, max_pairs=3)
    with patch("subagent.auth_source_search.call_llm") as mock_llm:
        cmd = outer_reflect_node(state)

    mock_llm.assert_not_called()
    assert cmd.goto == END


# -- synthesize_pair_answer --


def test_synthesize_pair_answer_merge():
    from unittest.mock import patch

    from subagent.auth_source_search import synthesize_pair_answer_node

    mock_resp = _make_llm_response("synthesis_formatter", merged_answer="合併後的答案")
    state = _base_state(curr_answer="新發現", answer="舊答案", pair_count=0)
    with patch("subagent.auth_source_search.call_llm", return_value=mock_resp):
        result = synthesize_pair_answer_node(state)

    assert result["answer"] == "合併後的答案"
    assert result["pair_count"] == 1


def test_synthesize_pair_answer_empty_curr_skips_llm():
    """Empty curr_answer -> skip LLM merge, just increment pair_count."""
    from unittest.mock import patch

    from subagent.auth_source_search import synthesize_pair_answer_node

    state = _base_state(curr_answer="", answer="舊答案", pair_count=2)
    with patch("subagent.auth_source_search.call_llm") as mock_llm:
        result = synthesize_pair_answer_node(state)

    mock_llm.assert_not_called()
    assert result["pair_count"] == 3


def test_synthesize_pair_answer_first_pair_no_merge():
    """First pair with empty accumulated answer: use curr_answer directly without LLM."""
    from unittest.mock import patch

    from subagent.auth_source_search import synthesize_pair_answer_node

    state = _base_state(curr_answer="第一個答案", answer="", pair_count=0)
    with patch("subagent.auth_source_search.call_llm") as mock_llm:
        result = synthesize_pair_answer_node(state)

    mock_llm.assert_not_called()
    assert result["answer"] == "第一個答案"
    assert result["pair_count"] == 1


# -- qa_agent --


def _make_nav():
    from unittest.mock import MagicMock

    nav = MagicMock()
    nav.get_tools.return_value = []
    nav._bookmarks = {}
    nav._current_path = None
    return nav


def _make_selection_response(names: list[str]):
    """Fake call_llm response for _select_documents."""
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock.tool_calls = [{"args": {"selected_names": names}}]
    return mock


def test_qa_agent_no_files_returns_empty():
    """When downloaded_reports is empty, skip Document QA and return curr_answer=''."""
    import asyncio

    from subagent.auth_source_search import qa_agent_node

    config = {"configurable": {"shared_navigator": _make_nav()}}
    result = asyncio.run(qa_agent_node(_base_state(downloaded_reports=[]), config))
    assert result["curr_answer"] == ""
    assert result["selected_reports"] == []


def test_qa_agent_selects_subset_of_documents():
    """LLM selects one document; only that one is passed to Document QA."""
    import asyncio
    from unittest.mock import MagicMock, patch

    from subagent.auth_source_search import qa_agent_node

    reports = [
        {"name": "ReportA", "path": "/a.json", "source": "investanchor"},
        {"name": "ReportB", "path": "/b.json", "source": "yuanta"},
    ]
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = {"answer": "分析結果"}
    config = {"configurable": {"shared_navigator": _make_nav()}}

    with (
        patch("subagent.auth_source_search.call_llm", return_value=_make_selection_response(["ReportA"])),
        patch("subagent.auth_source_search.build_document_qa_graph", return_value=fake_graph),
    ):
        result = asyncio.run(qa_agent_node(_base_state(downloaded_reports=reports), config))

    assert result["selected_reports"] == [reports[0]]
    call_kwargs = fake_graph.invoke.call_args[0][0]
    assert len(call_kwargs["file_paths"]) == 1
    assert call_kwargs["file_paths"][0]["name"] == "ReportA"


def test_qa_agent_retry_uses_all_documents():
    """On QA retry (qa_weakness set), all documents are used regardless of selection."""
    import asyncio
    from unittest.mock import MagicMock, patch

    from subagent.auth_source_search import qa_agent_node

    reports = [
        {"name": "ReportA", "path": "/a.json", "source": "investanchor"},
        {"name": "ReportB", "path": "/b.json", "source": "yuanta"},
    ]
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = {"answer": "補充答案"}
    config = {"configurable": {"shared_navigator": _make_nav()}}

    state = _base_state(downloaded_reports=reports, qa_weakness="缺少財務數據")
    with (
        patch("subagent.auth_source_search.call_llm") as mock_llm,
        patch("subagent.auth_source_search.build_document_qa_graph", return_value=fake_graph),
    ):
        result = asyncio.run(qa_agent_node(state, config))

    mock_llm.assert_not_called()
    assert result["selected_reports"] == reports


def test_qa_agent_selection_fallback_on_empty_llm():
    """If LLM returns empty selection, fall back to all documents."""
    import asyncio
    from unittest.mock import MagicMock, patch

    from subagent.auth_source_search import qa_agent_node

    reports = [
        {"name": "ReportA", "path": "/a.json", "source": "investanchor"},
        {"name": "ReportB", "path": "/b.json", "source": "yuanta"},
    ]
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = {"answer": "答案"}
    config = {"configurable": {"shared_navigator": _make_nav()}}

    with (
        patch("subagent.auth_source_search.call_llm", return_value=_make_selection_response([])),
        patch("subagent.auth_source_search.build_document_qa_graph", return_value=fake_graph),
    ):
        result = asyncio.run(qa_agent_node(_base_state(downloaded_reports=reports), config))

    assert result["selected_reports"] == reports


def test_qa_agent_sanitizes_sentinel_answer():
    """Document QA sentinel output is sanitised to empty string."""
    import asyncio
    from unittest.mock import MagicMock, patch

    from subagent.auth_source_search import qa_agent_node

    fake_graph = MagicMock()
    fake_graph.invoke.return_value = {"answer": "[Unable to extract answer from documents]"}
    config = {"configurable": {"shared_navigator": _make_nav()}}

    reports = [{"name": "Doc", "path": "/doc.json", "source": "investanchor"}]
    state = _base_state(downloaded_reports=reports)
    with (
        patch("subagent.auth_source_search.call_llm", return_value=_make_selection_response(["Doc"])),
        patch("subagent.auth_source_search.build_document_qa_graph", return_value=fake_graph),
    ):
        result = asyncio.run(qa_agent_node(state, config))

    assert result["curr_answer"] == ""


def test_qa_agent_exception_returns_empty():
    """If graph.invoke raises, qa_agent catches it and returns curr_answer=''."""
    import asyncio
    from unittest.mock import MagicMock, patch

    from subagent.auth_source_search import qa_agent_node

    fake_graph = MagicMock()
    fake_graph.invoke.side_effect = RuntimeError("Document QA crashed")
    config = {"configurable": {"shared_navigator": _make_nav()}}

    reports = [{"name": "Doc", "path": "/doc.json", "source": "investanchor"}]
    state = _base_state(downloaded_reports=reports)
    with (
        patch("subagent.auth_source_search.call_llm", return_value=_make_selection_response(["Doc"])),
        patch("subagent.auth_source_search.build_document_qa_graph", return_value=fake_graph),
    ):
        result = asyncio.run(qa_agent_node(state, config))

    assert result["curr_answer"] == ""
    assert "selected_reports" in result


def test_build_auth_source_graph_has_all_nodes():
    from subagent.auth_source_search import build_auth_source_graph

    graph = build_auth_source_graph()
    expected = {
        "plan_sub_goal",
        "generate_download_queries",
        "execute_downloads",
        "reflect_download",
        "qa_agent",
        "reflect_qa",
        "synthesize_pair_answer",
        "outer_reflect",
    }
    assert expected.issubset(set(graph.nodes.keys()))
