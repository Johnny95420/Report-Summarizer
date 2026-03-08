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
