"""Unit tests for the auth_source_search subagent (grows across milestones)."""
from unittest.mock import MagicMock, patch


def test_pdf_processor_accepts_existing_converter():
    """When converter is passed, PDFProcessor skips create_model_dict."""
    with patch("Utils.pdf_processor.create_model_dict") as mock_create:
        from Utils.pdf_processor import PDFProcessor

        fake_converter = MagicMock()
        proc = PDFProcessor(
            files=[], target_folder="/tmp/test_proc",
            converter=fake_converter,
        )
        mock_create.assert_not_called()
        assert proc.converter is fake_converter


def test_pdf_processor_creates_converter_when_none():
    """When converter is None (default), PDFProcessor calls create_model_dict."""
    fake_dict = MagicMock()
    with patch("Utils.pdf_processor.create_model_dict", return_value=fake_dict) as mock_create, \
         patch("Utils.pdf_processor.PdfConverter") as mock_pdfconv:
        from Utils.pdf_processor import PDFProcessor

        PDFProcessor(files=[], target_folder="/tmp/test_proc2")
        mock_create.assert_called_once()
        mock_pdfconv.assert_called_once_with(artifact_dict=fake_dict)


def test_auth_report_state_has_required_keys():
    from State.auth_source_state import AuthReportState

    required = {
        "question",
        "max_pairs", "max_download_reflections", "max_qa_reflections", "qa_budget",
        "sub_goal", "sub_goal_history",
        "download_queries", "download_weakness", "download_reflection_count",
        "downloaded_reports",
        "selected_reports",
        "curr_answer", "qa_weakness", "qa_reflection_count",
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
