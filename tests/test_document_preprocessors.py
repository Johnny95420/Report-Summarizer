"""Tests for Tools.document_preprocessors — BaseDocumentPreprocessor._build_outlines and config loading."""

from unittest.mock import MagicMock

from langchain_core.documents import Document

from Tools.document_preprocessors import _DEFAULT_READER_TMP, BaseDocumentPreprocessor


# ---------------------------------------------------------------------------
# Config loading: _DEFAULT_READER_TMP reads from retriever_config.yaml
# ---------------------------------------------------------------------------
class TestConfigLoading:
    def test_default_reader_tmp_is_string(self):
        """_DEFAULT_READER_TMP must be a non-empty string."""
        assert isinstance(_DEFAULT_READER_TMP, str)
        assert _DEFAULT_READER_TMP

    def test_default_reader_tmp_contains_reader_tmp(self):
        """_DEFAULT_READER_TMP should contain the config value 'reader_tmp' (from FAKE_CONFIG)."""
        assert "reader_tmp" in _DEFAULT_READER_TMP

    def test_default_reader_tmp_is_absolute_path(self):
        """_DEFAULT_READER_TMP should be an absolute path (anchored to project root)."""
        import os
        assert os.path.isabs(_DEFAULT_READER_TMP)


# ---------------------------------------------------------------------------
# Outline helpers
# ---------------------------------------------------------------------------
class _ConcretePreprocessor(BaseDocumentPreprocessor):
    def preprocess(self, *args, **kwargs):
        return MagicMock(), "/fake/path.json"


# ---------------------------------------------------------------------------
# _build_outlines: header truncation (I4)
# ---------------------------------------------------------------------------
class TestOutlineHeaderTruncation:
    def setup_method(self):
        self.preprocessor = _ConcretePreprocessor()

    def _make_doc(self, content: str, page_id: int = 0, table: bool = False) -> Document:
        metadata = {"page_id": page_id}
        if table:
            metadata["table"] = "<table>...</table>"
        return Document(page_content=content, metadata=metadata)

    def test_long_title_truncated_to_50_chars(self):
        """Title > 50 chars is sliced to 50 + '...[truncated]'."""
        title = "A" * 60
        doc = self._make_doc(f"# {title}")
        outlines = self.preprocessor._build_outlines([doc])
        assert outlines[0]["headers"][0]["title"] == "A" * 50 + "...[truncated]"

    def test_short_title_gets_suffix_unconditionally(self):
        """Suffix '...[truncated]' is always appended, even for short titles."""
        doc = self._make_doc("# Short title")
        outlines = self.preprocessor._build_outlines([doc])
        assert outlines[0]["headers"][0]["title"] == "Short title...[truncated]"

    def test_exactly_50_char_title(self):
        """Title of exactly 50 chars: all 50 chars kept + suffix."""
        title = "B" * 50
        doc = self._make_doc(f"# {title}")
        outlines = self.preprocessor._build_outlines([doc])
        assert outlines[0]["headers"][0]["title"] == "B" * 50 + "...[truncated]"

    def test_long_title_total_length(self):
        """For a 60-char title, result length = 50 + len('...[truncated]') = 64."""
        title = "C" * 60
        doc = self._make_doc(f"# {title}")
        outlines = self.preprocessor._build_outlines([doc])
        result = outlines[0]["headers"][0]["title"]
        assert len(result) == 50 + len("...[truncated]")

    def test_header_level_1(self):
        doc = self._make_doc("# H1 heading")
        outlines = self.preprocessor._build_outlines([doc])
        assert outlines[0]["headers"][0]["header_level"] == 1

    def test_header_level_2(self):
        doc = self._make_doc("## H2 heading")
        outlines = self.preprocessor._build_outlines([doc])
        assert outlines[0]["headers"][0]["header_level"] == 2

    def test_header_level_3(self):
        doc = self._make_doc("### H3 heading")
        outlines = self.preprocessor._build_outlines([doc])
        assert outlines[0]["headers"][0]["header_level"] == 3

    def test_whitespace_stripped_before_truncation(self):
        """Leading/trailing whitespace in header title is stripped before slicing."""
        doc = self._make_doc("#   stripped title   ")
        outlines = self.preprocessor._build_outlines([doc])
        assert outlines[0]["headers"][0]["title"].startswith("stripped title")

    def test_multiple_headers_in_one_doc(self):
        content = "# First header\n\nSome text.\n\n## Second header\n\nMore text."
        doc = self._make_doc(content)
        outlines = self.preprocessor._build_outlines([doc])
        headers = outlines[0]["headers"]
        assert len(headers) == 2
        assert headers[0]["header_level"] == 1
        assert headers[1]["header_level"] == 2

    def test_no_headings_returns_empty_list(self):
        doc = self._make_doc("Just plain text with no headings.")
        outlines = self.preprocessor._build_outlines([doc])
        assert outlines[0]["headers"] == []

    def test_table_doc_returns_table_summary_not_headers(self):
        doc = self._make_doc("Table content summary here.", table=True)
        outlines = self.preprocessor._build_outlines([doc])
        assert "table_summary" in outlines[0]
        assert "headers" not in outlines[0]

    def test_table_summary_truncated_to_200_chars(self):
        long_content = "X" * 300
        doc = self._make_doc(long_content, table=True)
        outlines = self.preprocessor._build_outlines([doc])
        assert len(outlines[0]["table_summary"]) == 200

    def test_page_id_preserved_in_text_doc(self):
        doc = self._make_doc("# Heading", page_id=7)
        outlines = self.preprocessor._build_outlines([doc])
        assert outlines[0]["page_id"] == 7

    def test_page_id_preserved_in_table_doc(self):
        doc = self._make_doc("summary", page_id=3, table=True)
        outlines = self.preprocessor._build_outlines([doc])
        assert outlines[0]["page_id"] == 3

    def test_mixed_docs_multiple_pages(self):
        """Multiple docs with mixed text and table entries are all processed."""
        docs = [
            self._make_doc("# Chapter 1", page_id=0),
            self._make_doc("table data", page_id=1, table=True),
            self._make_doc("## Section 2.1", page_id=2),
        ]
        outlines = self.preprocessor._build_outlines(docs)
        assert len(outlines) == 3
        assert "headers" in outlines[0]
        assert "table_summary" in outlines[1]
        assert "headers" in outlines[2]
