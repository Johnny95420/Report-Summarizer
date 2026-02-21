"""Tests for Tools.reader_models — sanitize_name, save/load roundtrip, corrupted file."""

import json

import pytest
from langchain_core.documents import Document

from Tools.reader_models import BaseReaderDocument, PDFReaderDocument, sanitize_name


# ---------------------------------------------------------------------------
# sanitize_name
# ---------------------------------------------------------------------------
class TestSanitizeName:
    def test_normal(self):
        assert sanitize_name("simple-name") == "simple-name"

    def test_special_chars(self):
        result = sanitize_name('file/with:*?"<>|chars')
        assert "/" not in result
        assert ":" not in result
        assert "*" not in result

    def test_spaces_replaced(self):
        result = sanitize_name("name with spaces")
        assert " " not in result
        assert "_" in result

    def test_long_string_truncated(self):
        long_name = "a" * 300
        result = sanitize_name(long_name)
        assert len(result) <= 200

    def test_long_string_has_hash_suffix(self):
        long_name = "x" * 300
        result = sanitize_name(long_name)
        # Should be first 150 chars + "_" + 8-char md5
        assert len(result) == 150 + 1 + 8


# ---------------------------------------------------------------------------
# save / load roundtrip — BaseReaderDocument
# ---------------------------------------------------------------------------
class TestBaseReaderDocumentRoundtrip:
    def test_save_load(self, tmp_path):
        pages = [
            Document(page_content="Page 0 content", metadata={"page_id": 0}),
            Document(page_content="Page 1 content", metadata={"page_id": 1}),
        ]
        doc = BaseReaderDocument(date="2026-01-01", name="test", outlines=[{"a": 1}], pages=pages)

        path = str(tmp_path / "base.json")
        doc.save(path)
        loaded = BaseReaderDocument.load(path)

        assert loaded.name == "test"
        assert loaded.date == "2026-01-01"
        assert len(loaded.pages) == 2
        assert loaded.pages[0].page_content == "Page 0 content"
        assert loaded.outlines == [{"a": 1}]


# ---------------------------------------------------------------------------
# save / load roundtrip — PDFReaderDocument
# ---------------------------------------------------------------------------
class TestPDFReaderDocumentRoundtrip:
    def test_save_load(self, tmp_path):
        pages = [Document(page_content="page", metadata={"page_id": 0})]
        tables = [Document(page_content="table summary", metadata={"table": "<table/>"})]
        doc = PDFReaderDocument(
            date="2026-02-01", name="pdf-test", outlines=[], pages=pages, highlights="hi", tables=tables,
        )

        path = str(tmp_path / "pdf.json")
        doc.save(path)
        loaded = PDFReaderDocument.load(path)

        assert loaded.name == "pdf-test"
        assert loaded.highlights == "hi"
        assert len(loaded.tables) == 1
        assert loaded.tables[0].page_content == "table summary"


# ---------------------------------------------------------------------------
# load with corrupted file
# ---------------------------------------------------------------------------
class TestLoadCorrupted:
    def test_corrupted_json(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("{invalid json!!!")

        with pytest.raises(ValueError, match="Failed to load"):
            BaseReaderDocument.load(path)

    def test_missing_pages_key(self, tmp_path):
        path = str(tmp_path / "no_pages.json")
        with open(path, "w") as f:
            json.dump({"date": "2026", "name": "x", "outlines": []}, f)

        with pytest.raises(ValueError, match="Failed to load"):
            BaseReaderDocument.load(path)

    def test_pdf_corrupted(self, tmp_path):
        path = str(tmp_path / "bad_pdf.json")
        with open(path, "w") as f:
            f.write("not json")

        with pytest.raises(ValueError, match="Failed to load"):
            PDFReaderDocument.load(path)
