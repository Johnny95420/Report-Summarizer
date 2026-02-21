import glob
import json
import logging
import os
import pathlib
import re
from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from Tools.reader_models import BaseReaderDocument, PDFReaderDocument, sanitize_name

logger = logging.getLogger("DocumentPreprocessors")
logger.setLevel(logging.ERROR)

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_DEFAULT_READER_TMP = str(_PROJECT_ROOT / "reader_tmp")


class BaseDocumentPreprocessor(ABC):
    chunk_size: int = 500
    chunk_overlap: int = 100

    @abstractmethod
    def preprocess(self, *args, **kwargs) -> tuple[BaseReaderDocument, str]: ...

    def _get_text_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n\n\n", "\n\n\n", "\n\n", "\n"],
        )

    def _build_outlines(self, doc_splits: list[Document]) -> list[dict]:
        outlines = []
        for doc in doc_splits:
            page_id = doc.metadata["page_id"]
            if "table" in doc.metadata:
                outlines.append({"page_id": page_id, "table_summary": doc.page_content[:200]})
            else:
                headings = re.findall(r"^(#{1,3})\s+(.+)", doc.page_content, re.MULTILINE)
                outlines.append({
                    "page_id": page_id,
                    "headers": [
                        {"header_level": len(hashes), "title": title.strip()[:20] + "...[truncated]"}
                        for hashes, title in headings
                    ],
                })
        return outlines


class PDFDocumentPreprocessor(BaseDocumentPreprocessor):
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        reader_tmp: str = _DEFAULT_READER_TMP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.reader_tmp = reader_tmp

    def preprocess(self, folder: str, name: str) -> tuple[PDFReaderDocument, str]:
        """Preprocess PDF JSON files for *name* found in *folder*.

        Checks reader_tmp for a cached result first; only processes if not found.
        Result is automatically saved to reader_tmp after processing.

        Returns:
            (doc, path): the loaded/processed PDFReaderDocument and the absolute
            path where it is stored on disk. Pass path to AgentDocumentReader.open_document().
        """
        cached_path = self._cache_path(name)
        if os.path.exists(cached_path):
            return PDFReaderDocument.load(cached_path), cached_path

        doc = self._process(folder, name)

        os.makedirs(self.reader_tmp, exist_ok=True)
        doc.save(cached_path)
        return doc, cached_path

    def _cache_path(self, name: str) -> str:
        return os.path.join(self.reader_tmp, sanitize_name(name) + "_doc.json")

    def _process(self, folder: str, name: str) -> PDFReaderDocument:
        text_splitter = self._get_text_splitter()
        main_docs: list[Document] = []
        table_docs: list[Document] = []
        date: str | None = None
        highlights = ""

        related_files = glob.glob(f"{folder}/{name}*")
        if not related_files:
            raise FileNotFoundError(f"No files found matching '{folder}/{name}*'")
        for file in related_files:
            with open(file, encoding="utf-8") as f:
                information = json.load(f)

            if "table" in information:
                doc = self._process_table(file, name, information)
                if doc is not None:
                    table_docs.append(doc)
            else:
                date = self._process_date(file, name, information)
                highlights = information.get("report_highlights") or ""
                try:
                    full_content = information["full_content"]
                except KeyError:
                    raise KeyError(f"Missing 'full_content' key in file: {file}")
                main_docs.append(Document(
                    full_content,
                    metadata={"path": file, "date": date},
                ))

        text_splits = text_splitter.split_documents(main_docs)
        doc_splits = text_splits + table_docs

        for idx, doc in enumerate(doc_splits):
            doc.metadata["page_id"] = idx

        outlines = self._build_outlines(doc_splits)

        return PDFReaderDocument(
            date=date,
            name=name,
            highlights=highlights,
            outlines=outlines,
            pages=doc_splits,
            tables=table_docs,
        )

    def _process_date(self, file: str, name: str, information: dict) -> str | None:
        date = information.get("date")
        if date is None or date == "None":
            logger.warning("Cannot get date from content in file: %s. Trying filename.", file)
            try:
                date = name.split("-")[-1]
                if "_" in date:
                    date = date.split("_")[0]
            except Exception:
                logger.warning("Cannot parse date from filename: %s. Setting to None.", file)
                date = None
        return date

    def _process_table(self, file: str, name: str, information: dict) -> Document | None:
        if len(information.get("table", "")) >= 100000:
            logger.warning("File: %s. Table string longer than 100000 chars.", file)
            return None

        date = self._process_date(file, name, information)
        metadata = {
            "path": file,
            "date": date,
            "context_heading": information.get("context_heading") or "None",
            "context_paragraph": information.get("context_paragraph") or "None",
            "summary": information.get("summary", ""),
            "table": information.get("table", ""),
        }
        return Document(information.get("summary", ""), metadata=metadata)
