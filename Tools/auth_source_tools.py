"""LLM formatter tools and download functions for auth_source_search.

Downloads call the stock_agent_tools HTTP server (STOCK_TOOLS_HOST:STOCK_TOOLS_PORT).
Auth is cookie-based: INVESTANCHOR_COOKIE and YUANTA_COOKIES env vars.

File strategy: two-layer directory.
  - Global cache: reader_tmp/ (persistent across runs)
  - Per-run dir:  reader_tmp/auth_run_{uuid}/ (symlinks, deleted after run)
"""

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from langchain_core.tools import tool
from omegaconf import OmegaConf

from Tools.reader_models import BaseReaderDocument, sanitize_name
from Utils.utils import http_session

logger = logging.getLogger("AuthSourceTools")

_cfg = OmegaConf.load(Path(__file__).parent.parent / "retriever_config.yaml")
_READER_TMP_DIR = _cfg.get("reader_tmp_dir", "reader_tmp")


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _get_stock_tools_base_url() -> str:
    host = os.environ.get("STOCK_TOOLS_HOST", "localhost")
    port = os.environ.get("STOCK_TOOLS_PORT", "8001")
    return f"http://{host}:{port}/api/news"


def _ensure_symlink(global_path: str, run_dir: str) -> str:
    """Create a symlink in run_dir pointing to global_path. Return symlink path."""
    link_name = Path(global_path).name
    link_path = os.path.join(run_dir, link_name)
    with contextlib.suppress(FileExistsError):
        os.symlink(os.path.abspath(global_path), link_path)
    return link_path


# ---------------------------------------------------------------------------
# LLM formatter tools (structured output via @tool)
# ---------------------------------------------------------------------------


@tool
def sub_goal_formatter(sub_goal: str) -> str:
    """Output the current research sub-goal for this round."""
    return sub_goal


@tool
def download_queries_formatter(
    investanchor: str | None,
    yuanta: str | None,
) -> str:
    """Output keyword search queries for InvestAnchor and Yuanta downloads.

    Set a source to null if it is not relevant for this sub-goal.
    """
    return json.dumps({"investanchor": investanchor, "yuanta": yuanta})


@tool
def document_selection_formatter(selected_names: list[str]) -> str:
    """Select which downloaded documents are relevant for the current sub-goal.

    Args:
        selected_names: List of document names (matching names in downloaded_reports).
                        Return all names if all are relevant.
                        Return empty list only if truly none are relevant.
    """
    return json.dumps(selected_names)


@tool
def reflect_download_formatter(grade: Literal["pass", "fail"], download_weakness: str) -> str:
    """Grade whether downloaded reports are sufficient for this sub-goal.

    Args:
        grade: 'pass' if reports are sufficient, 'fail' otherwise.
        download_weakness: Specific gap description when grade='fail'.
    """
    return grade


@tool
def reflect_qa_formatter(grade: Literal["pass", "fail"], qa_weakness: str) -> str:
    """Grade whether the QA answer adequately addresses the sub-goal.

    Args:
        grade: 'pass' if answer is sufficient, 'fail' otherwise.
        qa_weakness: Specific gap description when grade='fail'.
    """
    return grade


@tool
def synthesis_formatter(merged_answer: str) -> str:
    """Output the merged answer combining new findings into the accumulated answer."""
    return merged_answer


@tool
def outer_reflect_formatter(grade: Literal["pass", "fail"], hint: str) -> str:
    """Grade whether the main question is fully answered.

    Args:
        grade: 'pass' if question is answered, 'fail' if more research needed.
        hint: Guidance for the next sub-goal when grade='fail'.
    """
    return grade


# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------


def download_investanchor_report(
    query: str,
    max_results: int = 5,
    _cookie_provider=lambda: os.environ.get("INVESTANCHOR_COOKIE", ""),
    _run_dir: str | None = None,
) -> str:
    """Search InvestAnchor and save articles as a BaseReaderDocument JSON file.

    Returns JSON string {"name": str, "path": str, "source": "investanchor"} on success.
    Returns JSON string {"error": str, "source": "investanchor"} on failure.
    """
    # Strip spaces: both APIs use substring matching on titles, spaces break matching.
    query = query.replace(" ", "")

    cookie = _cookie_provider()
    if not cookie:
        logger.warning("INVESTANCHOR_COOKIE is not set")
        return json.dumps({"error": "cookie_not_set", "source": "investanchor"})

    # Deterministic naming -- same query always produces same filename
    doc_name = f"investanchor_{sanitize_name(query)}"
    global_dir = Path(_READER_TMP_DIR)
    global_dir.mkdir(parents=True, exist_ok=True)
    global_path = str(global_dir / f"{doc_name}.json")

    # L1 cache check: skip HTTP if global cache file already exists
    if os.path.exists(global_path):
        logger.info("Cache hit for InvestAnchor report: %s", query)
        path = _ensure_symlink(global_path, _run_dir) if _run_dir else global_path
        return json.dumps({"name": doc_name, "path": path, "source": "investanchor"})

    try:
        resp = http_session.get(
            f"{_get_stock_tools_base_url()}/investanchor",
            params={"keyword": query, "limit": max_results},
            headers={"X-Investanchor-Cookie": cookie},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("success") or not data.get("data"):
            logger.warning("InvestAnchor returned no results for query: %s", query)
            return json.dumps({"error": "no_results", "source": "investanchor"})

        pages = []
        for article in data["data"]:
            text = (
                f"# {article.get('title', '')}\n"
                f"**Date:** {article.get('date', '')}\n"
                f"**URL:** {article.get('url', '')}\n\n"
                f"{article.get('content', '')}"
            )
            pages.append(Document(page_content=text, metadata={"source": "investanchor"}))

        doc = BaseReaderDocument(date=None, name=doc_name, outlines=[], pages=pages)
        doc.save(global_path)

        # Symlink into per-run dir
        path = _ensure_symlink(global_path, _run_dir) if _run_dir else global_path

        return json.dumps({"name": doc_name, "path": path, "source": "investanchor"})

    except Exception as e:
        logger.error("InvestAnchor API error: %s", e)
        return json.dumps({"error": str(e), "source": "investanchor"})


def download_yuanta_report(
    query: str,
    max_results: int = 5,
    _cookie_provider=lambda: os.environ.get("YUANTA_COOKIES", ""),
    _converter=None,
    _run_dir: str | None = None,
) -> str:
    """Search Yuanta, download PDFs, and process them into PDFReaderDocument JSON files.

    Pipeline per report:
    1. Cache check: reader_tmp/{sanitize_name(title_date)}_doc.json -> skip if exists
    2. PDFProcessor (with shared converter) -> marker + LLM metadata + table summarization
    3. PDFDocumentPreprocessor -> PDFReaderDocument -> _doc.json in global cache
    4. Symlink into per-run dir

    Returns JSON string: list of {"name", "path", "source"} on success.
    Returns JSON string: {"error", "source"} on failure.
    """
    import asyncio
    import tempfile

    from Tools.document_preprocessors import PDFDocumentPreprocessor

    # Strip spaces: Yuanta API uses substring matching on titles, spaces break matching.
    query = query.replace(" ", "")

    cookie = _cookie_provider()
    if not cookie:
        logger.warning("YUANTA_COOKIES is not set")
        return json.dumps({"error": "cookie_not_set", "source": "yuanta"})

    try:
        resp = http_session.post(
            f"{_get_stock_tools_base_url()}/yuanta/search-and-download",
            params={"keyword": query, "limit": max_results},
            headers={"X-Yuanta-Cookies": cookie},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("success") or not data.get("data"):
            logger.warning("Yuanta returned no results for query: %s", query)
            return json.dumps({"error": "no_results", "source": "yuanta"})

        reader_tmp = Path(_READER_TMP_DIR)
        reader_tmp.mkdir(parents=True, exist_ok=True)
        preprocessor = PDFDocumentPreprocessor(reader_tmp=str(reader_tmp))
        results = []

        # Build converter once for the entire batch if not provided
        if _converter is None:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            _converter = PdfConverter(artifact_dict=create_model_dict())

        for report in data["data"]:
            title = report.get("title", "report")
            date = report.get("date", "")
            pdf_path = report.get("pdf_path")
            pdf_error = report.get("pdf_error")

            # Cache key: title + date (falls back to title-only if date is empty)
            cache_stem = sanitize_name(f"{title}_{date}" if date else title)
            cache_name = cache_stem + "_doc.json"
            global_cache_path = str(reader_tmp / cache_name)

            # L1 cache check: skip if _doc.json already exists in global cache
            if os.path.exists(global_cache_path):
                logger.info("Cache hit for Yuanta report: %s", title)
                path = _ensure_symlink(global_cache_path, _run_dir) if _run_dir else global_cache_path
                results.append({"name": title, "path": path, "source": "yuanta"})
                continue

            if not pdf_path or pdf_error:
                logger.warning("Yuanta report '%s' has no PDF: %s", title, pdf_error)
                continue

            if not os.path.exists(pdf_path):
                logger.warning("Yuanta PDF not found at path: %s", pdf_path)
                continue

            # Process PDF through full pipeline
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    from Utils.pdf_processor import PDFProcessor

                    proc = PDFProcessor(
                        files=[pdf_path],
                        target_folder=tmpdir,
                        model_name="deepseek/deepseek-chat",
                        converter=_converter,  # reuses GPU models, no reload
                    )
                    asyncio.run(proc.parse())

                    # PDFDocumentPreprocessor reads JSON output -> _doc.json
                    doc_name = Path(pdf_path).stem
                    _, doc_path = preprocessor.preprocess(tmpdir, doc_name)

                    # Symlink into per-run dir
                    path = _ensure_symlink(doc_path, _run_dir) if _run_dir else doc_path
                    results.append({"name": title, "path": path, "source": "yuanta"})

            except Exception as e:
                logger.warning("Failed to process Yuanta PDF '%s': %s", title, e)
                continue

        if not results:
            return json.dumps({"error": "all_processing_failed", "source": "yuanta"})

        return json.dumps(results)

    except Exception as e:
        logger.error("Yuanta API error: %s", e)
        return json.dumps({"error": str(e), "source": "yuanta"})
