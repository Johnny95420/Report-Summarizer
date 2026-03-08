"""Auth Source Search LangGraph subagent.

Downloads institutional reports (InvestAnchor, Yuanta) and answers research
questions via Document QA with a shared AgentDocumentReader navigator.

File strategy: two-layer directory.
  - Global cache: reader_tmp/ (persistent, used for cache hits)
  - Per-run dir:  reader_tmp/auth_run_{uuid}/ (symlinks, deleted after run)
"""

import json
import logging
import os
import pathlib
import re

import omegaconf
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from Prompt.auth_source_prompt import (
    generate_download_queries_instruction,
    plan_sub_goal_instruction,
)
from State.auth_source_state import AuthReportState
from Tools.auth_source_tools import (
    download_investanchor_report,
    download_queries_formatter,
    download_yuanta_report,
    sub_goal_formatter,
)
from Utils.utils import call_llm

_HERE = pathlib.Path(__file__).parent.parent
_config = omegaconf.OmegaConf.load(_HERE / "report_config.yaml")
MODEL_NAME = _config["MODEL_NAME"]
BACKUP_MODEL_NAME = _config["BACKUP_MODEL_NAME"]
LIGHT_MODEL_NAME = _config["LIGHT_MODEL_NAME"]
BACKUP_LIGHT_MODEL_NAME = _config["BACKUP_LIGHT_MODEL_NAME"]

_retriever_cfg = omegaconf.OmegaConf.load(_HERE / "retriever_config.yaml")
_READER_TMP_DIR = str(_retriever_cfg.get("reader_tmp_dir", "reader_tmp"))

logger = logging.getLogger("AuthSourceSearch")

_NO_ANSWER_SENTINEL = "[auth_source_search: 未能從機構報告中找到相關資訊]"


# ---------------------------------------------------------------------------
# _sanitize_qa_answer
# ---------------------------------------------------------------------------
_INVALID_ANSWER_MARKERS = [
    "[Unable to extract answer from documents]",
    "[LLM failed",
    "[BUDGET EXHAUSTED",
    "[Answer submitted.",
]
_PLANNING_PATTERN = re.compile(r"^\s*\[[ x]?\]\s+\d+\.", re.MULTILINE)


def _sanitize_qa_answer(answer: str) -> str:
    """Return empty string when the answer is a Document QA fallback artifact."""
    if not answer or not answer.strip():
        return ""
    for marker in _INVALID_ANSWER_MARKERS:
        if marker in answer:
            return ""
    if _PLANNING_PATTERN.search(answer):
        return ""
    return answer


# ---------------------------------------------------------------------------
# Navigator state persistence helpers (single-run safety net)
# ---------------------------------------------------------------------------
def _save_navigator_state(navigator, path: str) -> None:
    """Persist navigator bookmarks and current path to a JSON sidecar file."""
    bookmarks_serializable = {label: [bm.path, bm.doc_name, bm.page_id] for label, bm in navigator._bookmarks.items()}
    state = {
        "bookmarks": bookmarks_serializable,
        "last_open_path": navigator._current_path,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _restore_navigator_state(navigator, path: str) -> None:
    """Restore navigator bookmarks from a JSON sidecar file."""
    if not os.path.exists(path):
        return
    from Tools.text_navigator import Bookmark

    with open(path, encoding="utf-8") as f:
        saved = json.load(f)

    navigator._bookmarks = {label: Bookmark(*values) for label, values in saved.get("bookmarks", {}).items()}
    last_path = saved.get("last_open_path")
    if last_path and os.path.exists(last_path):
        navigator.open_document(last_path)


# ---------------------------------------------------------------------------
# execute_downloads node (Python -- no LLM)
# ---------------------------------------------------------------------------
def execute_downloads_node(state: AuthReportState, config: RunnableConfig) -> dict:
    """Download reports from InvestAnchor and Yuanta based on generated queries.

    Gets shared_pdf_converter and run_dir from config["configurable"].
    Deduplicates by (name, source). All API errors are logged and skipped.
    """
    download_queries = state.get("download_queries", {})
    existing_reports = state.get("downloaded_reports", [])
    existing_keys = {(r["name"], r["source"]) for r in existing_reports}
    new_reports = list(existing_reports)
    converter = config["configurable"].get("shared_pdf_converter")
    run_dir = config["configurable"].get("run_dir", "")

    def _add_if_new(result: dict) -> None:
        key = (result["name"], result["source"])
        if key not in existing_keys:
            new_reports.append(result)
            existing_keys.add(key)

    # InvestAnchor: returns single {"name", "path", "source"} dict
    if ia_query := download_queries.get("investanchor"):
        try:
            result_str = download_investanchor_report(ia_query, _run_dir=run_dir)
            result = json.loads(result_str)
            if "error" in result:
                logger.warning("[execute_downloads] investanchor error: %s", result.get("error"))
            else:
                _add_if_new(result)
        except Exception as e:
            logger.error("[execute_downloads] investanchor download failed: %s", e)

    # Yuanta: returns list of {"name", "path", "source"} dicts (or error dict)
    if yuanta_query := download_queries.get("yuanta"):
        try:
            result_str = download_yuanta_report(
                yuanta_query,
                _converter=converter,
                _run_dir=run_dir,
            )
            result = json.loads(result_str)
            if isinstance(result, dict) and "error" in result:
                logger.warning("[execute_downloads] yuanta error: %s", result.get("error"))
            elif isinstance(result, list):
                for item in result:
                    _add_if_new(item)
            else:
                _add_if_new(result)
        except Exception as e:
            logger.error("[execute_downloads] yuanta download failed: %s", e)

    return {"downloaded_reports": new_reports}


# ---------------------------------------------------------------------------
# plan_sub_goal node
# ---------------------------------------------------------------------------
def plan_sub_goal_node(state: AuthReportState) -> dict:
    """Plan the next sub-goal to research from institutional reports."""
    system_instruction = plan_sub_goal_instruction.format(
        question=state["question"],
        answer=state.get("answer", "") or "(none yet)",
        sub_goal_history="\n".join(state.get("sub_goal_history", [])) or "None",
        pair_count=state.get("pair_count", 0),
    )
    response = call_llm(
        MODEL_NAME,
        BACKUP_MODEL_NAME,
        prompt=[
            SystemMessage(content=system_instruction),
            HumanMessage(content="依據以上指示，規劃下一個研究子目標。"),
        ],
        tool=[sub_goal_formatter],
        tool_choice="required",
    )
    sub_goal = response.tool_calls[0]["args"]["sub_goal"]
    return {
        "sub_goal": sub_goal,
        "sub_goal_history": [sub_goal],
        "download_reflection_count": 0,
        "qa_reflection_count": 0,
        "curr_answer": "",
    }


# ---------------------------------------------------------------------------
# generate_download_queries node
# ---------------------------------------------------------------------------
def generate_download_queries_node(state: AuthReportState) -> dict:
    """Generate keyword search queries for each institutional report source."""
    existing = state.get("downloaded_reports", [])
    already_downloaded = "\n".join(f"- {r['name']} ({r['source']})" for r in existing) if existing else "None"
    system_instruction = generate_download_queries_instruction.format(
        sub_goal=state["sub_goal"],
        download_weakness=state.get("download_weakness", "") or "(no previous weakness)",
        already_downloaded=already_downloaded,
    )
    response = call_llm(
        MODEL_NAME,
        BACKUP_MODEL_NAME,
        prompt=[
            SystemMessage(content=system_instruction),
            HumanMessage(content="依據以上指示，產生搜尋關鍵字。"),
        ],
        tool=[download_queries_formatter],
        tool_choice="required",
    )
    args = response.tool_calls[0]["args"]
    return {
        "download_queries": {
            "investanchor": args.get("investanchor"),
            "yuanta": args.get("yuanta"),
        }
    }
