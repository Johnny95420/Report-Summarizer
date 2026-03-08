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
from langgraph.graph import END
from langgraph.types import Command

from Prompt.auth_source_prompt import (
    generate_download_queries_instruction,
    outer_reflect_instruction,
    plan_sub_goal_instruction,
    reflect_download_instruction,
    reflect_qa_instruction,
    synthesize_pair_answer_instruction,
)
from State.auth_source_state import AuthReportState
from Tools.auth_source_tools import (
    download_investanchor_report,
    download_queries_formatter,
    download_yuanta_report,
    outer_reflect_formatter,
    reflect_download_formatter,
    reflect_qa_formatter,
    sub_goal_formatter,
    synthesis_formatter,
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


# ---------------------------------------------------------------------------
# reflect_download node
# ---------------------------------------------------------------------------
def reflect_download_node(state: AuthReportState) -> Command:
    """Grade downloaded reports; route to qa_agent or retry generate_download_queries."""
    count = state.get("download_reflection_count", 0)
    max_count = state.get("max_download_reflections", 1)

    if count >= max_count:
        logger.warning("[reflect_download] Max retries (%d) reached, force-passing to qa_agent", max_count)
        return Command(goto="qa_agent")

    downloaded_reports = state.get("downloaded_reports", [])
    reports_summary = (
        "\n".join(f"- {r['name']} ({r['source']})" for r in downloaded_reports) if downloaded_reports else "None"
    )
    system_instruction = reflect_download_instruction.format(
        sub_goal=state["sub_goal"],
        reports_summary=reports_summary,
    )
    response = call_llm(
        MODEL_NAME,
        BACKUP_MODEL_NAME,
        prompt=[
            SystemMessage(content=system_instruction),
            HumanMessage(content="依據以上指示執行。"),
        ],
        tool=[reflect_download_formatter],
        tool_choice="required",
    )
    args = response.tool_calls[0]["args"]
    grade = args.get("grade", "fail")
    weakness = args.get("download_weakness", "")

    if grade == "pass":
        return Command(goto="qa_agent")
    return Command(
        update={"download_weakness": weakness, "download_reflection_count": count + 1},
        goto="generate_download_queries",
    )


# ---------------------------------------------------------------------------
# reflect_qa node
# ---------------------------------------------------------------------------
def reflect_qa_node(state: AuthReportState) -> Command:
    """Grade QA answer quality; route to synthesize_pair_answer or retry qa_agent."""
    count = state.get("qa_reflection_count", 0)
    max_count = state.get("max_qa_reflections", 1)

    if count >= max_count:
        logger.warning("[reflect_qa] Max retries (%d) reached, force-passing to synthesize", max_count)
        return Command(goto="synthesize_pair_answer")

    curr_answer = state.get("curr_answer", "")
    system_instruction = reflect_qa_instruction.format(
        sub_goal=state["sub_goal"],
        curr_answer=curr_answer or "(empty)",
    )
    response = call_llm(
        MODEL_NAME,
        BACKUP_MODEL_NAME,
        prompt=[
            SystemMessage(content=system_instruction),
            HumanMessage(content="依據以上指示執行。"),
        ],
        tool=[reflect_qa_formatter],
        tool_choice="required",
    )
    args = response.tool_calls[0]["args"]
    grade = args.get("grade", "fail")
    weakness = args.get("qa_weakness", "")

    if grade == "pass":
        return Command(goto="synthesize_pair_answer")
    return Command(
        update={"qa_weakness": weakness, "qa_reflection_count": count + 1},
        goto="qa_agent",
    )


# ---------------------------------------------------------------------------
# outer_reflect node
# ---------------------------------------------------------------------------
def outer_reflect_node(state: AuthReportState) -> Command:
    """Grade whether the main question is fully answered; loop back or END."""
    pair_count = state.get("pair_count", 0)
    max_pairs = state.get("max_pairs", 3)

    if pair_count >= max_pairs:
        logger.info("[outer_reflect] max_pairs=%d reached, ending", max_pairs)
        return Command(goto=END)

    system_instruction = outer_reflect_instruction.format(
        question=state["question"],
        answer=state.get("answer", "") or "(empty)",
    )
    response = call_llm(
        MODEL_NAME,
        BACKUP_MODEL_NAME,
        prompt=[
            SystemMessage(content=system_instruction),
            HumanMessage(content="依據以上指示執行。"),
        ],
        tool=[outer_reflect_formatter],
        tool_choice="required",
    )
    args = response.tool_calls[0]["args"]
    grade = args.get("grade", "fail")

    if grade == "pass":
        return Command(goto=END)
    return Command(goto="plan_sub_goal")


# ---------------------------------------------------------------------------
# synthesize_pair_answer node
# ---------------------------------------------------------------------------
def synthesize_pair_answer_node(state: AuthReportState) -> dict:
    """Merge curr_answer into the accumulated answer. Always increments pair_count."""
    curr_answer = state.get("curr_answer", "")
    answer = state.get("answer", "")
    pair_count = state.get("pair_count", 0)

    if not curr_answer.strip():
        return {"pair_count": pair_count + 1}

    if not answer.strip():
        return {"answer": curr_answer, "pair_count": pair_count + 1}

    system_instruction = synthesize_pair_answer_instruction.format(
        sub_goal=state["sub_goal"],
        curr_answer=curr_answer,
        accumulated_answer=answer,
    )
    response = call_llm(
        MODEL_NAME,
        BACKUP_MODEL_NAME,
        prompt=[
            SystemMessage(content=system_instruction),
            HumanMessage(content="依據以上指示執行。"),
        ],
        tool=[synthesis_formatter],
        tool_choice="required",
    )
    merged = response.tool_calls[0]["args"]["merged_answer"]
    return {"answer": merged, "pair_count": pair_count + 1}
