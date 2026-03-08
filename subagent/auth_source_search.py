"""Auth Source Search LangGraph subagent.

Downloads institutional reports (InvestAnchor, Yuanta) and answers research
questions via Document QA with a shared AgentDocumentReader navigator.

File strategy: two-layer directory.
  - Global cache: reader_tmp/ (persistent, used for cache hits)
  - Per-run dir:  reader_tmp/auth_run_{uuid}/ (symlinks, deleted after run)
"""

import asyncio
import json
import logging
import os
import pathlib
import re
import shutil
import uuid

import omegaconf
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from Prompt.auth_source_prompt import (
    generate_download_queries_instruction,
    outer_reflect_instruction,
    plan_sub_goal_instruction,
    reflect_download_instruction,
    reflect_qa_instruction,
    synthesize_pair_answer_instruction,
)
from Prompt.document_qa_prompt import DOCUMENT_QA_SYSTEM_PROMPT
from State.auth_source_state import AuthReportState
from subagent.document_qa import build_document_qa_graph, submit_answer
from Tools.auth_source_tools import (
    document_selection_formatter,
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


# ---------------------------------------------------------------------------
# Document selection helper
# ---------------------------------------------------------------------------
def _select_documents(sub_goal: str, downloaded_reports: list[dict], is_retry: bool) -> list[dict]:
    """LLM selects which downloaded documents are relevant for sub_goal.

    On retry (is_retry=True) always returns all reports to cast a wider net.
    Falls back to all reports if LLM returns empty list or call fails.
    """
    if is_retry or len(downloaded_reports) <= 1:
        return downloaded_reports

    doc_summary = "\n".join(f"- name: {r['name']} (source: {r['source']})" for r in downloaded_reports)
    system_instruction = (
        "You are selecting which research documents are relevant for a research sub-goal.\n\n"
        f"Sub-goal:\n{sub_goal}\n\n"
        f"Available documents:\n{doc_summary}\n\n"
        "Select the document names that are most likely to contain information relevant to "
        "the sub-goal. Include all documents if unsure."
    )
    try:
        result = call_llm(
            LIGHT_MODEL_NAME,
            BACKUP_LIGHT_MODEL_NAME,
            prompt=[
                SystemMessage(content=system_instruction),
                HumanMessage(content="Select relevant documents."),
            ],
            tool=[document_selection_formatter],
            tool_choice="required",
        )
        raw = result.tool_calls[0]["args"]["selected_names"]
        selected_names = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        logger.warning("[qa_agent] Document selection LLM call failed (%s) — using all reports", e)
        return downloaded_reports

    name_set = set(selected_names)
    selected = [r for r in downloaded_reports if r["name"] in name_set]
    if not selected:
        logger.warning("[qa_agent] LLM selected no documents — falling back to all reports")
        return downloaded_reports

    logger.info("[qa_agent] Selected %d/%d documents: %s", len(selected), len(downloaded_reports), selected_names)
    return selected


# ---------------------------------------------------------------------------
# qa_agent node (async)
# ---------------------------------------------------------------------------
async def qa_agent_node(state: AuthReportState, config: RunnableConfig) -> dict:
    """Run Document QA on downloaded reports using the shared navigator."""
    navigator = config["configurable"]["shared_navigator"]
    navigator_state_path = state.get("navigator_state_path", "")
    downloaded_reports = state.get("downloaded_reports", [])
    qa_budget = state.get("qa_budget", 30)
    sub_goal = state["sub_goal"]
    qa_weakness = state.get("qa_weakness", "")
    curr_answer = state.get("curr_answer", "")
    is_retry = bool(qa_weakness)

    if navigator_state_path:
        _restore_navigator_state(navigator, navigator_state_path)

    if not downloaded_reports:
        logger.warning("[qa_agent] No downloaded reports — skipping Document QA")
        if navigator_state_path:
            _save_navigator_state(navigator, navigator_state_path)
        return {"curr_answer": "", "selected_reports": []}

    selected_reports = []
    try:
        # Step 1: select relevant documents
        selected_reports = _select_documents(sub_goal, downloaded_reports, is_retry)
        file_paths = [{"name": r["name"], "path": r["path"]} for r in selected_reports]

        # Step 2: build question
        question = sub_goal
        if qa_weakness:
            question = f"{sub_goal}\n\n特別補強：{qa_weakness}"

        doc_list = "\n".join(f"- name: {fp['name']}\n  path: {fp['path']}" for fp in file_paths)
        system_prompt = DOCUMENT_QA_SYSTEM_PROMPT.format(budget=qa_budget, doc_list=doc_list)

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
        if curr_answer.strip():
            messages.append(HumanMessage(content=f"\n\n已有的初步答案（請在此基礎上補充）：\n{curr_answer}"))

        all_tools = navigator.get_tools() + [submit_answer]
        graph = build_document_qa_graph(all_tools)
        initial_state = {
            "messages": messages,
            "file_paths": file_paths,
            "question": question,
            "budget": qa_budget,
            "iteration": 0,
            "answer": "",
            "consecutive_errors": 0,
            "consecutive_text_only": 0,
        }
        invoke_config = {"configurable": {"tools": all_tools}}

        result = await asyncio.to_thread(graph.invoke, initial_state, invoke_config)
        raw_answer = result.get("answer", "")
    except Exception as e:
        logger.error("[qa_agent] Document QA failed: %s", e)
        raw_answer = ""
    finally:
        if navigator_state_path:
            _save_navigator_state(navigator, navigator_state_path)

    sanitized = _sanitize_qa_answer(raw_answer)
    return {"curr_answer": sanitized, "selected_reports": selected_reports}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def build_auth_source_graph():
    """Build and compile the auth source search StateGraph."""
    graph = StateGraph(AuthReportState)

    graph.add_node("plan_sub_goal", plan_sub_goal_node)
    graph.add_node("generate_download_queries", generate_download_queries_node)
    graph.add_node("execute_downloads", execute_downloads_node)
    graph.add_node("reflect_download", reflect_download_node)
    graph.add_node("qa_agent", qa_agent_node)
    graph.add_node("reflect_qa", reflect_qa_node)
    graph.add_node("synthesize_pair_answer", synthesize_pair_answer_node)
    graph.add_node("outer_reflect", outer_reflect_node)

    graph.set_entry_point("plan_sub_goal")
    graph.add_edge("plan_sub_goal", "generate_download_queries")
    graph.add_edge("generate_download_queries", "execute_downloads")
    graph.add_edge("execute_downloads", "reflect_download")
    # reflect_download → qa_agent | generate_download_queries  (via Command)
    graph.add_edge("qa_agent", "reflect_qa")
    # reflect_qa → synthesize_pair_answer | qa_agent  (via Command)
    graph.add_edge("synthesize_pair_answer", "outer_reflect")
    # outer_reflect → END | plan_sub_goal  (via Command)

    return graph.compile()


# Module-level singleton compiled once at import
auth_source_graph = build_auth_source_graph()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def run_auth_source_search(
    question: str,
    max_pairs: int = 3,
    max_download_reflections: int = 1,
    max_qa_reflections: int = 1,
    qa_budget: int = 30,
    converter=None,
    keep_run_dir: bool = False,
) -> str:
    """Run auth source search and return the answer string.

    Creates a fresh AgentDocumentReader, runs the compiled auth_source_graph
    in a thread (navigator is sync), cleans up the per-run dir on completion.

    Args:
        question: Research question to answer.
        max_pairs: Maximum [download, QA] rounds.
        max_download_reflections: Max reflect-and-retry cycles per download phase.
        max_qa_reflections: Max reflect-and-retry cycles per QA phase.
        qa_budget: Document QA iteration budget per qa_agent call.
        converter: Pre-built PdfConverter instance (shared GPU model).
        keep_run_dir: If True, keep the per-run directory after completion.

    Returns:
        Answer string in Traditional Chinese, or sentinel string if nothing found.
    """
    from Tools.text_navigator import AgentDocumentReader

    reader_tmp_dir = _READER_TMP_DIR
    os.makedirs(reader_tmp_dir, exist_ok=True)

    navigator = AgentDocumentReader()
    nav_state_path = os.path.join(reader_tmp_dir, f"auth_nav_{uuid.uuid4().hex}.json")

    # Per-run work directory (symlinks to global cache)
    run_dir = os.path.join(reader_tmp_dir, f"auth_run_{uuid.uuid4().hex}")
    os.makedirs(run_dir, exist_ok=True)

    initial_state: dict = {
        "question": question,
        "max_pairs": max_pairs,
        "max_download_reflections": max_download_reflections,
        "max_qa_reflections": max_qa_reflections,
        "qa_budget": qa_budget,
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
        "navigator_state_path": nav_state_path,
        "answer": "",
        "pair_count": 0,
    }
    invoke_config = {
        "configurable": {
            "shared_navigator": navigator,
            "shared_pdf_converter": converter,
            "run_dir": run_dir,
        }
    }

    try:
        result = await auth_source_graph.ainvoke(initial_state, invoke_config)
    finally:
        navigator.close_document()
        if not keep_run_dir:
            shutil.rmtree(run_dir, ignore_errors=True)
        if os.path.exists(nav_state_path):
            os.remove(nav_state_path)

    answer = result.get("answer", "")
    if not answer or not answer.strip():
        return _NO_ANSWER_SENTINEL
    return answer
