"""Document QA LangGraph sub-agent.

A ReAct loop that navigates financial documents using AgentDocumentReader
tools, controlled by an iteration budget, and terminates via submit_answer.

Graph topology:
    START → [agent] ←→ [tools]    (standard ReAct loop via ToolNode)
                 │
                 ├→ [extract_answer] → END  (submit_answer detected)
                 ├→ [force_answer] → [extract_answer] → END  (budget exhausted | consecutive_errors ≥ _CONSECUTIVE_ERROR_THRESHOLD | consecutive_text_only ≥ _CONSECUTIVE_TEXT_ONLY_THRESHOLD)
                 └→ [agent]  (text-only planning, no tool calls → loop back)
"""

import asyncio
import logging
import pathlib

import omegaconf
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from Prompt.document_qa_prompt import DOCUMENT_QA_SYSTEM_PROMPT
from State.document_qa_state import DocumentQAState
from Tools.reader_models import FileReference
from Tools.text_navigator import AgentDocumentReader
from Utils.utils import call_llm

# Load configurations
_HERE = pathlib.Path(__file__).parent.parent
config = omegaconf.OmegaConf.load(_HERE / "report_config.yaml")

MODEL_NAME = config["MODEL_NAME"]
BACKUP_MODEL_NAME = config["BACKUP_MODEL_NAME"]

logger = logging.getLogger("DocumentQA")

_CONSECUTIVE_ERROR_THRESHOLD: int = 3
_CONSECUTIVE_TEXT_ONLY_THRESHOLD: int = 3


# ---------------------------------------------------------------------------
# submit_answer tool
# ---------------------------------------------------------------------------
@tool
def submit_answer(answer: str) -> str:
    """Submit your final answer to the user.

    Call this tool ONCE when every todo item is marked [x] and you have all needed evidence.

    Agent Instruction:
    - The `answer` argument is delivered directly to the user — write the complete
      Traditional Chinese answer here, starting immediately with the content.
    - Do NOT include any English prose, preamble, todo items, or transition phrases
      inside the answer field. Begin the answer with the first fact or heading.
    - Call this tool exactly ONCE. Do not make any further tool calls after this.
    - Include specific numbers, dates, and document references so the answer stands alone.

    Args:
        answer: Complete final answer in Traditional Chinese.
    """
    return "[Answer submitted. Task complete.]"


# ---------------------------------------------------------------------------
# Iteration reminder builder
# ---------------------------------------------------------------------------
def _build_iteration_reminder(iteration: int, budget: int) -> str:
    """Build an ephemeral reminder injected before each LLM call."""
    remaining = budget - iteration
    if remaining <= 5:
        return (
            f"[URGENT — Iteration {iteration}/{budget} | {remaining} calls remaining]\n"
            f"Budget almost exhausted. Call submit_answer(answer='...') in THIS response. "
            f"Do not make any more tool calls — synthesise from what you have gathered so far."
        )
    elif iteration % 10 == 0:
        return (
            f"[Progress Check — Iteration {iteration}/{budget} | {remaining} calls remaining]\n"
            f"Review your todo list now. Update each item: [] (pending) or [x] (done). "
            f"No emoji. Add any newly discovered tasks and re-prioritise before continuing.\n"
        )
    else:
        return f"[Iteration {iteration}/{budget} | {remaining} calls remaining]"


# ---------------------------------------------------------------------------
# Module-level node functions
# ---------------------------------------------------------------------------
def agent_node(state: DocumentQAState, config: RunnableConfig):
    """Call LLM with tools. Inject ephemeral iteration reminder. Increment iteration counter."""
    tools = config["configurable"]["tools"]

    iteration = state.get("iteration", 0) + 1
    budget = state["budget"]

    reminder = _build_iteration_reminder(iteration, budget)
    messages = list(state["messages"]) + [HumanMessage(content=reminder)]

    logger.info("[agent] Iteration %d/%d — calling LLM (%s)", iteration, budget, MODEL_NAME)
    consecutive_errors = state.get("consecutive_errors", 0)
    try:
        response = call_llm(MODEL_NAME, BACKUP_MODEL_NAME, messages, tool=tools)
        consecutive_errors = 0
    except Exception as e:
        logger.error("[agent] LLM call failed at iteration %d: %s", iteration, e)
        consecutive_errors += 1
        if consecutive_errors >= _CONSECUTIVE_ERROR_THRESHOLD:
            response = AIMessage(
                content=f"[LLM failed {consecutive_errors} times consecutively. Forcing answer submission.]"
            )
        else:
            response = AIMessage(
                content=f"[LLM error: {e}. Retrying next iteration — consider simplifying your tool calls.]"
            )

    # Log response summary
    tool_calls = getattr(response, "tool_calls", [])
    tool_names = [tc["name"] for tc in tool_calls] if tool_calls else []
    content = response.content
    if isinstance(content, list):
        text = "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    else:
        text = str(content)
    text_preview = text[:150].replace("\n", " ") if text else "(empty)"
    logger.info("[agent] Iteration %d/%d — tools=%s, text=%s", iteration, budget, tool_names, text_preview)

    # Track consecutive text-only (no tool calls) responses for degenerate loop detection
    consecutive_text_only = state.get("consecutive_text_only", 0)
    if tool_calls:
        consecutive_text_only = 0
    else:
        consecutive_text_only += 1

    return {
        "messages": [response],
        "iteration": iteration,
        "consecutive_errors": consecutive_errors,
        "consecutive_text_only": consecutive_text_only,
    }


def extract_answer_node(state: DocumentQAState):
    """Read submit_answer args from the last AIMessage and write to state."""
    last_msg = state["messages"][-1]
    answer = ""
    tool_messages = []
    for tc in getattr(last_msg, "tool_calls", []):
        if tc["name"] == "submit_answer":
            answer = tc["args"].get("answer", "")
            tool_messages.append(ToolMessage(content="[Answer submitted. Task complete.]", tool_call_id=tc["id"]))
            logger.info("[extract_answer] Got answer via submit_answer (%d chars)", len(answer))
            break
    if not answer:
        # Fallback: extract text content from AIMessage
        content = last_msg.content
        if isinstance(content, list):
            answer = "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        else:
            answer = str(content)
        if answer:
            logger.warning("[extract_answer] No submit_answer found, using text fallback (%d chars)", len(answer))
        else:
            logger.error("[extract_answer] All extraction paths produced empty content; returning sentinel")
            answer = "[Unable to extract answer from documents]"
    return {"answer": answer, "messages": tool_messages}


def force_answer_node(state: DocumentQAState):
    """Force the agent to submit an answer immediately.

    Triggered by route_response when any of these conditions are met:
      - iteration >= budget (budget exhausted)
      - consecutive_errors >= _CONSECUTIVE_ERROR_THRESHOLD (LLM circuit breaker)
      - consecutive_text_only >= _CONSECUTIVE_TEXT_ONLY_THRESHOLD (degenerate loop)

    Only submit_answer is available in the tool list to prevent further exploration.
    """
    iteration = state.get("iteration", 0)
    consecutive_errors = state.get("consecutive_errors", 0)
    consecutive_text_only = state.get("consecutive_text_only", 0)
    budget = state["budget"]
    if consecutive_errors >= _CONSECUTIVE_ERROR_THRESHOLD:
        reason = f"consecutive_errors={consecutive_errors} >= {_CONSECUTIVE_ERROR_THRESHOLD}"
    elif consecutive_text_only >= _CONSECUTIVE_TEXT_ONLY_THRESHOLD:
        reason = f"consecutive_text_only={consecutive_text_only} >= {_CONSECUTIVE_TEXT_ONLY_THRESHOLD}"
    else:
        reason = f"budget exhausted ({iteration}/{budget})"
    logger.error("[force_answer] Triggered at iteration %d — reason: %s", iteration, reason)
    messages = list(state["messages"])
    messages.append(
        HumanMessage(
            content=(
                "[BUDGET EXHAUSTED — FINAL CALL]\n"
                "You MUST call submit_answer(answer='...') NOW with the best answer you can "
                "synthesise from everything gathered so far. This is your last chance to respond. "
                "Do NOT make any other tool calls. Write the complete answer in Traditional Chinese."
            )
        )
    )

    try:
        response = call_llm(MODEL_NAME, BACKUP_MODEL_NAME, messages, tool=[submit_answer], tool_choice="required")
    except Exception as e:
        logger.error("[force_answer] LLM call failed: %s; returning degraded answer", e)
        fallback_answer = "[系統無法產生完整答案：LLM呼叫失敗。請參考對話記錄中已收集的資訊。]"
        synthetic_msg = AIMessage(
            content="",
            tool_calls=[{"name": "submit_answer", "args": {"answer": fallback_answer}, "id": "force_answer_fallback"}],
        )
        return {"messages": [synthetic_msg]}

    logger.info(
        "[force_answer] LLM responded, tool_calls=%s", [tc["name"] for tc in getattr(response, "tool_calls", [])]
    )

    return {"messages": [response]}


def route_response(state: DocumentQAState) -> str:
    """Conditional edge after agent node."""
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, AIMessage):
        logger.debug("[route] Last message is not AIMessage, routing to agent")
        return "agent"

    # 1. submit_answer called → extract answer, go to END
    tool_calls = getattr(last_msg, "tool_calls", None) or []
    if any(tc["name"] == "submit_answer" for tc in tool_calls):
        logger.info("[route] submit_answer detected → extract_answer")
        return "extract_answer"

    # 2. Consecutive LLM errors → force answer
    if state.get("consecutive_errors", 0) >= _CONSECUTIVE_ERROR_THRESHOLD:
        logger.error("[route] %d consecutive errors → force_answer", state.get("consecutive_errors", 0))
        return "force_answer"

    # 3. Budget exhausted → force answer
    if state.get("iteration", 0) >= state["budget"]:
        logger.error("[route] Budget exhausted (%d/%d) → force_answer", state.get("iteration", 0), state["budget"])
        return "force_answer"

    # 4. Has tool calls → execute them
    if tool_calls:
        tool_names = [tc["name"] for tc in tool_calls]
        logger.info("[route] Tool calls %s → tools", tool_names)
        return "tools"

    # 5. No tool calls (text-only planning) — check for degenerate loop
    if state.get("consecutive_text_only", 0) >= _CONSECUTIVE_TEXT_ONLY_THRESHOLD:
        logger.error("[route] %d consecutive text-only responses → force_answer", state.get("consecutive_text_only", 0))
        return "force_answer"

    logger.info("[route] No tool calls (planning) → agent")
    return "agent"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def build_document_qa_graph(tools: list):
    """Build and compile the Document QA StateGraph.

    Args:
        tools: Full list of tool objects (navigator + submit_answer).

    Returns:
        Compiled LangGraph StateGraph.
    """
    tool_node = ToolNode(tools, handle_tool_errors=True)

    graph = StateGraph(DocumentQAState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("extract_answer", extract_answer_node)
    graph.add_node("force_answer", force_answer_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        route_response,
        {
            "tools": "tools",
            "extract_answer": "extract_answer",
            "force_answer": "force_answer",
            "agent": "agent",
        },
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("force_answer", "extract_answer")
    graph.add_edge("extract_answer", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Shared session setup (C2/I9)
# ---------------------------------------------------------------------------
def _prepare_qa_session(
    file_paths: list[dict],
    question: str,
    budget: int,
) -> tuple:
    """Create navigator, graph, initial state, and invoke config for a QA session.

    Returns:
        (navigator, graph, initial_state, invoke_config)
    """
    navigator = AgentDocumentReader()
    all_tools = navigator.get_tools() + [submit_answer]

    doc_list = "\n".join(f"- name: {fp['name']}\n  path: {fp['path']}" for fp in file_paths)
    system_prompt = DOCUMENT_QA_SYSTEM_PROMPT.format(budget=budget, doc_list=doc_list)

    graph = build_document_qa_graph(all_tools)

    initial_state = {
        "messages": [SystemMessage(content=system_prompt), HumanMessage(content=question)],
        "file_paths": file_paths,
        "question": question,
        "budget": budget,
        "iteration": 0,
        "answer": "",
        "consecutive_errors": 0,
        "consecutive_text_only": 0,
    }

    invoke_config = {"configurable": {"tools": all_tools}}

    return navigator, graph, initial_state, invoke_config


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------
def _validate_inputs(file_paths: list[dict], question: str, budget: int) -> None:
    """Validate inputs for run_document_qa / run_document_qa_async."""
    if not file_paths:
        raise ValueError("file_paths must be a non-empty list")
    for i, fp in enumerate(file_paths):
        try:
            FileReference(**fp)
        except Exception as e:
            raise ValueError(f"file_paths[{i}] must be a dict with 'name' and 'path' keys, got: {fp}") from e
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")
    if budget <= 0:
        raise ValueError(f"budget must be > 0, got {budget}")


def run_document_qa(
    file_paths: list[dict],
    question: str,
    budget: int = 50,
) -> str:
    """Run a document QA question synchronously.

    Args:
        file_paths: List of dicts with "name" and "path" keys.
        question: The question to answer.
        budget: Maximum iteration budget.

    Returns:
        The answer string.
    """
    _validate_inputs(file_paths, question, budget)
    navigator, graph, initial_state, invoke_config = _prepare_qa_session(file_paths, question, budget)

    try:
        result = graph.invoke(initial_state, config=invoke_config)
    finally:
        navigator.close_document()

    answer = result["answer"]
    if not answer or not answer.strip():
        raise RuntimeError("Document QA completed but produced an empty answer")
    return answer


async def run_document_qa_async(
    file_paths: list[dict],
    question: str,
    budget: int = 50,
) -> str:
    """Run a document QA question asynchronously.

    Args:
        file_paths: List of dicts with "name" and "path" keys.
        question: The question to answer.
        budget: Maximum iteration budget.

    Returns:
        The answer string.
    """
    _validate_inputs(file_paths, question, budget)
    navigator, graph, initial_state, invoke_config = _prepare_qa_session(file_paths, question, budget)

    try:
        result = await asyncio.to_thread(graph.invoke, initial_state, invoke_config)
    finally:
        navigator.close_document()

    answer = result["answer"]
    if not answer or not answer.strip():
        raise RuntimeError("Document QA completed but produced an empty answer")
    return answer


# ---------------------------------------------------------------------------
# Interactive test runner
# Usage:  cd /root/pdf_parser && python -m subagent.document_qa
#         cd /root/pdf_parser && python subagent/document_qa.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import glob as _glob
    import os
    import sys
    import time

    from dotenv import load_dotenv

    load_dotenv()

    # Verbose logging so iteration progress is visible in the terminal
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-14s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("DocumentQA").setLevel(logging.INFO)
    logging.getLogger("TextNavigator").setLevel(logging.WARNING)

    from Tools.document_preprocessors import PDFDocumentPreprocessor

    _ROOT = pathlib.Path(__file__).parent.parent
    RAW_MD = str(_ROOT / "raw_md")
    W = 72

    # ── Step 1: preprocess all raw_md docs (uses reader_tmp cache) ──────────
    print(f"\n{'─' * W}\n  Preprocessing documents (cached in reader_tmp/)\n{'─' * W}")
    preprocessor = PDFDocumentPreprocessor(chunk_size=1000, chunk_overlap=150)
    doc_paths: dict[str, str] = {}
    for json_file in sorted(_glob.glob(f"{RAW_MD}/*.json")):
        if "_table_" in json_file:
            continue
        name = os.path.splitext(os.path.basename(json_file))[0]
        _, path = preprocessor.preprocess(RAW_MD, name)
        doc_paths[name] = path
        print(f"  ok  {name[:70]}")

    if not doc_paths:
        print(f"  No .json files found in {RAW_MD}")
        sys.exit(1)

    # ── Step 2: select documents ─────────────────────────────────────────────
    names = sorted(doc_paths.keys())
    print(f"\n{'─' * W}\n  Available documents\n{'─' * W}")
    for i, name in enumerate(names, 1):
        print(f"  {i:2d}.  {name[:70]}")

    print("\n  Select docs (e.g. '1,3' | 'all') [all]: ", end="", flush=True)
    raw_sel = input().strip()
    if not raw_sel or raw_sel.lower() == "all":
        selected_pairs = [(n, doc_paths[n]) for n in names]
    else:
        idxs = [int(x) - 1 for x in raw_sel.replace(",", " ").split() if x.strip().isdigit()]
        selected_pairs = [(names[i], doc_paths[names[i]]) for i in idxs if 0 <= i < len(names)]

    if not selected_pairs:
        print("  No documents selected.")
        sys.exit(1)

    print(f"  Using {len(selected_pairs)} doc(s):")
    for n, _ in selected_pairs:
        print(f"    * {n[:70]}")

    # ── Step 3: enter question ────────────────────────────────────────────────
    print(f"\n{'─' * W}\n  Question\n{'─' * W}")
    print("  > ", end="", flush=True)
    question = input().strip()
    if not question:
        print("  No question entered.")
        sys.exit(1)

    # ── Step 4: budget ────────────────────────────────────────────────────────
    print("  Budget (iterations) [50]: ", end="", flush=True)
    raw_budget = input().strip()
    budget = int(raw_budget) if raw_budget.isdigit() and int(raw_budget) > 0 else 50

    # ── Step 5: run ───────────────────────────────────────────────────────────
    file_paths = [{"name": n, "path": p} for n, p in selected_pairs]
    print(f"\n{'=' * W}")
    print(f"  Q: {question[:68]}")
    print(f"  Docs: {len(file_paths)} | Budget: {budget}")
    print(f"{'=' * W}\n")

    t0 = time.time()
    try:
        answer = run_document_qa(file_paths, question, budget=budget)
        elapsed = time.time() - t0
        print(f"\n{'=' * W}")
        print(f"  ANSWER  ({elapsed:.1f}s)")
        print(f"{'=' * W}")
        print(answer)
        print(f"{'=' * W}\n")
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\n  ERROR after {elapsed:.1f}s: {exc}")
        sys.exit(1)
