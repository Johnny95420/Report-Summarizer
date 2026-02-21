"""Document QA LangGraph sub-agent.

A ReAct loop that navigates financial documents using AgentDocumentReader
tools, controlled by an iteration budget, and terminates via submit_answer.

Graph topology:
    START → [agent] ←→ [tools]    (standard ReAct loop via ToolNode)
                 │
                 ├→ [extract_answer] → END  (submit_answer detected)
                 ├→ [force_answer] → [extract_answer] → END  (budget exhausted)
                 └→ [agent]  (text-only planning, no tool calls → loop back)
"""

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
from Tools.text_navigator import AgentDocumentReader
from Utils.utils import call_llm

# Load configurations
_HERE = pathlib.Path(__file__).parent.parent
config = omegaconf.OmegaConf.load(_HERE / "report_config.yaml")

MODEL_NAME = config["MODEL_NAME"]
BACKUP_MODEL_NAME = config["BACKUP_MODEL_NAME"]

logger = logging.getLogger("DocumentQA")
logger.setLevel(logging.ERROR)


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
    try:
        response = call_llm(MODEL_NAME, BACKUP_MODEL_NAME, messages, tool=tools)
    except Exception as e:
        logger.error("[agent] LLM call failed at iteration %d: %s", iteration, e)
        response = AIMessage(content=f"[LLM error: {e}. Retrying next iteration — consider simplifying your tool calls.]")

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

    return {"messages": [response], "iteration": iteration}


def extract_answer_node(state: DocumentQAState):
    """Read submit_answer args from the last AIMessage and write to state."""
    last_msg = state["messages"][-1]
    answer = ""
    tool_messages = []
    for tc in getattr(last_msg, "tool_calls", []):
        if tc["name"] == "submit_answer":
            answer = tc["args"].get("answer", "")
            tool_messages.append(
                ToolMessage(content="[Answer submitted. Task complete.]", tool_call_id=tc["id"])
            )
            logger.info("[extract_answer] Got answer via submit_answer (%d chars)", len(answer))
            break
    if not answer:
        # Fallback: extract text content from AIMessage
        content = last_msg.content
        if isinstance(content, list):
            answer = "\n".join(
                b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            answer = str(content)
        if answer:
            logger.warning("[extract_answer] No submit_answer found, using text fallback (%d chars)", len(answer))
        else:
            logger.error("[extract_answer] No submit_answer found and text fallback is empty")
    return {"answer": answer, "messages": tool_messages}


def force_answer_node(state: DocumentQAState):
    """Final LLM call with only submit_answer available when budget is exhausted."""
    logger.warning("[force_answer] Budget exhausted at iteration %d — forcing submit_answer", state.get("iteration", 0))
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
        logger.error("[force_answer] LLM call failed: %s", e)
        response = AIMessage(content=f"[LLM error during forced answer: {e}]")
    logger.info("[force_answer] LLM responded, tool_calls=%s", [tc["name"] for tc in getattr(response, "tool_calls", [])])

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

    # 2. Budget exhausted → force answer
    if state.get("iteration", 0) >= state["budget"]:
        logger.warning("[route] Budget exhausted (%d/%d) → force_answer", state.get("iteration", 0), state["budget"])
        return "force_answer"

    # 3. Has tool calls → execute them
    if tool_calls:
        tool_names = [tc["name"] for tc in tool_calls]
        logger.info("[route] Tool calls %s → tools", tool_names)
        return "tools"

    # 4. No tool calls (text-only planning) → loop back to agent
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
# Convenience entry points
# ---------------------------------------------------------------------------
def _validate_inputs(file_paths: list[dict], question: str, budget: int) -> None:
    """Validate inputs for run_document_qa / run_document_qa_async."""
    if not file_paths:
        raise ValueError("file_paths must be a non-empty list")
    for i, fp in enumerate(file_paths):
        if not isinstance(fp, dict) or "name" not in fp or "path" not in fp:
            raise ValueError(f"file_paths[{i}] must be a dict with 'name' and 'path' keys, got: {fp}")
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
    navigator = AgentDocumentReader()
    all_tools = navigator.get_tools() + [submit_answer]

    doc_list = "\n".join(f"- name: {fp['name']}\n  path: {fp['path']}" for fp in file_paths)
    system_prompt = DOCUMENT_QA_SYSTEM_PROMPT.format(budget=budget, doc_list=doc_list)

    graph = build_document_qa_graph(all_tools)

    result = graph.invoke(
        {
            "messages": [SystemMessage(content=system_prompt), HumanMessage(content=question)],
            "file_paths": file_paths,
            "question": question,
            "budget": budget,
            "iteration": 0,
            "answer": "",
        },
        config={"configurable": {"tools": all_tools}},
    )

    return result["answer"]


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
    navigator = AgentDocumentReader()
    all_tools = navigator.get_tools() + [submit_answer]

    doc_list = "\n".join(f"- name: {fp['name']}\n  path: {fp['path']}" for fp in file_paths)
    system_prompt = DOCUMENT_QA_SYSTEM_PROMPT.format(budget=budget, doc_list=doc_list)

    graph = build_document_qa_graph(all_tools)

    result = await graph.ainvoke(
        {
            "messages": [SystemMessage(content=system_prompt), HumanMessage(content=question)],
            "file_paths": file_paths,
            "question": question,
            "budget": budget,
            "iteration": 0,
            "answer": "",
        },
        config={"configurable": {"tools": all_tools}},
    )

    return result["answer"]


