from dotenv import load_dotenv

load_dotenv(".env")
import asyncio
import logging

import sqlite3
from typing import Literal

import omegaconf

config = omegaconf.OmegaConf.load("report_config.yaml")
PROMPT_STYLE = config["PROMPT_STYLE"]

PLANNER_MODEL_NAME = config["PLANNER_MODEL_NAME"]
BACKUP_PLANNER_MODEL_NAME = config["BACKUP_PLANNER_MODEL_NAME"]

VERIFY_MODEL_NAME = config["VERIFY_MODEL_NAME"]
BACKUP_VERIFY_MODEL_NAME = config["BACKUP_VERIFY_MODEL_NAME"]

MODEL_NAME = config["MODEL_NAME"]
BACKUP_MODEL_NAME = config["BACKUP_MODEL_NAME"]

WRITER_MODEL_NAME = config["WRITER_MODEL_NAME"]
BACKUP_WRITER_MODEL_NAME = config["BACKUP_WRITER_MODEL_NAME"]

CONCLUDE_MODEL_NAME = config["CONCLUDE_MODEL_NAME"]
BACKUP_CONCLUDE_MODEL_NAME = config["BACKUP_CONCLUDE_MODEL_NAME"]

DEFAULT_REPORT_STRUCTURE = config["REPORT_STRUCTURE"]


from langchain_community.callbacks.infino_callback import get_num_tokens
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

if PROMPT_STYLE == "research":
    from Prompt.technical_research_prompt import *
elif PROMPT_STYLE == "industry":
    from Prompt.industry_prompt import *
else:
    raise ValueError("Only support indutry and technical_research prompt template")
from copy import deepcopy

from agentic_search import agentic_search_graph
from retriever import hybrid_retriever
from State.state import (
    RefinedSection,
    clearable_list_reducer,
    ReportState,
    ReportStateInput,
    ReportStateOutput,
    Section,
    SectionOutputState,
    SectionState,
)
from Tools.tools import (
    content_refinement_formatter,
    feedback_formatter,
    queries_formatter,
    refine_section_formatter,
    section_formatter,
)
from Utils.utils import (
    call_llm,
    call_llm_async,
    format_human_feedback,
    format_search_results_with_metadata,
    format_sections,
    selenium_api_search,
    track_expanded_context,
    web_search_deduplicate_and_format_sources,
)

logger = logging.getLogger("AgentLogger")
logger.setLevel(logging.ERROR)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info(
    f'VERIFY_MODEL_NAME : {config["VERIFY_MODEL_NAME"]}, MODEL_NAME : {config["MODEL_NAME"]}, CONCLUDE_MODEL_NAME : {config["CONCLUDE_MODEL_NAME"]}'
)

# =============================================================================
# Helper Functions
# =============================================================================


def search_relevance_doc(queries):
    """Search for relevant documents using hybrid retriever."""
    seen = set()
    info = []
    for q in queries:
        if q == "":
            continue
        results = hybrid_retriever.get_relevant_documents(q)
        for res in results:
            if res.page_content in seen:
                continue
            seen.add(res.page_content)
            if "table" in res.metadata:
                info.append(res)
            else:
                expanded_content = track_expanded_context(
                    res.metadata["content"], res.page_content, 1500, 1000
                )
                return_res = deepcopy(res)
                return_res.metadata["content"] = expanded_content
                info.append(return_res)

    return info


def _format_queries_history(queries_history: list) -> str:
    """Format queries history for display."""
    return "\n".join(f"{idx+1}. {q}" for idx, q in enumerate(queries_history))


def _format_follow_up_questions(follow_up_queries: list | None) -> str:
    """Format follow-up questions for display."""
    if follow_up_queries is None:
        return ""
    return "\n".join(f"{idx+1}. {q}" for idx, q in enumerate(follow_up_queries))


def _call_llm_with_retry(
    model_name: str,
    backup_model_name: str,
    messages: list,
    tool=None,
    tool_choice=None,
    max_retries: int = 5,
):
    """Call LLM with retry logic."""
    retry = 0
    while retry < max_retries:
        try:
            return call_llm(
                model_name, backup_model_name, messages, tool=tool, tool_choice=tool_choice
            )
        except Exception as e:
            retry += 1
            if retry >= max_retries:
                raise
            logger.error(f"LLM call failed (attempt {retry}/{max_retries}): {e}")


async def _call_llm_async_with_retry(
    model_name: str,
    backup_model_name: str,
    messages: list,
    tool=None,
    tool_choice=None,
    max_retries: int = 5,
):
    """Call LLM async with retry logic."""
    retry = 0
    while retry < max_retries:
        try:
            return await call_llm_async(
                model_name, backup_model_name, messages, tool=tool, tool_choice=tool_choice
            )
        except Exception as e:
            retry += 1
            if retry >= max_retries:
                raise
            logger.error(f"Async LLM call failed (attempt {retry}/{max_retries}): {e}")


# =============================================================================
# Report Planning Nodes
# =============================================================================


def generate_report_plan(state: ReportState, config: RunnableConfig):
    """Orchestrate report planning: query generation, search, and section generation."""
    topic = state["topic"]
    feedback = state.get("feedback_on_report_plan", None)
    configurable = config["configurable"]

    query_list = _generate_planner_queries(topic, feedback, configurable)
    source_str = _perform_planner_search(query_list, configurable)
    sections = _generate_report_sections(topic, source_str, feedback, configurable)

    return {"sections": sections, "curr_refine_iteration": 0}


def _generate_planner_queries(
    topic: str, feedback: str | None, configurable: dict
) -> list[str]:
    """Generate search queries for report planning."""
    report_structure = configurable["report_structure"]
    number_of_queries = configurable["number_of_queries"]

    if isinstance(report_structure, dict):
        report_structure = str(report_structure)

    formatted_feedback = format_human_feedback(feedback) if feedback else ""

    logger.info("===Start report planner query generation.===")
    system_instructions = report_planner_query_writer_instructions.format(
        topic=topic,
        report_organization=report_structure,
        number_of_queries=number_of_queries,
        feedback=formatted_feedback,
    )

    results = call_llm(
        MODEL_NAME,
        BACKUP_MODEL_NAME,
        [SystemMessage(content=system_instructions)]
        + [
            HumanMessage(
                content="Generate search queries that will help with planning the sections of the report."
            )
        ],
        tool=[queries_formatter],
        tool_choice="required",
    )
    logger.info("===End report planner query generation.===")
    return results.tool_calls[0]["args"]["queries"]


def _perform_planner_search(queries: list[str], configurable: dict) -> str:
    """Execute search and return formatted results."""
    use_web = configurable.get("use_web", False)
    use_local_db = configurable.get("use_local_db", False)

    if not use_web and not use_local_db:
        raise ValueError("Should use at least one searching tool")

    logger.info("===Start report planner query searching.===")
    source_str = ""

    if use_local_db:
        results = search_relevance_doc(queries)
        source_str = format_search_results_with_metadata(results)

    if use_web:
        web_results = selenium_api_search(queries, False)
        source_str2 = web_search_deduplicate_and_format_sources(web_results, False)
        source_str = source_str + "===\n\n" + source_str2

    logger.info("===End report planner query searching.===")
    return source_str


def _generate_report_sections(
    topic: str, source_str: str, feedback: str | None, configurable: dict
) -> list[Section]:
    """Generate report sections based on search results."""
    report_structure = configurable["report_structure"]

    if isinstance(report_structure, dict):
        report_structure = str(report_structure)

    formatted_feedback = format_human_feedback(feedback) if feedback else ""

    logger.info("===Start report plan generation.===")
    system_instructions = report_planner_instructions.format(
        topic=topic,
        report_organization=report_structure,
        context=source_str,
        feedback=formatted_feedback,
    )

    report_sections = call_llm(
        PLANNER_MODEL_NAME,
        BACKUP_PLANNER_MODEL_NAME,
        prompt=[SystemMessage(content=system_instructions)]
        + [
            HumanMessage(
                content="Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. Each section must have: name, description, plan, research, and content fields."
            )
        ],
        tool=[section_formatter],
        tool_choice="required",
    )

    sections = [
        Section(**tool_call["args"]) for tool_call in report_sections.tool_calls
    ]
    logger.info("===End report plan generation.===")
    return sections


def human_feedback(
    state: ReportState, config: RunnableConfig
) -> Command[Literal["generate_report_plan", "build_section_with_web_research"]]:
    """Handle human feedback on report plan."""
    sections = state["sections"]
    sections_str = "\n\n".join(
        f"Section: {section.name}\n"
        f"Description: {section.description}\n"
        f"Research needed: {'Yes' if section.research else 'No'}\n"
        for section in sections
    )
    feedback = interrupt(
        f"Please provide feedback on the following report plan. \n\n{sections_str}\n\n Does the report plan meet your needs? Pass 'true' to approve the report plan or provide feedback to regenerate the report plan:"
    )
    if isinstance(feedback, bool):
        logger.info("Human verify pass.")
        return Command(
            goto=[
                Send(
                    "build_section_with_web_research",
                    {"section": s, "search_iterations": 0},
                )
                for s in sections
                if s.research
            ]
        )
    elif isinstance(feedback, str):
        logger.info("Human verify fail.Back to generate_report_plan")
        return Command(
            goto="generate_report_plan",
            update={"feedback_on_report_plan": [feedback]},
        )
    else:
        logger.error("unknown type of feedback plase use str or bool (True)")
        raise TypeError(f"Interrupt value of type {type(feedback)} is not supported.")


# =============================================================================
# Section Writing Nodes
# =============================================================================


def generate_queries(state: SectionState, config: RunnableConfig):
    """Generate search queries for a section."""
    section = state["section"]
    queries = state.get("search_queries", [])
    if queries:
        return {"search_queries": queries}

    configurable = config["configurable"]
    number_of_queries = configurable["number_of_queries"]

    system_instruction = query_writer_instructions.format(
        topic=section.description, number_of_queries=number_of_queries
    )

    logger.info(f"== Start generate topic:{section.name} queries==")
    kwargs = call_llm(
        MODEL_NAME,
        BACKUP_MODEL_NAME,
        prompt=[SystemMessage(content=system_instruction)]
        + [HumanMessage(content="Generate search queries on the provided topic.")],
        tool=[queries_formatter],
        tool_choice="required",
    )
    logger.info(f"== End generate topic:{section.name} queries==")

    tool_calls = kwargs.tool_calls[0]["args"]
    return {"search_queries": tool_calls["queries"]}


async def search_db(state: SectionState, config: RunnableConfig):
    """Search databases for section information."""
    query_list = state["search_queries"]
    configurable = config["configurable"]
    use_web = configurable.get("use_web", False)
    use_local_db = configurable.get("use_local_db", False)

    if not use_web and not use_local_db:
        raise ValueError("Should use at least one searching tool")

    logger.info(
        f"== Start searching topic:{state['section'].name} queries : {query_list}=="
    )

    source_str = ""
    if use_local_db:
        results = search_relevance_doc(query_list)
        source_str = format_search_results_with_metadata(results)

    if use_web:
        search_results = await agentic_search_graph.ainvoke({"queries": query_list})
        source_str2 = search_results["source_str"]
        source_str = source_str + "===\n\n" + source_str2
    logger.info(f"== End searching topic:{state['section'].name}. ==")

    return {
        "source_str": source_str,
        "search_iterations": state["search_iterations"] + 1,
        "queries_history": query_list,
    }


def write_section(
    state: SectionState, config: RunnableConfig
) -> Command[Literal[END, "search_db"]]:
    """Write section content and determine if follow-up search is needed."""
    section = state["section"]
    configurable = config["configurable"]
    max_search_depth = configurable["max_search_depth"]

    # Prepare and truncate source if needed
    source_str = _prepare_source_for_writing(state)

    # Generate section content
    section.content = _generate_section_content(section, source_str, state)

    # Early stop if max depth reached
    if state["search_iterations"] >= max_search_depth:
        return Command(update={"completed_sections": [section]}, goto=END)

    # Grade content and determine next action
    return _grade_section_content(section, state)


def _prepare_source_for_writing(state: SectionState) -> str:
    """Prepare and truncate source string if token count exceeds threshold."""
    source_str = state["source_str"]
    section = state["section"]

    system_instructions = section_writer_instructions.format(
        section_title=section.name,
        section_topic=section.description,
        context=source_str,
        section_content=section.content or "",
        follow_up_queries=_format_follow_up_questions(state.get("follow_up_queries")),
    )

    num_tokens = get_num_tokens(system_instructions, "gpt-4o-mini")
    retry_limit = 5 if section.content else 10
    num_retries = 0

    logger.info(
        f"Start write section : {section.name}, num_input_tokens:{num_tokens}, retry:{num_retries}"
    )

    while num_tokens >= 120000 and num_retries < retry_limit:
        source_str = source_str[:-1500]
        system_instructions = section_writer_instructions.format(
            section_title=section.name,
            section_topic=section.description,
            context=source_str,
            section_content=section.content or "",
        )
        num_tokens = get_num_tokens(system_instructions, "gpt-4o-mini")
        num_retries += 1
        logger.info(
            f"Truncated source: {section.name}, num_input_tokens:{num_tokens}, retry:{num_retries}"
        )

    if num_retries >= retry_limit:
        logger.critical(
            f"There are too many tokens in the source string. Please consider reducing the amount of data searched each time."
        )
        # Return truncated source instead of raising error to maintain compatibility
        return source_str

    return source_str


def _generate_section_content(section: Section, source_str: str, state: SectionState) -> str:
    """Generate section content using LLM."""
    system_instructions = section_writer_instructions.format(
        section_title=section.name,
        section_topic=section.description,
        context=source_str,
        section_content=section.content or "",
        follow_up_queries=_format_follow_up_questions(state.get("follow_up_queries")),
    )

    logger.info(
        f"Start generate section content of topic:{section.name}, Search iteration:{state['search_iterations']}"
    )

    section_content = call_llm(
        WRITER_MODEL_NAME,
        BACKUP_WRITER_MODEL_NAME,
        [SystemMessage(content=system_instructions)]
        + [
            HumanMessage(
                content="Generate a report section based on the provided sources."
            )
        ],
    )

    logger.info(
        f"End generate section content of topic:{section.name}, Search iteration:{state['search_iterations']}"
    )
    return section_content.content


def _grade_section_content(
    section: Section, state: SectionState
) -> Command[Literal[END, "search_db"]]:
    """Grade section content and return command for next action."""
    queries_history = _format_queries_history(state["queries_history"])

    system_instructions = section_grader_instructions.format(
        section_topic=section.description,
        section=section.content,
        queries_history=queries_history,
    )

    logger.info(
        f"Start grade section content of topic:{section.name}, Search iteration:{state['search_iterations']}"
    )

    feedback = _call_llm_with_retry(
        VERIFY_MODEL_NAME,
        BACKUP_VERIFY_MODEL_NAME,
        [SystemMessage(content=system_instructions)]
        + [
            HumanMessage(
                content="Grade the report and consider follow-up questions for missing information.**Remember to use tool to output suitable format**"
            )
        ],
        tool=[feedback_formatter],
        tool_choice="required",
    )

    feedback_data = feedback.tool_calls[0]["args"]

    if feedback_data["grade"] == "pass":
        logger.info(f"Section:{section.name} pass model check or reach search depth.")
        return Command(update={"completed_sections": [section]}, goto=END)
    else:
        logger.info(
            f'Section:{section.name} fail model check.follow_up_queries:{feedback_data["follow_up_queries"]}'
        )
        return Command(
            update={
                "search_queries": feedback_data["follow_up_queries"],
                "section": section,
                "follow_up_queries": feedback_data["follow_up_queries"],
            },
            goto="search_db",
        )


# =============================================================================
# Routing and Refinement Nodes
# =============================================================================


def route_node(state: ReportState):
    """Route completed sections to formatter."""
    completed_sections = state["completed_sections"]
    completed_report_sections = format_sections(completed_sections)
    return {"report_sections_from_research": completed_report_sections}


def should_refine(state: ReportState):
    """Determine if sections should be refined."""
    logger.info("===Checking if sections should be refined===")
    if state["curr_refine_iteration"] < state["refine_iteration"]:
        return "refine_sections"
    else:
        return "gather_complete_section"


async def _refine_content_for_section(section: Section, full_context: str) -> Section:
    """Refine content for a single section based on full report context."""
    if not section.research:
        return section

    logger.info(f"Refining content for section: {section.name}")

    system_instructions = content_refinement_instructions.format(
        section_name=section.name,
        section_content=section.content,
        full_context=full_context,
    )

    refined_output = await call_llm_async(
        WRITER_MODEL_NAME,
        BACKUP_WRITER_MODEL_NAME,
        [SystemMessage(content=system_instructions)]
        + [
            HumanMessage(
                content="Refine the section content based on the full report context for consistency and integration. Use the tool to format outputs."
            )
        ],
        tool=[content_refinement_formatter],
        tool_choice="required",
    )

    refined_content = refined_output.tool_calls[0]["args"]["refined_content"]
    section.content = refined_content
    return section


async def gather_complete_section(state: ReportState, config: RunnableConfig):
    """Gather and refine completed sections for consistency."""
    logger.info("===Gathering and refining completed sections===")
    completed_sections = state["completed_sections"]
    full_context = format_sections(completed_sections)

    refined_sections = await asyncio.gather(
        *[_refine_content_for_section(s, full_context) for s in completed_sections]
    )

    completed_report_sections = format_sections(refined_sections)
    return {
        "report_sections_from_research": completed_report_sections,
        "completed_sections": refined_sections,
    }


async def _refine_single_section(
    section: Section, full_context: str, number_of_queries: int
) -> tuple[Section, list[str] | None]:
    """Refine a single section and return new queries."""
    if not section.research:
        return section, None

    system_instructions = refine_section_instructions.format(
        section_name=section.name,
        section_description=section.description,
        section_content=section.content,
        full_context=full_context,
        number_of_queries=number_of_queries,
    )

    context = [SystemMessage(content=system_instructions)] + [
        HumanMessage(
            content="""Refine the section based on the full report context and give me new queries.
                    **YOU MUST USE TOOL and give me**
                    - refined_description
                    - refined_content
                    - new_queries
                    **in formatted outputs**."""
        )
    ]

    refined_output = await _call_llm_async_with_retry(
        WRITER_MODEL_NAME,
        BACKUP_WRITER_MODEL_NAME,
        context,
        tool=[refine_section_formatter],
        tool_choice="required",
    )

    refined_data = refined_output.tool_calls[0]["args"]
    section.description += "\n\n" + refined_data["refined_description"]
    section.content = refined_data["refined_content"]

    return section, refined_data["new_queries"]


async def refine_sections(state: ReportState, config: RunnableConfig):
    """Refine all sections and trigger follow-up searches."""
    logger.info("===Refining sections===")
    configurable = config["configurable"]
    number_of_queries = configurable["number_of_queries"]
    sections = state["completed_sections"]
    full_context = format_sections(sections)

    refined_sections = await asyncio.gather(
        *[_refine_single_section(s, full_context, number_of_queries) for s in sections]
    )

    return Command(
        update={
            "completed_sections": "__CLEAR__",
            "curr_refine_iteration": state["curr_refine_iteration"] + 1,
        },
        goto=[
            Send(
                "build_section_with_web_research",
                {"section": s, "search_iterations": 0, "search_queries": q},
            )
            for s, q in refined_sections
            if s.research
        ],
    )


# =============================================================================
# Final Section Writing Nodes
# =============================================================================


def initiate_final_section_writing(state: ReportState):
    """Initiate final section writing for non-research sections."""
    return [
        Send(
            "write_final_sections",
            {
                "section": s,
                "report_sections_from_research": state["report_sections_from_research"],
            },
        )
        for s in state["sections"]
        if not s.research
    ]


def write_final_sections(state: SectionState, config: RunnableConfig):
    """Write final sections (e.g., introduction, conclusion)."""
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]
    system_instructions = final_section_writer_instructions.format(
        section_title=section.name,
        section_topic=section.description,
        context=completed_report_sections,
    )
    logger.info(f"Start write section:{section.name}")
    section_content = call_llm(
        CONCLUDE_MODEL_NAME,
        BACKUP_CONCLUDE_MODEL_NAME,
        [SystemMessage(content=system_instructions)]
        + [
            HumanMessage(
                content="Generate a report section based on the provided sources."
            )
        ],
    )
    logger.info(f"End write section:{section.name}")

    section.content = section_content.content
    return {"completed_sections": [section]}


def compile_final_report(state: ReportState):
    """Compile final report from all sections."""
    logger.info(f"Aggregate final report")
    sections = state["sections"]
    completed_sections = {s.name: s.content for s in state["completed_sections"]}

    for section in sections:
        section.content = completed_sections[section.name]

    all_sections = "\n\n".join([s.content for s in sections])

    return {"final_report": all_sections}


# =============================================================================
# Graph Builder
# =============================================================================


class ReportGraphBuilder:
    """Builder for report generation LangGraph (sync and async)."""

    def __init__(self, checkpointer=None, async_checkpointer=None):
        if checkpointer is None:
            sqlite_conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
            self.checkpointer = SqliteSaver(sqlite_conn)
        else:
            self.checkpointer = checkpointer

        self.async_checkpointer = async_checkpointer
        self._graph = None
        self._async_graph = None

    def _build_section_graph(self) -> StateGraph:
        """Build the section subgraph (shared by sync/async)."""
        section_builder = StateGraph(SectionState, output=SectionOutputState)
        section_builder.add_node("generate_queries", generate_queries)
        section_builder.add_node("search_db", search_db)
        section_builder.add_node("write_section", write_section)

        section_builder.add_edge(START, "generate_queries")
        section_builder.add_edge("generate_queries", "search_db")
        section_builder.add_edge("search_db", "write_section")

        return section_builder

    def _build_main_graph(self, section_graph: StateGraph) -> StateGraph:
        """Build the main report graph (shared by sync/async)."""
        builder = StateGraph(
            ReportState, input=ReportStateInput, output=ReportStateOutput
        )
        builder.add_node("generate_report_plan", generate_report_plan)
        builder.add_node("human_feedback", human_feedback)
        builder.add_node(
            "build_section_with_web_research", section_graph.compile()
        )
        builder.add_node("route", route_node)
        builder.add_node("refine_sections", refine_sections)
        builder.add_node("gather_complete_section", gather_complete_section)
        builder.add_node("write_final_sections", write_final_sections)
        builder.add_node("compile_final_report", compile_final_report)

        builder.add_edge(START, "generate_report_plan")
        builder.add_edge("generate_report_plan", "human_feedback")
        builder.add_edge("build_section_with_web_research", "route")
        builder.add_conditional_edges(
            "route",
            should_refine,
            {
                "refine_sections": "refine_sections",
                "gather_complete_section": "gather_complete_section",
            },
        )
        builder.add_conditional_edges(
            "gather_complete_section",
            initiate_final_section_writing,
            ["write_final_sections"],
        )
        builder.add_edge("write_final_sections", "compile_final_report")
        builder.add_edge("compile_final_report", END)

        return builder

    def get_graph(self):
        """Get synchronous LangGraph for report generation."""
        if self._graph is None:
            section_graph = self._build_section_graph()
            main_graph = self._build_main_graph(section_graph)
            self._graph = main_graph.compile(checkpointer=self.checkpointer)
        return self._graph

    def get_async_graph(self):
        """Get asynchronous LangGraph for report generation."""
        if self._async_graph is None:
            section_graph = self._build_section_graph()
            main_graph = self._build_main_graph(section_graph)

            if self.async_checkpointer is not None:
                self._async_graph = main_graph.compile(
                    checkpointer=self.async_checkpointer
                )
            else:
                self._async_graph = main_graph.compile()
        return self._async_graph
