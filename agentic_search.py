import asyncio
import logging
from typing import List, TypedDict, Set

# Load configurations
import omegaconf

config = omegaconf.OmegaConf.load("report_config.yaml")

LIGHT_MODEL_NAME = config["LIGHT_MODEL_NAME"]
BACKUP_LIGHT_MODEL_NAME = config["BACKUP_LIGHT_MODEL_NAME"]

MODEL_NAME = config["MODEL_NAME"]
BACKUP_MODEL_NAME = config["BACKUP_MODEL_NAME"]

VERIFY_MODEL_NAME = config["VERIFY_MODEL_NAME"]
BACKUP_VERIFY_MODEL_NAME = config["BACKUP_VERIFY_MODEL_NAME"]

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from Prompt.agentic_search_prompt import *
from Tools.tools import (
    quality_formatter,
    queries_formatter,
    summary_formatter,
    searching_budget_formatter,
    searching_grader_formatter,
)
from Utils.utils import (
    call_llm,
    call_llm_async,
    selenium_api_search,
    web_search_deduplicate_and_format_sources,
)
from langgraph.types import Command

# Setup logger
logger = logging.getLogger("AgenticSearch")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# TODO:The final results are not getting better after applied this node
def queries_rewriter(queries: List[str]) -> List[str]:
    str_queries = ""
    for idx, q in enumerate(queries):
        str_queries += f"{idx}. {q}\n"

    system_instruction = query_rewriter_instruction.format(
        queries_to_refine=str_queries
    )
    results = call_llm(
        MODEL_NAME,
        BACKUP_MODEL_NAME,
        prompt=[SystemMessage(content=system_instruction)]
        + [HumanMessage(content="Refine search queries on the provided queries.")],
        tool=[queries_formatter],
        tool_choice="required",
    )
    queries = results.tool_calls[0]["args"]["queries"]
    return queries


class AgenticSearchState(TypedDict):
    queries: List[str]
    followed_up_queries: List[str]
    web_results: List[dict]
    filtered_web_results: List[dict]
    compressed_web_results: List[dict]
    source_str: str
    max_num_iterations: int
    curr_num_iterations: int
    url_memo: Set[str]


async def check_search_quality_async(query: str, document: str) -> int:
    score = None
    retry = 0
    while retry < 5 and score is None:
        system_instruction = results_filter_instruction.format(
            query=query, document=document
        )
        results = await call_llm_async(
            LIGHT_MODEL_NAME,
            BACKUP_LIGHT_MODEL_NAME,
            prompt=[SystemMessage(content=system_instruction)]
            + [
                HumanMessage(
                    content="Generate the score of document on the provided query."
                )
            ],
            tool=[quality_formatter],
            tool_choice="required",
        )
        try:
            score = results.tool_calls[0]["args"]["score"]
        except (IndexError, KeyError):
            logger.warning(f"Failed to get score from tool call for query: {query}")
            retry += 1

    return score


def get_searching_budget(state: AgenticSearchState):
    # queries = state["queries"]
    # query_list = ""
    # for q in queries:
    #     query_list += f"- {q}\n"
    # system_instruction = iteration_budget_instruction.format(query_list=query_list)

    # budget_value = None
    # retry = 0
    # while retry < 5 and budget_value is None:
    #     result = call_llm(
    #         MODEL_NAME,
    #         BACKUP_MODEL_NAME,
    #         prompt=[SystemMessage(content=system_instruction)]
    #         + [
    #             HumanMessage(
    #                 content="Please give me the budget of searching iterations."
    #             )
    #         ],
    #         tool=[searching_budget_formatter],
    #         tool_choice="required",
    #     )
    #     try:
    #         budget_value = result.tool_calls[0]["args"]["budget"]
    #     except (IndexError, KeyError):
    #         logger.warning(f"Failed to get budget from tool call")
    #         retry += 1
    # logger.info(f"searching budget : {budget_value}")
    return {"max_num_iterations": 1}


def perform_web_search(state: AgenticSearchState):
    url_memo = state.get("url_memo", set())
    queries = state["queries"]
    curr_num_iterations = state.get("curr_num_iterations", 0)
    followed_up_queries = state.get("followed_up_queries", "")
    if followed_up_queries:
        logger.info(
            f"Performing followed up web search for original queries: {queries}, followed up queries:{followed_up_queries}"
        )
        queries = followed_up_queries

    else:
        queries = state["queries"]
        logger.info(f"Performing web search for queries: {queries}")

    web_results = selenium_api_search(queries, True)
    dedup_results = []
    for results in web_results:
        dedup_results.append({"results": []})
        for result in results["results"]:
            if result["url"] not in url_memo:
                url_memo.add(result["url"])
                dedup_results[-1]["results"].append(result)

    return {
        "web_results": dedup_results,
        "curr_num_iterations": curr_num_iterations + 1,
    }


def filter_and_format_results(state: AgenticSearchState):
    async def _filter_and_format_results():
        followed_up_queries = state.get("followed_up_queries", "")
        queries = followed_up_queries if followed_up_queries else state["queries"]
        web_results = state["web_results"]
        logger.info("Filtering and formatting web search results.")

        tasks = []
        for query, response in zip(queries, web_results):
            for result in response["results"]:
                document = f"Title:{result['title']}\n\nContent:{result['content']}\n\nRaw Content:{result['raw_content']}"
                tasks.append(
                    (
                        query,
                        result,
                        check_search_quality_async(query, document),
                    )
                )

        scores = await asyncio.gather(*[task[2] for task in tasks])

        filtered_web_results = []
        results_by_query = {query: [] for query in queries}

        for query, result, score in [
            (tasks[i][0], tasks[i][1], scores[i]) for i in range(len(tasks))
        ]:
            if score is not None and score >= 2:
                result["score"] = score
                results_by_query[query].append(result)

        for query in queries:
            filtered_web_results.append({"results": results_by_query[query]})

        logger.info("Finished filtering and formatting results.")
        return {"filtered_web_results": filtered_web_results}

    return asyncio.run(_filter_and_format_results())


def compress_raw_content(state: AgenticSearchState):
    followed_up_queries = state.get("followed_up_queries", "")
    queries = followed_up_queries if followed_up_queries else state["queries"]
    filtered_web_results = state["filtered_web_results"]
    final_results = []
    for query, results in zip(queries, filtered_web_results):
        query_results = {"results": []}
        for result in results["results"]:
            document = f"Title:{result['title']},Brief Content:{result['content']},Full Content:{result['raw_content']}"
            system_instruction = results_compress_instruction.format(
                query=query, document=document
            )
            compressed_result = call_llm(
                MODEL_NAME,
                BACKUP_MODEL_NAME,
                prompt=[SystemMessage(content=system_instruction)]
                + [
                    HumanMessage(
                        content="Please help me to summary every piece of document directly, indirectly, potentially, or partially related to the query."
                    )
                ],
                tool=[summary_formatter],
                tool_choice="required",
            )
            logger.info(f"{len(compressed_result.tool_calls)} tool calls in compress")
            summary_content = ""
            for tool_call in compressed_result.tool_calls:
                summary_content += (
                    tool_call["args"]["summary_content"] + "====" + "\n\n"
                )
            result["raw_content"] = summary_content
            query_results["results"].append(result)
        final_results.append(query_results)

    return {"compressed_web_results": final_results}


def aggregate_final_results(state: AgenticSearchState):
    compressed_web_results = state["compressed_web_results"]
    source_str = state.get("source_str", "")
    source_str += (
        "\n\n" if source_str else ""
    ) + web_search_deduplicate_and_format_sources(compressed_web_results, True)

    return {"source_str": source_str}


def check_searching_results(state: AgenticSearchState):
    queries = state["queries"]
    source_str = state["source_str"]
    if state["curr_num_iterations"] >= state["max_num_iterations"]:
        return Command(goto=END)

    system_instruction = searching_results_grader.format(
        query=queries, context=source_str
    )
    feedback = call_llm(
        MODEL_NAME,
        BACKUP_MODEL_NAME,
        prompt=[SystemMessage(content=system_instruction)]
        + [
            HumanMessage(
                content="Grade the source_str and consider follow-up queries for missing information"
            )
        ],
        tool=[searching_grader_formatter],
        tool_choice="required",
    )
    feedback = feedback.tool_calls[0]["args"]
    if feedback["grade"] == "pass":
        return Command(goto=END)
    else:
        return Command(
            update={"followed_up_queries": feedback["follow_up_queries"]},
            goto="perform_web_search",
        )


class AgenticSearchGraphBuilder:
    def __init__(self):
        self._graph = None

    def get_graph(self):
        if self._graph is None:
            builder = StateGraph(AgenticSearchState)
            builder.add_node("get_searching_budget", get_searching_budget)
            builder.add_node("perform_web_search", perform_web_search)
            builder.add_node("filter_and_format_results", filter_and_format_results)
            builder.add_node("compress_raw_content", compress_raw_content)
            builder.add_node("aggregate_final_results", aggregate_final_results)
            builder.add_node("check_searching_results", check_searching_results)

            builder.add_edge(START, "get_searching_budget")
            builder.add_edge("get_searching_budget", "perform_web_search")
            builder.add_edge("perform_web_search", "filter_and_format_results")
            builder.add_edge("filter_and_format_results", "compress_raw_content")
            builder.add_edge("compress_raw_content", "aggregate_final_results")
            builder.add_edge("aggregate_final_results", "check_searching_results")

            self._graph = builder.compile()
        return self._graph


agentic_search_graph_builder = AgenticSearchGraphBuilder()
agentic_search_graph = agentic_search_graph_builder.get_graph()
