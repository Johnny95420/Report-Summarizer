import asyncio
import copy
import logging
from typing import List, Set, TypedDict

# Load configurations
import omegaconf

config = omegaconf.OmegaConf.load("report_config.yaml")

LIGHT_MODEL_NAME = config["LIGHT_MODEL_NAME"]
BACKUP_LIGHT_MODEL_NAME = config["BACKUP_LIGHT_MODEL_NAME"]

MODEL_NAME = config["MODEL_NAME"]
BACKUP_MODEL_NAME = config["BACKUP_MODEL_NAME"]

VERIFY_MODEL_NAME = config["VERIFY_MODEL_NAME"]
BACKUP_VERIFY_MODEL_NAME = config["BACKUP_VERIFY_MODEL_NAME"]

from langchain_community.callbacks.infino_callback import get_num_tokens
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from Prompt.agentic_search_prompt import *
from Tools.tools import (
    quality_formatter,
    queries_formatter,
    searching_budget_formatter,
    searching_grader_formatter,
    summary_formatter,
)
from Utils.utils import (
    call_llm,
    call_llm_async,
    selenium_api_search,
    web_search_deduplicate_and_format_sources,
)

# Setup logger
logger = logging.getLogger("AgenticSearch")
logger.setLevel(logging.ERROR)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def select_model_based_on_tokens(
    content: str, token_threshold: int = 4096
) -> tuple[str, str]:
    """
    Select appropriate model based on content token length.

    Args:
        content: The text content to analyze
        token_threshold: Token count threshold for model selection

    Returns:
        Tuple of (primary_model, backup_model)
    """
    content_tokens = get_num_tokens(content, "gpt-4o-mini")

    if content_tokens > token_threshold:
        # Use heavy models for long content
        return MODEL_NAME, BACKUP_MODEL_NAME
    else:
        # Use light models for shorter content
        return LIGHT_MODEL_NAME, BACKUP_LIGHT_MODEL_NAME


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
    system_instruction = results_filter_instruction.format(
        query=query, document=document
    )

    # Select appropriate model based on document token length
    model_name, backup_model_name = select_model_based_on_tokens(document)

    results = await call_llm_async(
        model_name,
        backup_model_name,
        prompt=[SystemMessage(content=system_instruction)]
        + [
            HumanMessage(
                content="Generate the score of document on the provided query."
            )
        ],
        tool=[quality_formatter],
        tool_choice="required",
    )
    score = results.tool_calls[0]["args"]["score"]
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
    budget_value = 1
    return {"max_num_iterations": budget_value}


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


async def filter_and_format_results(state: AgenticSearchState):
    followed_up_queries = state.get("followed_up_queries", "")
    queries = followed_up_queries if followed_up_queries else state["queries"]
    web_results = state["web_results"]
    logger.info("Filtering and formatting web search results.")

    # Create semaphore to limit concurrent operations
    semaphore = asyncio.Semaphore(2)

    async def check_quality_with_metadata(
        query: str, result: dict
    ) -> tuple[str, dict, int]:
        """Check quality with semaphore control and metadata preservation."""
        async with semaphore:
            try:
                document = f"Title:{result['title']}\n\nContent:{result['content']}\n\nRaw Content:{result['raw_content']}"
                score = await check_search_quality_async(query, document)
                return query, result, score
            except Exception as e:
                logger.warning(
                    f"Quality check failed for result {result.get('url', 'unknown')}: {e}"
                )
                return query, result, 0  # Return 0 score for failed checks

    # Create tasks with metadata preserved
    quality_tasks = []
    for query, response in zip(queries, web_results):
        for result in response["results"]:
            quality_tasks.append(check_quality_with_metadata(query, result))

    # Execute all quality checks with error handling
    try:
        quality_results = await asyncio.gather(*quality_tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f"Batch quality check failed: {e}")
        return {"filtered_web_results": [{"results": []} for _ in queries]}

    # Process results with error handling
    results_by_query = {query: [] for query in queries}

    for quality_result in quality_results:
        if isinstance(quality_result, Exception):
            logger.warning(f"Quality check exception: {quality_result}")
            continue

        query, search_result, score = quality_result
        if score is not None and score > 2:
            search_result["score"] = score
            results_by_query[query].append(search_result)

    # Build final results structure
    filtered_web_results = [{"results": results_by_query[query]} for query in queries]

    total_filtered = sum(len(results["results"]) for results in filtered_web_results)
    logger.info(f"Finished filtering: {total_filtered} results passed quality check.")

    return {"filtered_web_results": filtered_web_results}


async def compress_raw_content(state: AgenticSearchState):
    followed_up_queries = state.get("followed_up_queries", "")
    queries = followed_up_queries if followed_up_queries else state["queries"]
    filtered_web_results = state["filtered_web_results"]

    # Create semaphore to limit concurrent compression operations
    semaphore = asyncio.Semaphore(2)

    async def compress_content_with_metadata(
        query_idx: int, result_idx: int, query: str, result: dict
    ):
        """Compress content with semaphore control and metadata preservation."""
        async with semaphore:
            try:
                document = f"Title:{result['title']},Brief Content:{result['content']},Full Content:{result['raw_content']}"
                system_instruction = results_compress_instruction.format(
                    query=query, document=document
                )

                # Select appropriate model based on raw_content token length
                model_name, backup_model_name = select_model_based_on_tokens(
                    result["raw_content"]
                )

                compressed_result = await call_llm_async(
                    model_name,
                    backup_model_name,
                    prompt=[SystemMessage(content=system_instruction)]
                    + [
                        HumanMessage(
                            content="Please help me to summary every piece of document directly, indirectly, potentially, or partially related to the query."
                        )
                    ],
                    tool=[summary_formatter],
                    tool_choice="required",
                )

                return query_idx, result_idx, result, compressed_result

            except Exception as e:
                logger.warning(
                    f"Content compression failed for result {result.get('url', 'unknown')}: {e}"
                )
                # Return original result with empty summary on failure
                return query_idx, result_idx, result, None

    # Create compression tasks with metadata preserved
    compression_tasks = []
    for query_idx, (query, results) in enumerate(zip(queries, filtered_web_results)):
        for result_idx, result in enumerate(results["results"]):
            compression_tasks.append(
                compress_content_with_metadata(query_idx, result_idx, query, result)
            )

    # Execute all compression tasks with error handling
    try:
        compression_results = await asyncio.gather(
            *compression_tasks, return_exceptions=True
        )
    except Exception as e:
        logger.error(f"Batch compression failed: {e}")
        return {"compressed_web_results": [{"results": []} for _ in queries]}

    # Organize results back into the original structure
    final_results = [{"results": []} for _ in range(len(queries))]

    successful_compressions = 0
    for compression_result in compression_results:
        if isinstance(compression_result, Exception):
            logger.warning(f"Compression exception: {compression_result}")
            continue

        query_idx, result_idx, original_result, compressed_result = compression_result

        if compressed_result is not None:
            # Process successful compression
            summary_content = ""
            for tool_call in compressed_result.tool_calls:
                summary_content += (
                    tool_call["args"]["summary_content"] + "====" + "\n\n"
                )

            new_result = copy.deepcopy(original_result)
            new_result["raw_content"] = summary_content
            final_results[query_idx]["results"].append(new_result)
            successful_compressions += 1
        else:
            # Use original content for failed compressions
            final_results[query_idx]["results"].append(original_result)

    logger.info(
        f"Content compression completed: {successful_compressions}/{len(compression_tasks)} successful"
    )
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
    if state["curr_num_iterations"] >= state["max_num_iterations"]:
        return Command(goto=END)

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
