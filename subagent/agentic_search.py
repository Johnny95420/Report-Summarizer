import asyncio
import copy
import logging
import pathlib
import re
import sys

# Ensure project root is on sys.path when run as a script (python subagent/agentic_search.py)
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import omegaconf

_HERE = pathlib.Path(__file__).parent.parent
config = omegaconf.OmegaConf.load(_HERE / "report_config.yaml")

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
from State.agentic_search_state import AgenticSearchState
from Tools.tools import (
    answer_formatter,
    quality_formatter,
    queries_formatter,
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


def _format_sources_section(answer: str, registry: list[dict]) -> str:
    """Re-number [N] citations sequentially and append ### Sources from registry.

    Args:
        answer: Answer body with [N] inline citations but no ### Sources section.
        registry: Ordered list of {title, url}; index+1 = citation number.

    Returns:
        Answer with citations renumbered 1-based and ### Sources appended.
        Returns original answer unchanged if registry is empty or no citations found.
    """
    if not registry or not answer:
        return answer

    cited = sorted(set(int(m) for m in re.findall(r'\[(\d+)\]', answer)))
    if not cited:
        return answer

    remap = {orig: new for new, orig in enumerate(cited, 1)}
    renumbered = re.sub(
        r'\[(\d+)\]',
        lambda m: f"[{remap.get(int(m.group(1)), int(m.group(1)))}]",
        answer,
    )

    lines = ["### Sources"]
    for orig in cited:
        idx = orig - 1  # registry is 0-based
        if 0 <= idx < len(registry):
            lines.append(f"- [{remap[orig]}] {registry[idx]['title']} — {registry[idx]['url']}")

    return renumbered + "\n\n" + "\n".join(lines)


def select_model_based_on_tokens(content: str, token_threshold: int = 4096) -> tuple[str, str]:
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
def queries_rewriter(queries: list[str]) -> list[str]:
    str_queries = ""
    for idx, q in enumerate(queries):
        str_queries += f"{idx}. {q}\n"

    system_instruction = query_rewriter_instruction.format(queries_to_refine=str_queries)
    results = call_llm(
        MODEL_NAME,
        BACKUP_MODEL_NAME,
        prompt=[SystemMessage(content=system_instruction)]
        + [HumanMessage(content="Refine search queries on the provided queries.")],
        tool=[queries_formatter],
        tool_choice="required",
    )
    try:
        queries = results.tool_calls[0]["args"]["queries"]
    except (IndexError, KeyError, TypeError) as e:
        logger.error("Failed to parse rewritten queries: %s", e)
        return queries  # return original queries unchanged
    return queries


async def check_search_quality_async(query: str, document: str) -> int:
    score = None
    system_instruction = results_filter_instruction.format(query=query, document=document)

    results = await call_llm_async(
        LIGHT_MODEL_NAME,
        BACKUP_LIGHT_MODEL_NAME,
        prompt=[SystemMessage(content=system_instruction)]
        + [HumanMessage(content="Generate the score of document on the provided query.")],
        tool=[quality_formatter],
        tool_choice="required",
    )
    try:
        score = results.tool_calls[0]["args"]["score"]
    except (IndexError, KeyError, TypeError) as e:
        logger.warning("Failed to parse quality score: %s", e)
        score = 0
    return score


def get_searching_budget(state: AgenticSearchState):
    # If max_num_iterations is already set in the initial state (e.g. for testing), keep it;
    # otherwise default to 1.
    budget_value = state.get("max_num_iterations") or 1
    return {"max_num_iterations": budget_value}


def generate_queries_from_question(state: AgenticSearchState):
    """Generate keyword-based search queries from the research question."""
    question = state["question"]
    num_queries = state.get("num_queries") or 3

    system_instruction = query_writer_instructions.format(question=question, num_queries=num_queries)
    result = call_llm(
        MODEL_NAME,
        BACKUP_MODEL_NAME,
        prompt=[SystemMessage(content=system_instruction)]
        + [HumanMessage(content="Generate search queries covering all aspects of the research question.")],
        tool=[queries_formatter],
        tool_choice="required",
    )
    try:
        queries = result.tool_calls[0]["args"]["queries"]
    except (IndexError, KeyError, TypeError) as e:
        logger.error("Failed to parse queries from question: %s", e)
        # Fallback: use the question itself as a single query
        queries = [question]

    logger.info("Generated %d queries from question", len(queries))
    return {"queries": queries}


def perform_web_search(state: AgenticSearchState):
    url_memo = state.get("url_memo", set())
    queries = state["queries"]
    curr_num_iterations = state.get("curr_num_iterations", 0)
    followed_up_queries = state.get("followed_up_queries", [])
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
                # Only track real URLs in url_memo; _partN chunk keys are ephemeral
                if result["url"].startswith("http"):
                    url_memo.add(result["url"])
                dedup_results[-1]["results"].append(result)

    return {
        "web_results": dedup_results,
        "curr_num_iterations": curr_num_iterations + 1,
        "url_memo": url_memo,
    }


async def filter_and_format_results(state: AgenticSearchState):
    followed_up_queries = state.get("followed_up_queries", [])
    queries = followed_up_queries if followed_up_queries else state["queries"]
    web_results = state["web_results"]
    logger.info("Filtering and formatting web search results.")

    # Create semaphore to limit concurrent operations
    semaphore = asyncio.Semaphore(5)

    async def check_quality_with_metadata(query: str, result: dict) -> tuple[str, dict, int | None]:
        """Check quality with semaphore control and metadata preservation."""
        async with semaphore:
            try:
                raw = result.get('raw_content') or ""
                raw_preview = raw[:500] + "...[greater than 500 words truncated]" if len(raw) > 500 else raw
                document = (
                    f"Title:{result['title']}\n\nContent:{result['content']}\n\nRaw Content:{raw_preview}"
                )
                score = await check_search_quality_async(query, document)
                return query, result, score
            except Exception as e:
                logger.error(f"Quality check failed for result {result.get('url', 'unknown')}: {e}")
                return query, result, None  # None score: include result rather than silently discard

    # Create tasks with metadata preserved
    quality_tasks = []
    for query, response in zip(queries, web_results):
        for result in response["results"]:
            quality_tasks.append(check_quality_with_metadata(query, result))

    # asyncio.gather with return_exceptions=True never raises; exceptions become list elements
    quality_results = await asyncio.gather(*quality_tasks, return_exceptions=True)

    # Process results with error handling
    results_by_query = {query: [] for query in queries}

    for quality_result in quality_results:
        if isinstance(quality_result, Exception):
            logger.error(f"Quality check exception: {quality_result}")
            continue

        query, search_result, score = quality_result
        if score is None:
            # LLM failed: include result rather than silently discard the entire corpus
            results_by_query[query].append(search_result)
        elif score >= 3:
            search_result["score"] = score
            results_by_query[query].append(search_result)

    # Build final results structure
    filtered_web_results = [{"results": results_by_query[query]} for query in queries]

    total_filtered = sum(len(results["results"]) for results in filtered_web_results)
    logger.info(f"Finished filtering: {total_filtered} results passed quality check.")

    return {"filtered_web_results": filtered_web_results}


async def compress_raw_content(state: AgenticSearchState):
    followed_up_queries = state.get("followed_up_queries", [])
    queries = followed_up_queries if followed_up_queries else state["queries"]
    filtered_web_results = state["filtered_web_results"]

    # Increase semaphore: fewer LLM calls after pass-through means higher concurrency is safe
    semaphore = asyncio.Semaphore(4)

    _COMPRESS_CHAR_THRESHOLD = 5000

    async def compress_content_with_metadata(query_idx: int, result_idx: int, query: str, result: dict):
        """Compress content with semaphore control and metadata preservation.

        Short content (< _COMPRESS_CHAR_THRESHOLD chars) is passed through unchanged to avoid
        unnecessary LLM calls. Only long content goes through LLM compression.
        """
        # Pass-through: short/empty/None content needs no compression
        if len(result.get("raw_content") or "") < _COMPRESS_CHAR_THRESHOLD:
            return query_idx, result_idx, result, "passthrough"

        async with semaphore:
            try:
                document = (
                    f"Title:{result['title']},Brief Content:{result['content']},Full Content:{result['raw_content']}"
                )
                system_instruction = results_compress_instruction.format(query=query, document=document)

                compressed_result = await call_llm_async(
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

                return query_idx, result_idx, result, compressed_result

            except Exception as e:
                logger.error(f"Content compression failed for result {result.get('url', 'unknown')}: {e}")
                return query_idx, result_idx, result, None

    # Create compression tasks with metadata preserved
    compression_tasks = []
    for query_idx, (query, results) in enumerate(zip(queries, filtered_web_results)):
        for result_idx, result in enumerate(results["results"]):
            compression_tasks.append(compress_content_with_metadata(query_idx, result_idx, query, result))

    # asyncio.gather with return_exceptions=True never raises; exceptions become list elements
    compression_results = await asyncio.gather(*compression_tasks, return_exceptions=True)

    # Organize results back into the original structure
    final_results = [{"results": []} for _ in range(len(queries))]

    successful_compressions = 0
    for compression_result in compression_results:
        if isinstance(compression_result, Exception):
            logger.error(f"Compression exception: {compression_result}")
            continue

        query_idx, result_idx, original_result, compressed_result = compression_result

        if compressed_result == "passthrough":
            # Short content: use original result unchanged
            final_results[query_idx]["results"].append(original_result)
            continue

        if compressed_result is not None:
            # Process successful LLM compression
            try:
                summary_content = ""
                for tool_call in compressed_result.tool_calls:
                    summary_content += tool_call["args"]["summary_content"] + "====" + "\n\n"
            except (IndexError, KeyError, TypeError) as e:
                logger.error("Failed to parse compression result: %s", e)
                final_results[query_idx]["results"].append(original_result)
                continue

            new_result = copy.deepcopy(original_result)
            new_result["raw_content"] = summary_content
            final_results[query_idx]["results"].append(new_result)
            successful_compressions += 1
        else:
            # LLM call failed: fall back to original content
            final_results[query_idx]["results"].append(original_result)

    logger.info(f"Content compression completed: {successful_compressions}/{len(compression_tasks)} successful")
    return {"compressed_web_results": final_results}


def aggregate_final_results(state: AgenticSearchState):
    """Format current iteration's search results into materials (reset each round)."""
    compressed_web_results = state["compressed_web_results"]
    materials = web_search_deduplicate_and_format_sources(compressed_web_results, True)
    return {"materials": materials}


def synthesize_answer(state: AgenticSearchState):
    """Synthesize an updated answer from current materials and previous answer."""
    question = state["question"]
    materials = state.get("materials", "")
    previous_answer = state.get("answer", "")

    system_instruction = answer_synthesizer_instructions.format(
        question=question,
        previous_answer=previous_answer,
        materials=materials,
    )

    result = call_llm(
        MODEL_NAME,
        BACKUP_MODEL_NAME,
        prompt=[SystemMessage(content=system_instruction)]
        + [
            HumanMessage(
                content="Synthesize an updated comprehensive answer incorporating the new search materials and previous answer."
            )
        ],
        tool=[answer_formatter],
        tool_choice="required",
    )
    try:
        answer = result.tool_calls[0]["args"]["answer"]
    except (IndexError, KeyError, TypeError) as e:
        logger.error("Failed to parse synthesized answer: %s", e)
        # Fallback: return materials as raw answer
        answer = previous_answer + ("\n\n" if previous_answer else "") + materials

    logger.info("Synthesized answer (iteration %d)", state.get("curr_num_iterations", 0))
    return {"answer": answer}


def check_searching_results(state: AgenticSearchState):
    """Grade the current answer quality and decide whether to continue searching."""
    question = state["question"]
    answer = state.get("answer", "")
    if state["curr_num_iterations"] >= state["max_num_iterations"]:
        return Command(goto=END)

    system_instruction = searching_results_grader.format(question=question, answer=answer)

    feedback = call_llm(
        MODEL_NAME,
        BACKUP_MODEL_NAME,
        prompt=[SystemMessage(content=system_instruction)]
        + [HumanMessage(content="Grade the answer quality and consider follow-up queries for missing information.")],
        tool=[searching_grader_formatter],
        tool_choice="required",
    )
    try:
        feedback = feedback.tool_calls[0]["args"]
    except (IndexError, KeyError, TypeError) as e:
        logger.error("Failed to parse search grader feedback: %s", e)
        return Command(goto=END)
    if feedback["grade"] == "pass":
        return Command(goto=END)
    else:
        follow_up_queries = feedback["follow_up_queries"]
        if isinstance(follow_up_queries, str):
            follow_up_queries = [follow_up_queries]
        return Command(
            update={"followed_up_queries": follow_up_queries},
            goto="perform_web_search",
        )


class AgenticSearchGraphBuilder:
    def __init__(self):
        self._graph = None

    def get_graph(self):
        if self._graph is None:
            builder = StateGraph(AgenticSearchState)
            builder.add_node("get_searching_budget", get_searching_budget)
            builder.add_node("generate_queries_from_question", generate_queries_from_question)
            builder.add_node("perform_web_search", perform_web_search)
            builder.add_node("filter_and_format_results", filter_and_format_results)
            builder.add_node("compress_raw_content", compress_raw_content)
            builder.add_node("aggregate_final_results", aggregate_final_results)
            builder.add_node("synthesize_answer", synthesize_answer)
            builder.add_node("check_searching_results", check_searching_results)

            builder.add_edge(START, "get_searching_budget")
            builder.add_edge("get_searching_budget", "generate_queries_from_question")
            builder.add_edge("generate_queries_from_question", "perform_web_search")
            builder.add_edge("perform_web_search", "filter_and_format_results")
            builder.add_edge("filter_and_format_results", "compress_raw_content")
            builder.add_edge("compress_raw_content", "aggregate_final_results")
            builder.add_edge("aggregate_final_results", "synthesize_answer")
            builder.add_edge("synthesize_answer", "check_searching_results")

            self._graph = builder.compile()
        return self._graph


agentic_search_graph_builder = AgenticSearchGraphBuilder()
agentic_search_graph = agentic_search_graph_builder.get_graph()


if __name__ == "__main__":

    _DEFAULT_QUESTION = (
        "Main Question: What are Tesla's key revenue drivers in 2024?\n"
        "- Sub-question 1: What is the automotive vs. energy revenue split?\n"
        "- Sub-question 2: How did FSD subscription revenue perform?\n"
        "- Sub-question 3: What were the gross margin trends by segment?"
    )
    question = (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else _DEFAULT_QUESTION)

    print(f"[agentic_search] question:\n{question}\n")
    print("=" * 60)

    import time

    num_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    async def _run():
        timings = []
        accumulated = {}  # full state accumulated from all node updates
        t_prev = time.perf_counter()
        t_start = t_prev

        async for event in agentic_search_graph.astream(
            {"question": question, "url_memo": set(), "max_num_iterations": num_iterations},
            stream_mode="updates",
        ):
            t_now = time.perf_counter()
            for node_name, node_updates in event.items():
                elapsed = t_now - t_prev
                timings.append((node_name, elapsed))
                print(f"  [{elapsed:6.2f}s] {node_name}")
                if isinstance(node_updates, dict):
                    accumulated.update(node_updates)
            t_prev = t_now

        total = time.perf_counter() - t_start
        print(f"\n[profiling] total: {total:.2f}s")
        print(f"{'node':<40} {'time(s)':>8}")
        print("-" * 50)
        for node, t in timings:
            print(f"{node:<40} {t:>8.2f}")

        return accumulated

    final_state = asyncio.run(_run())

    print("\n[answer]\n")
    print(final_state.get("answer", "(no answer)"))
    print("\n[url_memo]")
    for url in sorted(final_state.get("url_memo", [])):
        print(f"  {url}")
    print(f"\n[iterations] {final_state.get('curr_num_iterations', '?')}")
