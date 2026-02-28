import json
import logging
import math
import os
import time

import requests
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_litellm import ChatLiteLLM
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, RequestException, Timeout
from tavily import TavilyClient
from urllib3.util.retry import Retry

from State.state import Section

tavily_client = TavilyClient()


# Configure HTTP session with retry strategy
def create_http_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


http_session = create_http_session()

logger = logging.getLogger("Utils")
logger.setLevel(logging.ERROR)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


except_model_name = set(["o3-mini", "o4-mini", "gpt-5", "gpt-5-nano", "gpt-5-mini"])


def call_llm(model_name: str, backup_model_name: str, prompt: list, tool=None, tool_choice=None):
    temperature = 1 if model_name in except_model_name else 0.5
    backup_temperature = 1 if backup_model_name in except_model_name else 0.5

    primary = ChatLiteLLM(
        model=model_name,
        temperature=temperature,
    )

    if tool:
        primary = primary.bind_tools(tools=tool, tool_choice=tool_choice)

    def _validate_tool_calls(msg):
        if tool and tool_choice == "required" and not getattr(msg, "tool_calls", None):
            raise ValueError("Required tool call missing")
        if not (getattr(msg, "content", None) or getattr(msg, "tool_calls", None)):
            raise ValueError("Empty model output")
        return msg

    validated_primary = primary | RunnableLambda(_validate_tool_calls)

    backup = ChatLiteLLM(
        model=backup_model_name,
        temperature=backup_temperature,
    )
    if tool:
        backup = backup.bind_tools(tools=tool, tool_choice=tool_choice)

    model = validated_primary.with_fallbacks([backup])

    return model.invoke(prompt)


async def call_llm_async(model_name: str, backup_model_name: str, prompt: list, tool=None, tool_choice=None):
    temperature = 1 if model_name in except_model_name else 0.5
    backup_temperature = 1 if backup_model_name in except_model_name else 0.5

    primary = ChatLiteLLM(
        model=model_name,
        temperature=temperature,
    )

    if tool:
        primary = primary.bind_tools(tools=tool, tool_choice=tool_choice)

    def _validate_tool_calls(msg):
        if tool and tool_choice == "required" and not getattr(msg, "tool_calls", None):
            raise ValueError("Required tool call missing")
        if not (getattr(msg, "content", None) or getattr(msg, "tool_calls", None)):
            raise ValueError("Empty model output")
        return msg

    validated_primary = primary | RunnableLambda(_validate_tool_calls)

    backup = ChatLiteLLM(
        model=backup_model_name,
        temperature=backup_temperature,
    )
    if tool:
        backup = backup.bind_tools(tools=tool, tool_choice=tool_choice)

    model = validated_primary.with_fallbacks([backup])
    return await model.ainvoke(prompt)


def track_expanded_context(
    original_context: str,
    critical_context: str,
    forward_capacity: int = 10000,
    backward_capacity: int = 2500,
):
    start_idx = original_context.find(critical_context)
    if start_idx != -1:
        end_idx = start_idx + len(critical_context)
        desired_start_idx = max(0, start_idx - backward_capacity)
        desired_end_idx = min(len(original_context), end_idx + forward_capacity)
        start_boundary_pos = original_context.rfind("\n\n", 0, desired_start_idx)
        final_start_idx = 0 if start_boundary_pos == -1 else start_boundary_pos + 2

        end_boundary_pos = original_context.find("\n\n", desired_end_idx)
        final_end_idx = len(original_context) if end_boundary_pos == -1 else end_boundary_pos
        expanded_context = original_context[final_start_idx:final_end_idx]

        return expanded_context

    else:
        logger.critical("Can not find critical content")
        return None


def format_human_feedback(feedbacks: list[str]) -> str:
    """Format a list of human feedbacks into string"""
    formatted_str = ""
    for idx, feedback in enumerate(feedbacks):
        formatted_str += f"""
        {"=" * 60}
        feedback {idx} : {feedback}

        """
    return formatted_str


def format_sections(sections: list[Section]) -> str:
    """Format a list of sections into a string"""
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
        {"=" * 60}
        Section {idx}: {section.name}
        {"=" * 60}
        Description:
        {section.description}
        Requires Research:
        {section.research}

        Content:
        {section.content if section.content else "[Not yet written]"}

        """
    return formatted_str


def format_search_results(results: list[Document], char_limit: int = 500):
    formatted_text = "Sources:\n\n"
    if char_limit is None:
        char_limit = math.inf

    for doc in results:
        formatted_text += f"Source {doc.metadata['path']}:\n===\n"
        formatted_text += "Content from source:"
        raw_content = doc.page_content
        if len(raw_content) > char_limit:
            raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {char_limit} chars: {raw_content}\n\n"
        else:
            formatted_text += f"{raw_content}\n\n"

    return formatted_text


def format_search_results_with_metadata(results: list[Document]):
    formatted_text = "Sources:\n\n"
    for doc in results:
        if "table" in doc.metadata:
            formatted_text += f"Source {doc.metadata['path']}:\n===\n"
            formatted_text += "Report Date:\n"
            formatted_text += doc.metadata["date"]
            formatted_text += "Context Heading:\n"
            formatted_text += doc.metadata["context_heading"]
            formatted_text += "Context Paragraph:\n"
            formatted_text += doc.metadata["context_paragraph"]
            formatted_text += "Summary:\n"
            formatted_text += doc.metadata["summary"]
            formatted_text += "Table Content:\n"
            formatted_text += doc.metadata["table"]

        elif "content" in doc.metadata:
            formatted_text += f"Source {doc.metadata['path']}:\n===\n"
            formatted_text += "Report Date:\n"
            formatted_text += doc.metadata["date"]
            formatted_text += "Source Content:\n"
            formatted_text += doc.metadata["content"]

    return formatted_text


def tavily_search(search_queries, include_raw_content: bool):
    search_docs = []
    for query in search_queries:
        search_docs.append(
            tavily_client.search(
                query,
                max_results=3,
                include_raw_content=include_raw_content,
                topic="general",
            )
        )
    return search_docs


def call_search_engine(search_queries, include_raw_content: bool):
    host = os.environ.get("SEARCH_HOST", None)
    port = os.environ.get("SEARCH_PORT", None)
    memo = set()
    search_docs = []

    # Check if host and port are configured
    if not host or not port:
        logger.error("SEARCH_HOST and SEARCH_PORT environment variables are required")
        return search_docs

    for query in search_queries:
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                logger.info(f"Searching query: {query}, attempt {attempt + 1}")

                output = http_session.get(
                    f"http://{host}:{port}/search_and_crawl",
                    params={
                        "query": query,
                        "include_raw_content": include_raw_content,
                        "max_results": 10,
                        "timeout": 600,
                    },
                    timeout=600,  # Give slightly more time than the service timeout
                )
                output.raise_for_status()  # Raise exception for HTTP errors

                try:
                    output_data = output.json()
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response for query '{query}': {e}")
                    if attempt == max_retries - 1:
                        # Add empty result on final attempt
                        search_docs.append({"results": []})
                    continue

                break  # Success, exit retry loop

            except Timeout as e:
                logger.warning(f"Timeout for query '{query}' on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Max retries exceeded for query '{query}' due to timeout")
                    search_docs.append({"results": []})
                    continue
                time.sleep(retry_delay * (2**attempt))  # Exponential backoff

            except ConnectionError as e:
                logger.warning(f"Connection error for query '{query}' on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Max retries exceeded for query '{query}' due to connection error")
                    search_docs.append({"results": []})
                    continue
                time.sleep(retry_delay * (2**attempt))

            except RequestException as e:
                logger.error(f"Request failed for query '{query}': {e}")
                if attempt == max_retries - 1:
                    search_docs.append({"results": []})
                    continue
                time.sleep(retry_delay * (2**attempt))

        else:
            # If we reach here, the loop completed without breaking (all retries failed)
            continue

        # Process successful response
        if include_raw_content:
            for result in output_data.get("results", []):
                result["title"] = result["title"].replace("/", "_")
                if result.get("raw_content", "") is None:
                    continue
        search_docs.append(output_data)
    return search_docs


def web_search_deduplicate_and_format_sources(search_response, include_raw_content=True):
    # Collect all results
    sources_list = []
    for response in search_response:
        sources_list.extend(response["results"])

    # Deduplicate by URL
    sources_list = sorted(sources_list, key=lambda x: x.get("score", 1), reverse=True)
    unique_sources = {}
    for source in sources_list:
        if source["url"] not in unique_sources:
            unique_sources[source["url"]] = source

    # Format output
    formatted_text = "Sources:\n\n"
    for _i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        url = source['url']
        formatted_text += f"URL: {url if url.startswith('http') else '[content excerpt]'}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            raw_content = source.get("raw_content", "")
            if raw_content is None:
                raw_content = ""
                logger.critical(f"Warning: No raw_content found for source {source['url']}")
            formatted_text += f"{raw_content}\n\n"
    return formatted_text.strip()


# %%
