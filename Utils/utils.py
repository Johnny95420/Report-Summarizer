import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import omegaconf
import requests
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_litellm import ChatLiteLLM
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, RequestException, Timeout
from tavily import TavilyClient
from urllib3.util.retry import Retry

from State.state import Section

_cfg = omegaconf.OmegaConf.load(Path(__file__).parent.parent / "report_config.yaml")
_MAX_TOKENS: int = int(_cfg.get("MAX_TOKENS", 65536))

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

# Timeout and retry constants for the decoupled /search and /crawl endpoints
_MAX_RETRIES = 3
_SEARCH_MAX_RESULTS = 10
_SEARCH_SERVICE_TIMEOUT = 30   # passed as ?timeout= to GET /search
_SEARCH_HTTP_TIMEOUT = 45      # requests client timeout for GET /search

_CRAWL_SERVICE_TIMEOUT = 60    # passed as "timeout" in POST /crawl body (per-URL)
_CRAWL_HTTP_TIMEOUT = 180      # requests client timeout for POST /crawl (server crawls in parallel)

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
        max_tokens=_MAX_TOKENS,
    )

    if tool:
        primary = primary.bind_tools(tools=tool, tool_choice=tool_choice)

    def _validate_tool_calls(msg):
        if tool and tool_choice == "required" and not getattr(msg, "tool_calls", None):
            raise ValueError("Required tool call missing")
        content = getattr(msg, "content", None)
        effective_content = content.strip() if isinstance(content, str) else content
        if not (effective_content or getattr(msg, "tool_calls", None)):
            raise ValueError("Empty model output")
        finish_reason = (getattr(msg, "response_metadata", None) or {}).get("finish_reason")
        if finish_reason in ("length", "max_tokens"):
            raise ValueError("Output truncated by token limit")
        return msg

    validated_primary = primary | RunnableLambda(_validate_tool_calls)

    backup = ChatLiteLLM(
        model=backup_model_name,
        temperature=backup_temperature,
        max_tokens=_MAX_TOKENS,
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
        max_tokens=_MAX_TOKENS,
    )

    if tool:
        primary = primary.bind_tools(tools=tool, tool_choice=tool_choice)

    def _validate_tool_calls(msg):
        if tool and tool_choice == "required" and not getattr(msg, "tool_calls", None):
            raise ValueError("Required tool call missing")
        content = getattr(msg, "content", None)
        effective_content = content.strip() if isinstance(content, str) else content
        if not (effective_content or getattr(msg, "tool_calls", None)):
            raise ValueError("Empty model output")
        finish_reason = (getattr(msg, "response_metadata", None) or {}).get("finish_reason")
        if finish_reason in ("length", "max_tokens"):
            raise ValueError("Output truncated by token limit")
        return msg

    validated_primary = primary | RunnableLambda(_validate_tool_calls)

    backup = ChatLiteLLM(
        model=backup_model_name,
        temperature=backup_temperature,
        max_tokens=_MAX_TOKENS,
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


def _collect_unique_urls(batches: list[dict]) -> list[str]:
    """Return ordered list of unique URLs from a list of {"results": [...]} batch dicts."""
    seen: set[str] = set()
    urls: list[str] = []
    for batch in batches:
        for result in batch.get("results", []):
            url = result.get("url") or ""
            if url and url not in seen:
                urls.append(url)
                seen.add(url)
    return urls


def _search_one(
    base_url: str,
    query: str,
    time_filter: str,
    gl: str,
    hl: str,
    max_results: int = _SEARCH_MAX_RESULTS,
) -> dict:
    """GET /search for a single query. Returns {"results": [...]} or {"results": []} on failure."""
    params = {
        "query": query,
        "max_results": max_results,
        "timeout": _SEARCH_SERVICE_TIMEOUT,
        "gl": gl,
        "hl": hl,
    }
    if time_filter != "all":
        params["time_filter"] = time_filter

    for attempt in range(_MAX_RETRIES):
        try:
            logger.info(f"GET /search query='{query}' attempt {attempt + 1}")
            response = http_session.get(
                f"{base_url}/search", params=params, timeout=_SEARCH_HTTP_TIMEOUT
            )
            response.raise_for_status()
            try:
                return response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse /search JSON for '{query}': {e}")
                if attempt == _MAX_RETRIES - 1:
                    return {"results": []}
                continue
        except Timeout:
            logger.warning(f"/search timeout for '{query}' attempt {attempt + 1}")
            if attempt == _MAX_RETRIES - 1:
                logger.error(f"Max retries exceeded for /search '{query}' (timeout)")
                return {"results": []}
        except ConnectionError:
            logger.warning(f"/search connection error for '{query}' attempt {attempt + 1}")
            if attempt == _MAX_RETRIES - 1:
                logger.error(f"Max retries exceeded for /search '{query}' (connection error)")
                return {"results": []}
        except RequestException as e:
            logger.error(f"/search request error for '{query}': {e}")
            if attempt == _MAX_RETRIES - 1:
                return {"results": []}
        time.sleep(2 ** attempt)
    return {"results": []}  # unreachable with _MAX_RETRIES > 0; guards against future refactors


def call_search_api(
    search_queries: list[str],
    time_filter: str = "month",
    gl: str = "tw",
    hl: str = "zh-tw",
    max_results: int = _SEARCH_MAX_RESULTS,
) -> list[dict]:
    """Call GET /search for each query.

    Returns list[{"results": [{"title", "url", "content"}]}].
    Results do NOT contain raw_content — call call_crawl_api separately if needed.
    """
    host = os.environ.get("SEARCH_HOST")
    port = os.environ.get("SEARCH_PORT")
    if not host or not port:
        logger.error("SEARCH_HOST and SEARCH_PORT environment variables are required")
        return [{"results": []} for _ in search_queries]

    base_url = f"http://{host}:{port}"
    results: list[dict] = [{}] * len(search_queries)
    with ThreadPoolExecutor(max_workers=len(search_queries)) as pool:
        futures = {
            pool.submit(_search_one, base_url, q, time_filter=time_filter, gl=gl, hl=hl, max_results=max_results): i
            for i, q in enumerate(search_queries)
        }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return results


def call_crawl_api(urls: list[str], crawl_timeout: int = _CRAWL_SERVICE_TIMEOUT) -> dict[str, str | None]:
    """POST /crawl for a batch of URLs (crawled in parallel on the server).

    Returns {url: raw_content} mapping. raw_content is None on crawl failure or bot detection.
    """
    if not urls:
        return {}

    host = os.environ.get("SEARCH_HOST")
    port = os.environ.get("SEARCH_PORT")
    if not host or not port:
        logger.error("SEARCH_HOST and SEARCH_PORT environment variables are required")
        return {u: None for u in urls}

    base_url = f"http://{host}:{port}"
    fallback: dict[str, str | None] = {u: None for u in urls}

    for attempt in range(_MAX_RETRIES):
        try:
            logger.info(f"POST /crawl {len(urls)} URLs attempt {attempt + 1}")
            response = http_session.post(
                f"{base_url}/crawl",
                json={"urls": urls, "timeout": crawl_timeout},
                timeout=_CRAWL_HTTP_TIMEOUT,
            )
            response.raise_for_status()
            try:
                data = response.json()
                return {r["url"]: r.get("raw_content") for r in data.get("results", [])}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse /crawl JSON: {e}")
                if attempt == _MAX_RETRIES - 1:
                    return fallback
                continue
        except Timeout:
            logger.warning(f"/crawl timeout attempt {attempt + 1}")
            if attempt == _MAX_RETRIES - 1:
                logger.error("Max retries exceeded for /crawl (timeout)")
                return fallback
        except ConnectionError:
            logger.warning(f"/crawl connection error attempt {attempt + 1}")
            if attempt == _MAX_RETRIES - 1:
                logger.error("Max retries exceeded for /crawl (connection error)")
                return fallback
        except RequestException as e:
            logger.error(f"/crawl request error: {e}")
            if attempt == _MAX_RETRIES - 1:
                return fallback
        time.sleep(2 ** attempt)
    return fallback  # unreachable with _MAX_RETRIES > 0; guards against future refactors


def call_search_engine(
    search_queries,
    include_raw_content: bool,
    time_filter: str = "month",
    gl: str = "tw",
    hl: str = "zh-tw",
) -> list[dict]:
    """Deprecated: use call_search_api + call_crawl_api separately.

    Kept as a backward-compat wrapper for callers that have not yet been migrated.
    """
    search_docs = call_search_api(search_queries, time_filter=time_filter, gl=gl, hl=hl)

    if not include_raw_content:
        return search_docs

    raw_content_map = call_crawl_api(_collect_unique_urls(search_docs))

    for batch in search_docs:
        for result in batch.get("results", []):
            result["title"] = result.get("title", "").replace("/", "_")  # backward-compat: normalise slash in titles
            result["raw_content"] = raw_content_map.get(result.get("url") or "")

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
