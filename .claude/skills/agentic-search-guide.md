# Agentic Search System

## Overview

The agentic search system (`agentic_search.py`) is an autonomous, iterative web search engine that dynamically generates follow-up queries to build comprehensive research context. It operates as a standalone LangGraph subgraph invoked by the main report writer.

## Graph Structure

```
AgenticSearchGraphBuilder.get_graph()
    START
      -> get_searching_budget           # Determine iteration depth
      -> perform_web_search             # Execute web queries via Selenium
      -> filter_and_format_results      # Async quality scoring
      -> compress_raw_content           # LLM-based content compression
      -> aggregate_final_results        # Format and accumulate results
      -> check_searching_results        # Grade sufficiency (pass/fail)
         -> END                         # If sufficient or max iterations
         -> perform_web_search          # If insufficient, loop with follow-ups
```

## State Schema

```python
class AgenticSearchState(TypedDict):
    queries: list[str]                  # Original input queries
    followed_up_queries: list[str]      # Auto-generated follow-up queries
    web_results: list[dict]             # Raw search results per query
    filtered_web_results: list[dict]    # Quality-filtered results
    compressed_web_results: list[dict]  # Compressed content results
    source_str: str                     # Accumulated formatted output
    max_num_iterations: int             # Budget (currently hardcoded to 1)
    curr_num_iterations: int            # Current iteration count
    url_memo: set[str]                  # URL deduplication set
```

## Node Details

### 1. `get_searching_budget`

**Purpose**: Determine how many search iterations to perform.

**Current behavior**: Hardcoded to `budget_value = 1`. The LLM-based budget allocation (1-3 iterations based on query complexity) is commented out as a TODO.

**Original design** (commented out):
- Simple factual queries -> budget 1
- Composite topics -> budget 2
- Complex multi-faceted analysis -> budget 3

### 2. `perform_web_search`

**Purpose**: Execute web searches via the Selenium API service.

**Behavior**:
- First iteration: uses `state["queries"]` (original queries)
- Follow-up iterations: uses `state["followed_up_queries"]`
- Calls `selenium_api_search(queries, include_raw_content=True)`
- Deduplicates results using `url_memo` set
- Increments `curr_num_iterations`

**Input/Output flow**:
```
queries -> selenium_api_search -> deduplicate by url_memo -> web_results
```

### 3. `filter_and_format_results`

**Purpose**: Async quality filtering of search results using LLM scoring.

**Behavior**:
- For each (query, result) pair, calls `check_search_quality_async()`
- Quality scoring: 1-5 scale via `quality_formatter` tool
- **Threshold**: score > 2 to pass
- Uses `Semaphore(2)` for concurrent LLM calls
- Failed quality checks default to score 0

**Model selection**: Dynamic based on document token length:
- Content > 4096 tokens -> heavy model (`MODEL_NAME`)
- Content <= 4096 tokens -> light model (`LIGHT_MODEL_NAME`)

**Error handling**: `asyncio.gather(*tasks, return_exceptions=True)` - exceptions logged but don't break the pipeline.

### 4. `compress_raw_content`

**Purpose**: LLM-based content compression to reduce context size while preserving key details.

**Behavior**:
- For each filtered result, generates a compressed summary
- Uses `results_compress_instruction` prompt + `summary_formatter` tool
- Replaces `raw_content` with compressed summary
- `Semaphore(2)` for concurrent operations
- Failed compressions fall back to original content

**Key compression rules** (from prompt):
- Preserve: proper nouns, dates, technical specs, numbers, key statements
- Remove: navigation, ads, boilerplate, unrelated content
- Format: thematic grouping with Markdown lists (no tables)
- Source metadata: exact preservation required

### 5. `aggregate_final_results`

**Purpose**: Format compressed results into a unified source string.

**Behavior**:
- Calls `web_search_deduplicate_and_format_sources(compressed_web_results, True)`
- Appends to existing `source_str` (accumulates across iterations)

**Output format**:
```
Sources:

Source [Title]:
===
URL: [url]
===
Most relevant content from source: [content]
===
[raw_content]
```

### 6. `check_searching_results`

**Purpose**: Grade whether accumulated results sufficiently answer the queries.

**Behavior**:
- If `curr_num_iterations >= max_num_iterations` -> END (early exit)
- Otherwise, uses `searching_results_grader` prompt + `searching_grader_formatter` tool
- **Pass**: go to END
- **Fail**: generate follow-up queries, loop back to `perform_web_search`

**Grading criteria**:
- Complete: directly answers core question, sufficient depth, high confidence
- Incomplete: partial answer, tangential, lacks specificity, raises new questions

## Invocation

The agentic search graph is invoked from `report_writer.py`'s `search_db` node:

```python
# In report_writer.py search_db():
search_results = await agentic_search_graph.ainvoke({"queries": query_list})
source_str = search_results["source_str"]
```

The module-level singleton:
```python
agentic_search_graph_builder = AgenticSearchGraphBuilder()
agentic_search_graph = agentic_search_graph_builder.get_graph()
```

## Web Search Backend

The agentic search relies on `selenium_api_search()` from `Utils/utils.py`, which calls the Selenium scraping FastAPI service:

```
GET http://{SEARCH_HOST}:{SEARCH_PORT}/search_and_crawl
    ?query=...
    &include_raw_content=true
    &max_results=5
    &timeout=600
```

### Large Content Handling

When `include_raw_content=True`:
1. Results > 70,000 chars are truncated to 20,000
2. Results > 5,000 chars are saved to temp files
3. Temp files are indexed by `ContentExtractor` (hybrid vector + BM25)
4. Most relevant chunks are re-injected into results

## Selenium Scraping Service (`Utils/selenium_searching_api.py`)

Standalone FastAPI service with:

- **`DriverPool`**: Pool of `undetected_chromedriver` instances for anti-bot evasion
  - Max drivers: configurable via `MAX_DRIVERS`
  - Health checks: automatic restart on failure
  - Thread-safe queue-based resource management
  - Cleanup via `atexit` handler
- **Content extraction**: Trafilatura for HTML-to-markdown conversion
- **Endpoint**: `GET /search_and_crawl`

## Performance Characteristics

- Semaphore(2) limits concurrent LLM calls in filter and compress stages
- URL deduplication prevents re-fetching across iterations
- Dynamic model selection reduces cost for short content
- Content compression reduces downstream token usage
- Temp file indexing handles large web pages without memory overflow
