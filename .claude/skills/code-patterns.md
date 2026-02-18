# Code Patterns & Conventions

## Naming Conventions

| Element | Convention | Examples |
|---|---|---|
| Functions | `snake_case` | `generate_report_plan`, `call_llm_async` |
| Private functions | `_leading_underscore` | `_generate_planner_queries`, `_prepare_source_for_writing` |
| Classes | `PascalCase` | `ReportGraphBuilder`, `ContentExtractor`, `DriverPool` |
| Constants | `UPPER_SNAKE_CASE` | `PLANNER_MODEL_NAME`, `MAX_DRIVERS`, `DEFAULT_REPORT_STRUCTURE` |
| Graph node names | `snake_case` (descriptive) | `generate_report_plan`, `build_section_with_web_research` |
| State keys | `snake_case` in TypedDict | `search_iterations`, `source_str`, `completed_sections` |
| Test classes | `Test` + PascalCase description | `TestSearchRelevanceDoc`, `TestPrepareSourceForWriting` |
| Test methods | `test_` + snake_case description | `test_uses_deepcopy_and_appends_expanded_copy` |

## Module-Level Configuration Pattern

All main modules follow this initialization pattern at module scope:

```python
from dotenv import load_dotenv
load_dotenv(".env")                                    # 1. Load env vars first

import pathlib
import omegaconf

_HERE = pathlib.Path(__file__).parent                  # 2. Resolve project root
config = omegaconf.OmegaConf.load(_HERE / "report_config.yaml")  # 3. Load YAML config

MODEL_NAME = config["MODEL_NAME"]                     # 4. Extract as module constants
BACKUP_MODEL_NAME = config["BACKUP_MODEL_NAME"]
# ... more constants
```

**Important**: `load_dotenv()` must run before any imports that read env vars. This is why `E402` (module-level import not at top) is ignored in ruff config.

**Path resolution**: Always use `pathlib.Path(__file__).parent` for config paths (enforced by `test_project_config.py`). Never use relative string paths.

## LLM Calling Pattern

### Primary + Fallback Model

Every LLM call uses a primary model with automatic fallback:

```python
from Utils.utils import call_llm, call_llm_async

# Sync call
result = call_llm(
    MODEL_NAME,              # Primary model
    BACKUP_MODEL_NAME,       # Fallback model
    prompt=[SystemMessage(content=system_instructions)]
    + [HumanMessage(content="Your instruction...")],
    tool=[queries_formatter],         # Optional: tool binding
    tool_choice="required",           # Optional: force tool use
)
```

### Internal Implementation (`Utils/utils.py`)

```python
def call_llm(model_name, backup_model_name, prompt, tool=None, tool_choice=None):
    # Temperature: 1.0 for reasoning models (o3-mini, o4-mini, gpt-5*), 0.5 for others
    temperature = 1 if model_name in except_model_name else 0.5

    primary = ChatLiteLLM(model=model_name, temperature=temperature)
    if tool:
        primary = primary.bind_tools(tools=tool, tool_choice=tool_choice)

    # Validation: ensure tool calls exist when required
    validated_primary = primary | RunnableLambda(_validate_tool_calls)

    backup = ChatLiteLLM(model=backup_model_name, temperature=backup_temperature)
    if tool:
        backup = backup.bind_tools(tools=tool, tool_choice=tool_choice)

    model = validated_primary.with_fallbacks([backup])
    return model.invoke(prompt)
```

### Retry Wrapper

```python
def _call_llm_with_retry(model_name, backup_model_name, messages, tool=None, tool_choice=None, max_retries=5):
    retry = 0
    while retry < max_retries:
        try:
            return call_llm(model_name, backup_model_name, messages, tool=tool, tool_choice=tool_choice)
        except Exception as e:
            retry += 1
            if retry >= max_retries:
                raise
```

### Token-Based Model Selection

In `agentic_search.py`, model selection is dynamic based on content length:

```python
def select_model_based_on_tokens(content, token_threshold=4096):
    content_tokens = get_num_tokens(content, "gpt-4o-mini")
    if content_tokens > token_threshold:
        return MODEL_NAME, BACKUP_MODEL_NAME       # Heavy model
    else:
        return LIGHT_MODEL_NAME, BACKUP_LIGHT_MODEL_NAME  # Light model
```

## Tool-Based Structured Output

LLM outputs are structured via `@tool` decorators in `Tools/tools.py`:

```python
@tool
def queries_formatter(thought: str, queries: list[str]):
    """Take thoughts and a list of queries..."""
    return {"search_queries": queries}
```

Usage pattern:
```python
result = call_llm(
    MODEL_NAME, BACKUP_MODEL_NAME,
    prompt=messages,
    tool=[queries_formatter],
    tool_choice="required",      # Force tool use
)
query_list = result.tool_calls[0]["args"]["queries"]
```

Tools are never executed - they serve as schema definitions for structured LLM output. The `tool_calls` list on the response contains the parsed arguments.

## Error Handling Patterns

### HTTP Retry with Exponential Backoff

```python
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "POST"],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
```

### Web Search Retry

```python
for attempt in range(max_retries):
    try:
        output = http_session.get(url, params=params, timeout=600)
        output.raise_for_status()
        break
    except Timeout:
        time.sleep(retry_delay * (2**attempt))  # Exponential backoff
    except ConnectionError:
        time.sleep(retry_delay * (2**attempt))
```

### Token Overflow Handling

```python
while num_tokens >= 120000 and num_retries < retry_limit:
    source_str = source_str[:-1500]  # Trim 1500 chars from end
    # Recalculate tokens...
    num_retries += 1
```

## Async Patterns

### Semaphore-Limited Concurrency

```python
semaphore = asyncio.Semaphore(2)

async def task_with_limit(item):
    async with semaphore:
        return await process(item)

results = await asyncio.gather(*[task_with_limit(i) for i in items], return_exceptions=True)
```

### Error-Tolerant Gather

```python
results = await asyncio.gather(*tasks, return_exceptions=True)
for result in results:
    if isinstance(result, Exception):
        logger.warning(f"Task failed: {result}")
        continue
    # Process successful result
```

## Import Conventions

### First-Party Module Organization

```python
# Ruff isort recognizes these as first-party:
from Prompt.industry_prompt import ...   # Prompt templates
from State.state import ...              # State definitions
from Tools.tools import ...              # Tool formatters
from Utils.utils import ...              # Utility functions
```

Configured in `pyproject.toml`:
```toml
[tool.ruff.lint.isort]
known-first-party = ["Prompt", "State", "Tools", "Utils"]
```

### Import Rules

- `report_writer.py`: **Explicit imports only** from `Prompt/` (no wildcard `*`), enforced by tests
- `agentic_search.py`: Wildcard `from Prompt.agentic_search_prompt import *` is acceptable
- `E402` ignored: `load_dotenv()` must run before imports that need env vars

## Logging Pattern

Each module creates its own named logger:

```python
import logging

logger = logging.getLogger("AgentLogger")     # report_writer.py (INFO level)
logger = logging.getLogger("AgenticSearch")   # agentic_search.py (ERROR level)
logger = logging.getLogger("Utils")           # utils.py (ERROR level)
logger = logging.getLogger("Retriever")       # retriever.py (ERROR level)

# Standard handler setup
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
```

Note: `report_writer.py` uses INFO level (enforced by test); all other modules use ERROR level.

## Data Processing Patterns

### Deep Copy Before Mutation

When expanding document context, always deepcopy to avoid mutating the original:

```python
return_res = deepcopy(res)
return_res.metadata["content"] = expanded_content
info.append(return_res)
```

This is enforced by tests in `test_report_writer.py` and `test_utils.py`.

### URL Deduplication

Multiple layers of deduplication:
1. `search_relevance_doc`: `seen = set()` tracking `page_content`
2. `perform_web_search`: `url_memo: set[str]` tracking URLs across iterations
3. `web_search_deduplicate_and_format_sources`: dedup by URL, sorted by score

### Large File Handling

When web results exceed 5000 chars, content is written to temp files and indexed via `ContentExtractor`:

```python
if len(result["raw_content"]) >= 70000:
    result["raw_content"] = result["raw_content"][:20000]  # Hard truncate at 70K

if len(result["raw_content"]) >= 5000:
    file_path = f"{temp_files_path}/{result['title']}.txt"
    with open(file_path, "w") as f:
        f.write(result["raw_content"])
    large_files.append(file_path)
    result["raw_content"] = ""  # Clear from result dict

# Re-index and query
content_extractor.update(large_files)
search_results = content_extractor.query(query)
```
