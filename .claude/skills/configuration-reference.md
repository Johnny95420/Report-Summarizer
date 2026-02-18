# Configuration Reference

## Configuration Hierarchy

```
.env                    # API keys and service endpoints (loaded first)
report_config.yaml      # Model names and report structure
retriever_config.yaml   # RAG retriever settings
RunnableConfig          # Runtime execution parameters (per-invocation)
```

All YAML files are loaded at **module import time** via `OmegaConf.load()`. Changes require module reimport or application restart.

## Environment Variables (`.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key (for GPT models via LiteLLM) |
| `GEMINI_API_KEY` | Yes* | - | Google Gemini API key |
| `DEEPSEEK_API_KEY` | Yes* | - | DeepSeek API key |
| `REPLICATE_API_KEY` | No | - | Replicate API key |
| `TAVILY_API_KEY` | Yes | - | Tavily web search API key |
| `SEARCH_HOST` | Yes** | `None` | Hostname for Selenium scraping service |
| `SEARCH_PORT` | Yes** | `None` | Port for Selenium scraping service |
| `TEMP_DIR` | No | `"./temp"` | Directory for temporary files from web search |

\* Required if using those providers in `report_config.yaml`
\** Required if `use_web=True` in runtime config

**Security**: `.env` is in `.gitignore`. Never commit API keys.

## Report Configuration (`report_config.yaml`)

### Model Name Fields

Every model has a primary and backup. The backup is used when the primary fails (via `with_fallbacks()`).

| Field | Used By | Purpose |
|---|---|---|
| `PLANNER_MODEL_NAME` | `report_writer.py` | Generate report section plan |
| `BACKUP_PLANNER_MODEL_NAME` | `report_writer.py` | Fallback for planner |
| `MODEL_NAME` | Both | General queries, search budget, grading |
| `BACKUP_MODEL_NAME` | Both | Fallback for general model |
| `VERIFY_MODEL_NAME` | Both | Section quality grading |
| `BACKUP_VERIFY_MODEL_NAME` | Both | Fallback for verifier |
| `WRITER_MODEL_NAME` | `report_writer.py` | Section content writing, refinement |
| `BACKUP_WRITER_MODEL_NAME` | `report_writer.py` | Fallback for writer |
| `CONCLUDE_MODEL_NAME` | `report_writer.py` | Final sections (intro, conclusion) |
| `BACKUP_CONCLUDE_MODEL_NAME` | `report_writer.py` | Fallback for conclusion writer |
| `LIGHT_MODEL_NAME` | `agentic_search.py` | Short content processing (<4K tokens) |
| `BACKUP_LIGHT_MODEL_NAME` | `agentic_search.py` | Fallback for light model |

### Temperature Handling

Temperature is set automatically in `call_llm()`:
- **Temperature 1.0**: for reasoning models in `except_model_name` set: `{"o3-mini", "o4-mini", "gpt-5", "gpt-5-nano", "gpt-5-mini"}`
- **Temperature 0.5**: for all other models

### Report Structure Field

`REPORT_STRUCTURE` defines the template for report organization:

```yaml
REPORT_STRUCTURE: |
  Use this structure and Traditional Chinese to create a report on the user-provided topic:

  1. Brief Summary (No Research Needed)
  2. Main Body Sections (With Subtopics and Research)
  3. Future Areas of Focus (No Research Needed)
```

This value becomes `DEFAULT_REPORT_STRUCTURE` in `report_writer.py` and can be overridden per-invocation via `RunnableConfig`.

### Example Configuration

```yaml
PLANNER_MODEL_NAME: "gpt-4o"
BACKUP_PLANNER_MODEL_NAME: "gpt-4o-mini"
LIGHT_MODEL_NAME: "gpt-4o-mini"
BACKUP_LIGHT_MODEL_NAME: "gpt-4o-mini"
VERIFY_MODEL_NAME: "gpt-4o"
BACKUP_VERIFY_MODEL_NAME: "gpt-4o-mini"
MODEL_NAME: "gpt-4o"
BACKUP_MODEL_NAME: "gpt-4o-mini"
WRITER_MODEL_NAME: "gpt-4o"
BACKUP_WRITER_MODEL_NAME: "gpt-4o-mini"
CONCLUDE_MODEL_NAME: "gpt-4o"
BACKUP_CONCLUDE_MODEL_NAME: "gpt-4o-mini"
REPORT_STRUCTURE: |
  Use this structure and Traditional Chinese...
```

## Retriever Configuration (`retriever_config.yaml`)

| Field | Type | Default | Description |
|---|---|---|---|
| `raw_file_path` | list[str] \| null | - | Directories containing preprocessed JSON files. Set to `null` to skip loading (disables local retrieval) |
| `split_chunk_size` | int | 1500 | Character count per text chunk for splitting |
| `split_chunk_overlap` | int | 250 | Character overlap between adjacent chunks |
| `embedding_model` | str | `"BAAI/bge-m3"` | HuggingFace embedding model name |
| `top_k` | int | 5 | Number of results per retriever (vector and BM25 each return top_k) |
| `hybrid_weight` | list[float] | `[0.4, 0.6]` | Ensemble weights: `[vector_weight, bm25_weight]`. Must sum to 1.0 |

### Example Configuration

```yaml
raw_file_path:
  - "/data/preprocessed/financial_reports/"
  - "/data/preprocessed/industry_analysis/"
split_chunk_size: 1500
split_chunk_overlap: 250
embedding_model: "BAAI/bge-m3"
top_k: 5
hybrid_weight: [0.4, 0.6]
```

## Runtime Configuration (`RunnableConfig`)

Passed per-invocation when streaming or invoking the graph:

```python
config = RunnableConfig({
    "configurable": {
        "thread_id": str,            # Unique ID for checkpointing/resumption
        "number_of_queries": int,    # Queries generated per stage (e.g., 5)
        "use_web": bool,             # Enable web search via Selenium
        "use_local_db": bool,        # Enable local RAG retrieval
        "max_search_depth": int,     # Max search iterations per section (e.g., 3)
        "report_structure": str,     # Report structure template (overrides YAML)
        "refine_iteration": int,     # Number of refinement loops (from input state)
    }
})
```

### Parameter Details

| Parameter | Required | Description |
|---|---|---|
| `thread_id` | Yes | Unique identifier for the research task. Used by checkpointer for state persistence and resumption |
| `number_of_queries` | Yes | How many search queries to generate at each stage (planning, section research, refinement) |
| `use_web` | Yes | Whether to search the web via Selenium/agentic search. At least one of `use_web`/`use_local_db` must be True |
| `use_local_db` | Yes | Whether to search the local RAG database. At least one of `use_web`/`use_local_db` must be True |
| `max_search_depth` | Yes | Maximum recursive search iterations per section. If a section fails quality grading, it re-searches up to this limit |
| `report_structure` | Yes | Report organization template. Can override the YAML `REPORT_STRUCTURE`. Supports dict or string format |

### Input State Parameters

These are provided via `ReportStateInput`:

```python
input_data = ReportStateInput(
    topic="Your research topic...",    # The subject to research
    refine_iteration=1,                # Number of refinement loops (0 = no refinement)
)
```

## Code-Level Constants

These are hardcoded in the source and not configurable via files:

| Constant | Location | Value | Description |
|---|---|---|---|
| `except_model_name` | `Utils/utils.py` | `{"o3-mini", "o4-mini", "gpt-5", ...}` | Models that use temperature=1.0 |
| Token threshold | `agentic_search.py` | 4096 | Threshold for light vs heavy model selection |
| Quality score threshold | `agentic_search.py` | >2 | Minimum score to pass quality filter (1-5 scale) |
| Max LLM retries | `report_writer.py` | 5 | Retry limit for `_call_llm_with_retry` |
| Token overflow limit | `report_writer.py` | 120,000 | Max tokens before source truncation |
| Source truncation step | `report_writer.py` | 1,500 chars | Characters removed per truncation iteration |
| Raw content size limit | `Utils/utils.py` | 70,000 chars | Web results exceeding this are truncated to 20K |
| Large file threshold | `Utils/utils.py` | 5,000 chars | Web results exceeding this are saved to temp files |
| Table size limit | `retriever.py` | 100,000 chars | Tables exceeding this are skipped |
| HTTP retry | `Utils/utils.py` | 3 | Max HTTP request retries |
| HTTP backoff | `Utils/utils.py` | 1 (factor) | Exponential backoff base for HTTP retries |
| Async semaphore | `agentic_search.py` | 2 | Concurrent LLM call limit |
| ContentExtractor chunk | `Utils/utils.py` | 300 | Chunk size for dynamic web content indexing |
| ContentExtractor k | `Utils/utils.py` | 3 | Top-k for dynamic content retrieval |
| ContentExtractor weights | `Utils/utils.py` | [0.8, 0.2] | Vector-heavy ensemble for web content |
| Context expansion fwd | `report_writer.py` | 1,500 chars | Forward expansion from matched chunk |
| Context expansion bwd | `report_writer.py` | 1,000 chars | Backward expansion from matched chunk |

## Test Configuration (`tests/conftest.py`)

Tests use a `FAKE_CONFIG` dict that satisfies all module-level `OmegaConf.load()` calls:

```python
FAKE_CONFIG = {
    "PLANNER_MODEL_NAME": "test-model",
    "BACKUP_PLANNER_MODEL_NAME": "test-model",
    # ... all model names set to "test-model"
    "REPORT_STRUCTURE": "default",
    "raw_file_path": None,  # Skips data loading in retriever
}
```

This is auto-applied via `autouse=True` fixtures.
