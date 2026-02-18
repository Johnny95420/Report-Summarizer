# CLAUDE.md - Report-Summarizer (Deep Research Agent)

## Project Overview

A LangGraph-based multi-agent system for automated, in-depth research report generation. Combines Retrieval-Augmented Generation (RAG), agentic web search, and human-in-the-loop verification to produce professional financial/investment analysis reports.

**Domain**: Financial analysis, Taiwan stock market, US macro economics, industry research (primarily Traditional Chinese output).

## Quick Reference

### Essential Commands

```bash
# Install dependencies
poetry install

# Run tests
pytest tests/

# Lint & format
ruff check --fix . && ruff format .

# Run all pre-commit hooks
pre-commit run --all-files
```

### Key Entry Points

| File | Purpose |
|---|---|
| `report_writer.py` | Main orchestrator - multi-agent report generation workflow |
| `agentic_search.py` | Autonomous iterative web search engine |
| `retriever.py` | Hybrid RAG retriever (vector + BM25) |
| `preprocess_files.py` | PDF/audio preprocessing for RAG |

### Required Config Files (not in git)

| File | Purpose |
|---|---|
| `.env` | API keys: `OPENAI_API_KEY`, `TAVILY_API_KEY`, `SEARCH_HOST`, `SEARCH_PORT`, etc. |
| `report_config.yaml` | Model names (12 fields), report structure template |
| `retriever_config.yaml` | RAG settings: file paths, chunk size, embedding model, hybrid weights |

---

## Tech Stack

- **Python 3.12** (strict: `>=3.12,<3.13`)
- **LangGraph** + **LangChain** ecosystem for agent orchestration
- **LiteLLM** for multi-provider LLM access (OpenAI, Gemini, DeepSeek)
- **ChromaDB** + **BM25** for hybrid retrieval
- **BAAI/bge-m3** multilingual embeddings
- **Selenium** + **undetected-chromedriver** for web scraping
- **Ruff** for linting/formatting, **pytest** for testing
- **Poetry** for dependency management
- **Docker** with NVIDIA PyTorch base image for GPU workloads

---

## Project Structure

```
Report-Summarizer/
  report_writer.py          # Main LangGraph workflow (planning, writing, refinement)
  agentic_search.py         # Autonomous web search subgraph
  retriever.py              # Hybrid RAG retriever (ChromaDB + BM25)
  preprocess_files.py       # Data preprocessing entry point
  Prompt/
    industry_prompt.py      # Report generation prompt templates (~567 lines)
    agentic_search_prompt.py # Search workflow prompts (~221 lines)
  State/
    state.py                # TypedDict state schemas + Pydantic models
  Tools/
    tools.py                # @tool decorators for structured LLM output
  Utils/
    utils.py                # Core utilities (LLM calls, search, formatting)
    pdf_processor.py        # PDF-to-JSON conversion pipeline
    audio_processor.py      # Audio transcription (under development)
    selenium_searching_api.py # FastAPI web scraping service
  tests/
    conftest.py             # Shared fixtures (auto-mocks OmegaConf, dotenv, env vars)
    test_report_writer.py   # AST-based implementation verification
    test_prompts.py         # Prompt template validation
    test_utils.py           # Utility function tests
    test_project_config.py  # Configuration pattern tests
    test_run_industry_report.py # End-to-end integration test
```

---

## Architecture Summary

The system uses **nested LangGraph subgraphs**:

```
Main Graph (ReportState)
  generate_report_plan -> human_feedback (interrupt) -> [parallel sections via Send()]
    Section Subgraph (SectionState)
      generate_queries -> search_db -> write_section -> [pass: END / fail: loop to search_db]
        Agentic Search Subgraph (AgenticSearchState)
          web_search -> filter -> compress -> aggregate -> [pass: END / fail: loop]
  -> route -> [refine_sections loop] -> gather_complete_section
  -> write_final_sections (intro, conclusion) -> compile_final_report -> END
```

Key architectural patterns:
- **Send() API** for dynamic parallel section processing
- **Command() API** for conditional routing with state updates
- **interrupt()** for human-in-the-loop feedback gates
- **SQLite checkpointing** for pause/resume of long-running workflows
- **Primary + backup model** with automatic failover for every LLM call
- **Tool-based structured output** via `@tool` decorators with `tool_choice="required"`

> **Deep dive**: See `.claude/skills/architecture.md` for full graph structure, state schemas, reducers, and concurrency model.

---

## Coding Standards

### Ruff Configuration

- **Target**: Python 3.12, line length 120
- **Rules**: F (Pyflakes), E/W (pycodestyle), I (isort), N (pep8-naming), UP (pyupgrade), B (flake8-bugbear), SIM (flake8-simplify), T20 (flake8-print), RUF (Ruff-specific)
- **Ignored**: `E501` (line length by formatter), `E402` (load_dotenv before imports), `T201` (print allowed), `F403`/`F405` (star imports in Prompt/), `RUF001` (unicode in prompts)
- **Per-file**: `tests/*` ignores `RUF012`, `SIM102`; `Prompt/*` ignores `RUF001`
- **Format**: double quotes, space indent, docstring code formatting enabled

### Pre-commit Hooks

Runs on every commit:
1. File hygiene: trailing whitespace, EOF newline, YAML/TOML/JSON syntax, large file check (>1024KB), merge conflict detection, private key detection
2. Ruff lint with `--fix`
3. Ruff format

### Naming Conventions

- Functions: `snake_case`, private with `_prefix`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Test classes: `Test` + PascalCase; test methods: `test_` + snake_case

### Import Rules

- First-party modules: `Prompt`, `State`, `Tools`, `Utils` (configured in ruff isort)
- `report_writer.py`: **explicit imports only** from `Prompt/` (no wildcard) - enforced by tests
- `load_dotenv()` must execute before imports needing env vars (hence `E402` ignore)
- Config paths: always use `pathlib.Path(__file__).parent` - enforced by tests

### Critical Patterns

- **Always `deepcopy()`** documents before mutating metadata (enforced by tests)
- **Prompts use "words" not "tokens"** for length specifications (enforced by tests)
- **XML tags in prompts must be closed** (enforced by tests)
- **All `.format()` calls must include all required keywords** (enforced by tests)
- **LLM temperature**: 1.0 for reasoning models (`o3-mini`, `o4-mini`, `gpt-5*`), 0.5 for others

> **Deep dive**: See `.claude/skills/code-patterns.md` for LLM calling patterns, error handling, async patterns, logging conventions, and data processing patterns.

---

## Testing

Tests run without GPU, config files, API keys, or running services. All heavy dependencies are auto-mocked.

```bash
pytest tests/          # Run all
pytest tests/ -v       # Verbose
pytest tests/test_report_writer.py::TestSearchRelevanceDoc  # Specific test
```

### Testing Approach

- **AST-based code inspection**: Verify implementation patterns (deepcopy usage, format() keywords, import style) without executing LLM calls
- **Source text inspection**: Check for typos, logger levels, deprecated API usage
- **Integration tests**: Full LangGraph workflow with mocked LLM/search (uses `AsyncSqliteSaver` in-memory)
- **Prompt validation**: XML tag closure, length unit enforcement

### Key Test Fixtures (`conftest.py`)

Three `autouse=True` fixtures:
- `_mock_omegaconf`: `OmegaConf.load()` returns `FAKE_CONFIG` (all models = "test-model", `raw_file_path=None`)
- `_mock_dotenv`: `load_dotenv()` is a no-op
- `_mock_env_vars`: Sets dummy `TAVILY_API_KEY` and `OPENAI_API_KEY`

> **Deep dive**: See `.claude/skills/testing-guide.md` for writing new tests, mock patterns, integration test setup, and AST inspection techniques.

---

## Prompt Engineering

All prompts use **XML-style tags** (`<Task>`, `<Context>`, etc.) and **Python `.format()` placeholders**.

| Prompt File | Content |
|---|---|
| `Prompt/industry_prompt.py` | Report planning, section writing, grading, refinement, content polish (~567 lines) |
| `Prompt/agentic_search_prompt.py` | Search budget, query rewriting, quality filtering, content compression (~221 lines) |

Key rules:
- Language: Traditional Chinese for Taiwan/financial topics, English for global
- Persona: Senior institutional research analyst (J.P. Morgan standard)
- Source integrity: **zero tolerance for fabricated citations** - every `[Source Title]` must exactly match provided sources
- Current time injection: `curr_date` appended to all industry prompts at import time

> **Deep dive**: See `.claude/skills/prompt-engineering.md` for full prompt reference, XML tag conventions, format placeholder details, and rules for modifying prompts.

---

## Detailed Skill Files

For progressive deep dives, see these files in `.claude/skills/`:

| Skill | What You'll Learn |
|---|---|
| **`architecture.md`** | Full graph structure, state schemas, custom reducers, Send/Command/interrupt patterns, checkpointing, concurrency model |
| **`testing-guide.md`** | How to run tests, write new tests, AST inspection patterns, integration test setup, mock strategies |
| **`prompt-engineering.md`** | Complete prompt reference, XML conventions, format placeholders, language rules, modification guidelines |
| **`development-setup.md`** | Docker setup, Poetry install, ML packages, config file creation, running the application |
| **`code-patterns.md`** | LLM calling with fallbacks, tool-based output, retry/backoff, async semaphores, logging, deep copy rules |
| **`agentic-search-guide.md`** | Search graph nodes, quality scoring pipeline, content compression, Selenium service, large content handling |
| **`rag-retrieval-guide.md`** | Hybrid retrieval architecture, context expansion strategy, document schemas, ContentExtractor, PDF processing |
| **`configuration-reference.md`** | All config fields with types/defaults, model temperature rules, runtime config parameters, hardcoded constants |
