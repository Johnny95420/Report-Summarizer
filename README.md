# 🧠 Deep Research Agent

> ⚠️ This project is inspired by LangChain's `open_deep_research` and is customized for domain-specific research automation. It features task-tailored prompt engineering and a redesigned web search post-processing pipeline to control context size and ensure cleaner input for the Large Language Models (LLMs) during section generation.

A modular and automated research report generation tool designed for **in-depth topic analysis** using Retrieval-Augmented Generation (RAG), human-in-the-loop verification, and multi-agent LLM coordination. This project is built with [LangGraph](https://github.com/langchain-ai/langgraph) for structured and dynamic execution.

---

## 🚀 Features

- 🔍 **Hybrid Retrieval**: Combines web search with a local database for comprehensive information gathering.
- 🧱 **Flexible Report Structure**: Configure the report structure and prompt style via YAML files.
- 🤖 **Diverse Workflows**:
    - **Deep Report Generation (`report_writer.py`)**: A full-fledged report writing process where multiple agents (Planner, Researcher, Writer, Reviewer) collaborate.
    - **Agentic Search (`subagent/agentic_search.py`)**: A standalone, agent-driven deep search module that dynamically generates follow-up questions for multi-step information exploration.
- 👤 **Human-in-the-Loop**: Supports user feedback to regenerate or revise the report plan.
- 📑 **Parallel Processing**: Capable of generating multiple report sections simultaneously for efficient execution.
- 🕸️ **Enhanced Web Scraping**: Delegates web search and crawling to an external Selenium-based service (not bundled), configured via `SEARCH_HOST` and `SEARCH_PORT` env vars.
- 📄 **Advanced Information Preprocessing**: Includes powerful PDF and audio processing engines to convert unstructured data into RAG-friendly formats.

---

## 📁 File Overview

| File/Folder             | Description                                                                     |
| ----------------------- | ------------------------------------------------------------------------------- |
| `report_writer.py`      | **(Main)** Orchestrates multi-agent collaboration for planning and writing in-depth research reports using LangGraph. |
| `subagent/agentic_search.py` | **(Core Module)** Implements the agentic search logic, enabling autonomous and iterative research. |
| `subagent/document_qa.py` | Document QA agent: ReAct loop with text navigator for financial document analysis. |
| `preprocess_files.py`   | Typer CLI for batch PDF preprocessing. Run as: `python preprocess_files.py INPUT_DIR OUTPUT_DIR [--model] [--no-table]` |
| `retriever.py`          | Implements the hybrid retriever, combining local vector search with keyword search. |
| `Prompt/`               | Contains prompt templates for the industry/stock analysis report style. |
| `State/`                | Defines the state objects used in LangGraph (`ReportState`, `SectionState`, `AgenticSearchState`, `DocumentQAState`). |
| `Tools/`                | LLM output formatters (`tools.py`), `AgentDocumentReader` text navigator with page navigation/search/bookmarks (`text_navigator.py`), Pydantic reader models (`reader_models.py`), and PDF document preprocessors (`document_preprocessors.py`). |
| `Utils/`                | Contains various utility functions, including web API wrappers, PDF/audio processors, and content deduplication. |
| `report_config.yaml`    | **(User-created)** Sets model names, report structure, and generation style. |
| `retriever_config.yaml` | **(User-created)** Configures retriever behavior, text splitting parameters, and the embedding model. |

---

## ⚙️ Configuration

Before running, ensure you have created the necessary configuration files and environment variables.

### 1. `report_config.yaml` (Example)

This file controls the core parameters of the report generator. Create it in the root directory.

```yaml
# --- Example report_config.yaml ---
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
  Use this structure and Traditional Chinese to create a report on the user-provided topic:

  1. Brief Summary (No Research Needed)
  2. Main Body Sections (With Subtopics and Research)
  3. Future Areas of Focus (No Research Needed)
```

### 2. `retriever_config.yaml` (Example)

This file configures the behavior of the local RAG retriever, vector search, and text navigator. Create it in the root directory.

```yaml
# --- Example retriever_config.yaml ---
# IMPORTANT: Update this path to point to your preprocessed data folder.
raw_file_path:
  - "/path/to/your/preprocessed_data/"
split_chunk_size: 1500
split_chunk_overlap: 250
embedding_model: "BAAI/bge-m3"          # Recommended embedding model
top_k: 5
hybrid_weight: [0.4, 0.6]

# AgentDocumentReader settings
navigator_top_k: 5                       # Top-k results for semantic/keyword search
navigator_persist_dir: "navigator_tmp"   # Chroma cache directory (reused across sessions)
reader_tmp_dir: "reader_tmp"             # Preprocessed JSON output directory
```

### 3. `.env`

Create a `.env` file and fill in your API keys and service settings.

```env
OPENAI_API_KEY="your_openai_api_key"
GEMINI_API_KEY="your_gemini_api_key"
DEEPSEEK_API_KEY="your_deepseek_api_key"
REPLICATE_API_KEY="your_replicate_api_key"
TAVILY_API_KEY="your_tavily_api_key"
SEARCH_HOST="localhost" # Host for the Selenium scraping service
SEARCH_PORT="8000"      # Port for the Selenium scraping service
```

---

## 📄 Advanced Information Retrieval

### PDF Processing Engine (`Utils/pdf_processor.py`)

This module provides a powerful PDF processing pipeline designed to convert unstructured PDF files into structured, RAG-optimized JSON data. It leverages LLMs to intelligently parse, analyze, and enrich content, making complex information within PDFs easily accessible to your agent.

### Audio Processing Engine (`Utils/audio_processor.py`)

(Under development...)

### Information Retrieval Strategy

Our RAG process is inspired by Ilya Rice's award-winning strategy, using a "small-to-large" chunking technique to provide LLMs with complete, coherent context for improved accuracy.

---

## 📦 Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
# Install main dependencies
poetry install

# Install with dev tools (ipython)
poetry install --extras dev
```

**ML/Audio packages** (`funasr`, `marker-pdf`) depend on `torch`, which is expected to be provided by the container environment. Install them separately after `poetry install`:

```bash
pip install --no-deps funasr marker-pdf
```

> **Note:** `torch` is intentionally excluded from `pyproject.toml` — it should be pre-installed in the container with the appropriate CUDA version.

---

## 🧪 Usage

### 1. Preprocess Local Data (Optional)

If you have local PDF or audio files to include in the retrieval, place them in a source directory (e.g., `data/raw_files/`). Then, run the preprocessing script:

```bash
python preprocess_files.py DATA_DIR OUTPUT_DIR [--model deepseek/deepseek-chat] [--no-table]
```

This script processes PDF files and saves structured JSON output. Ensure the output directory is correctly specified under `raw_file_path` in your `retriever_config.yaml` so the RAG pipeline can find the data.

### 2. Run Report Generation

#### Document QA Agent (`subagent/document_qa.py`)

A ReAct-loop agent that answers questions against pre-processed JSON documents using `AgentDocumentReader`.

**Entry points:**

```python
from subagent.document_qa import run_document_qa, run_document_qa_async

# Synchronous
answer = run_document_qa(
    file_paths=[{"name": "Annual Report 2024", "path": "/path/to/doc.json"}],
    question="What was the net profit in Q3?",
    budget=50,  # max LLM calls
)

# Async (non-blocking, runs in asyncio.to_thread)
answer = await run_document_qa_async(file_paths=[...], question="...", budget=50)
```

**`file_paths` format:** list of `{"name": str, "path": str}` dicts. `path` must point to a `.json` file produced by `PDFDocumentPreprocessor`.

**Dual circuit breakers** (prevent infinite loops):
- `consecutive_errors >= 3`: three back-to-back LLM failures → triggers forced answer submission
- `consecutive_text_only >= 3`: three planning-only responses with no tool calls → triggers forced answer submission

**Graceful degradation:**
- `force_answer_node` returns a Chinese fallback answer on LLM failure instead of raising
- `extract_answer_node` returns `"[Unable to extract answer from documents]"` sentinel on empty content

**Resource cleanup:** `AgentDocumentReader` is always closed in a `finally` block regardless of graph success or failure.

---

#### AgentDocumentReader / Text Navigator (`Tools/text_navigator.py`)

Long-lived reader that provides page navigation, semantic search, keyword search, and bookmarks over pre-processed JSON documents.

**Key constraints:**
- `open_document(path)` only accepts `.json` files (output of `PDFDocumentPreprocessor`). Passing other file types raises `ValueError`.
- Documents are indexed with Chroma + BM25. The Chroma SQLite cache is stored in `navigator_tmp/{doc_name}/` and **reused across sessions** — no re-embedding if the page count matches.

**`get_metadata()` return shape:**

```python
{
    "name": str,        # document name
    "date": str,        # document date
    "char_count": int,  # total character count via len() — accurate for CJK text
    "has_outline": bool,
}
```

Note: `char_count` uses `len()` (character count) rather than `.split()` (word count), which is accurate for Chinese/Japanese/Korean text without whitespace delimiters.

**Bookmarks:** persist across `open_document()` calls (unlike page position, which resets to 0 on each open).

---

#### Deep Report (`report_writer.py`)

This is the main, most feature-complete report generation workflow.

```python
from langchain_core.runnables import RunnableConfig
from State.state import ReportStateInput
from report_writer import graph, DEFAULT_REPORT_STRUCTURE

# Configure the execution
config = RunnableConfig({
    "thread_id": "your-research-id", # A unique ID for your research task
    "number_of_queries": 5,
    "use_web": True,
    "use_local_db": True,
    "max_search_depth": 3,
    "report_structure": DEFAULT_REPORT_STRUCTURE,
})

# Set the report topic
topic = "An in-depth analysis of supply chain challenges and opportunities in the global semiconductor industry as of Q1 2025..."
input_data = ReportStateInput(topic=topic)

# Start the graph execution
for event in graph.stream(input_data, config, stream_mode="updates"):
    if "__interrupt__" in event:
        # The process will pause here when user feedback is needed
        print(event["__interrupt__"][0].value)
        # Enter your feedback here to continue
```

