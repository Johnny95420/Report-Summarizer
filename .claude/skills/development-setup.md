# Development Setup

## Prerequisites

- Python 3.12 (required: `>=3.12,<3.13`)
- Poetry (dependency management)
- Docker + NVIDIA Container Toolkit (for GPU workloads)
- Node.js 18 LTS (for md-to-pdf export)

## Installation

### Poetry (Main Dependencies)

```bash
# Install main dependencies (no virtualenv in container)
poetry install

# Install with dev tools (ipython, pre-commit, ruff)
poetry install --extras dev
```

### ML Packages (via pip, NOT poetry)

ML packages depend on `torch`, which must come from the container's NVIDIA PyTorch build. Installing via poetry would override NGC's custom build.

```bash
# Explicit sub-deps first
pip install --no-cache-dir \
    "transformers>=4.41.0,<5.0.0" \
    "huggingface-hub>=0.33.4,<1.0.0"

# ML packages with --no-deps to preserve NGC PyTorch
pip install --no-cache-dir --no-deps \
    "sentence-transformers" \
    "marker-pdf"

# Additional dependencies
pip install --no-cache-dir \
    "surya-ocr" "pdftext" "ftfy" "filetype" \
    "pypdfium2==4.30.0" "markdownify" "google-genai"
```

### Pre-commit Hooks

```bash
pre-commit install
```

## Docker Environment

### Base Image

`nvcr.io/nvidia/pytorch:26.01-py3` - NVIDIA PyTorch with CUDA support.

### Building & Running

```bash
# Build
docker-compose build

# Run (interactive, GPU-enabled)
docker-compose up -d
docker exec -it agent-dev bash
```

### Container Details (`docker-compose.yml`)

- **Runtime**: nvidia (full GPU access)
- **Shared memory**: 8GB (`shm_size: "8g"`)
- **Network**: host mode (direct access to host ports)
- **Volumes**:
  - Source code mounted to `/root/pdf_parser`
  - SSH keys mounted read-only
- **Environment**: `.env` file loaded, `NVIDIA_VISIBLE_DEVICES=all`

### Installed in Container

- System: git, curl, wget, vim, libpango (for WeasyPrint)
- Node.js 18 LTS + md-to-pdf
- Claude Code CLI
- Poetry (no virtualenv)
- All ML packages (see pip section above)

## Required Configuration Files

You must create these files before running the application. They are **not** tracked in git.

### 1. `.env`

```env
OPENAI_API_KEY="your_openai_api_key"
GEMINI_API_KEY="your_gemini_api_key"
DEEPSEEK_API_KEY="your_deepseek_api_key"
REPLICATE_API_KEY="your_replicate_api_key"
TAVILY_API_KEY="your_tavily_api_key"
SEARCH_HOST="localhost"
SEARCH_PORT="8000"
```

### 2. `report_config.yaml`

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
  Use this structure and Traditional Chinese to create a report...
```

### 3. `retriever_config.yaml`

```yaml
raw_file_path:
  - "/path/to/preprocessed_data/"
split_chunk_size: 1500
split_chunk_overlap: 250
embedding_model: "BAAI/bge-m3"
top_k: 5
hybrid_weight: [0.4, 0.6]
```

## Running the Application

### Preprocessing Local Data

```bash
python preprocess_files.py
```

Processes PDFs/audio into RAG-friendly JSON. Output directory must match `raw_file_path` in `retriever_config.yaml`.

### Starting the Selenium Search Service

The Selenium scraping API (`Utils/selenium_searching_api.py`) runs as a FastAPI service:

```bash
# Runs on SEARCH_HOST:SEARCH_PORT (default: localhost:8000)
uvicorn Utils.selenium_searching_api:app --host 0.0.0.0 --port 8000
```

### Running Report Generation

```python
from langchain_core.runnables import RunnableConfig
from State.state import ReportStateInput
from report_writer import ReportGraphBuilder, DEFAULT_REPORT_STRUCTURE

builder = ReportGraphBuilder()
graph = builder.get_graph()

config = RunnableConfig({
    "configurable": {
        "thread_id": "your-research-id",
        "number_of_queries": 5,
        "use_web": True,
        "use_local_db": True,
        "max_search_depth": 3,
        "report_structure": DEFAULT_REPORT_STRUCTURE,
    }
})

input_data = ReportStateInput(topic="Your research topic...", refine_iteration=1)

for event in graph.stream(input_data, config, stream_mode="updates"):
    if "__interrupt__" in event:
        # Human feedback gate - approve with True or provide feedback string
        print(event["__interrupt__"][0].value)
```

### Async Execution

```python
async with AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
    builder = ReportGraphBuilder(async_checkpointer=checkpointer)
    graph = builder.get_async_graph()
    result = await graph.ainvoke(input_data, config)
```

## Linting & Formatting

```bash
# Run ruff linter with auto-fix
ruff check --fix .

# Run ruff formatter
ruff format .

# Run all pre-commit hooks manually
pre-commit run --all-files
```
