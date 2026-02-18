# Testing Guide

## Running Tests

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_report_writer.py

# Run a specific test class
pytest tests/test_report_writer.py::TestSearchRelevanceDoc

# Run with verbose output
pytest tests/ -v
```

## Test Infrastructure

### Fixtures (`tests/conftest.py`)

Three `autouse=True` fixtures mock heavy dependencies so tests run without GPU, config files, API keys, or running services:

1. **`_mock_omegaconf`**: Patches `OmegaConf.load()` to return `FAKE_CONFIG` dict containing all expected model name keys + `raw_file_path: None` (skips data loading in retriever)
2. **`_mock_dotenv`**: Replaces `load_dotenv()` with a no-op
3. **`_mock_env_vars`**: Sets dummy `TAVILY_API_KEY` and `OPENAI_API_KEY` env vars

### Shared AST Helper

`find_function(tree, name)` - walks an AST module to find a function/async function definition by name at any depth. Used for code inspection tests.

### Project Root

`ROOT = Path(__file__).resolve().parent.parent` - resolves to `/home/user/Report-Summarizer/`

## Testing Patterns

### 1. AST-Based Code Inspection

The project uses `ast.parse()` + `ast.walk()` to verify implementation details without executing the code. This avoids the need for heavy mocking of LLM/search dependencies.

**Example: Verify deepcopy usage in `search_relevance_doc`**
```python
def test_uses_deepcopy_and_appends_expanded_copy(self):
    source = (ROOT / "report_writer.py").read_text()
    tree = ast.parse(source)
    func = find_function(tree, "search_relevance_doc")
    src = ast.unparse(func)
    assert "deepcopy" in src
    assert "info.append(return_res)" in src
```

**Example: Verify all `.format()` calls include a keyword**
```python
def test_all_format_calls_include_follow_up_queries(self):
    source = (ROOT / "report_writer.py").read_text()
    tree = ast.parse(source)
    func = find_function(tree, "_prepare_source_for_writing")
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "format":
                keywords = {kw.arg for kw in node.keywords}
                assert "follow_up_queries" in keywords
```

### 2. Source Text Inspection

Simpler checks that read source files as plain text:

```python
def test_logger_level_is_info(self):
    source = (ROOT / "report_writer.py").read_text()
    assert "logger.setLevel(logging.INFO)" in source

def test_no_wildcard_import_from_prompt(self):
    source = (ROOT / "report_writer.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and "Prompt" in node.module:
                for alias in node.names:
                    assert alias.name != "*"
```

### 3. Prompt Validation Tests

Tests in `test_prompts.py` validate prompt template correctness:

- **XML tag closure**: Ensures all XML-style tags in prompts are properly closed
- **Length unit enforcement**: Verifies prompts use "words" not "tokens" for length specifications (regex: `\b\d+[\s-]*tokens?\b`)

### 4. Integration Tests (`test_run_industry_report.py`)

Full end-to-end test of the LangGraph workflow with all LLM/search calls mocked:

**Setup pattern**:
```python
patches = {
    "report_writer.call_llm": _mock_call_llm,
    "report_writer.call_llm_async": _mock_call_llm_async,
    "report_writer.get_num_tokens": lambda *a, **kw: 100,
    "report_writer.selenium_api_search": lambda *a, **kw: [],
    "report_writer.web_search_deduplicate_and_format_sources": lambda *a, **kw: "mock web results",
    "report_writer.agentic_search_graph": MagicMock(
        ainvoke=AsyncMock(return_value={"source_str": "mock agentic search results"})
    ),
}
```

**Mock LLM responses** are tool-name-aware:
- `queries_formatter` -> returns mock queries
- `section_formatter` -> returns mock sections (one research, one non-research)
- `feedback_formatter` -> returns grade="pass"
- `content_refinement_formatter` -> returns refined content
- Plain calls (no tool) -> returns "Mock generated section content."

**Flow verification**:
1. Stream until `__interrupt__` (human feedback gate)
2. Resume with `Command(resume=True)` to approve
3. Assert `compile_final_report` produces non-empty `final_report`

Uses `AsyncSqliteSaver.from_conn_string(":memory:")` for in-memory checkpointing.

### 5. Configuration Tests (`test_project_config.py`)

- Verify `langchain-classic` is declared in `pyproject.toml`
- Verify all config-loading modules use `Path(__file__).parent` for robust path resolution

## Writing New Tests

### Guidelines

1. **No GPU/API keys/config files required**: Tests must run in CI without external dependencies
2. **Prefer AST inspection** over execution for verifying implementation patterns
3. **Use `conftest.py` fixtures** - they auto-mock OmegaConf, dotenv, and env vars
4. **Mock heavy dependencies** at the module level when testing graph execution
5. **Per-file ignores**: `tests/*` ignores `RUF012` (class-level lists) and `SIM102` (nested ifs)

### Test File Naming

- `test_<module_name>.py` for unit tests of a specific module
- `test_run_<workflow>.py` for integration tests of a workflow

### Adding a New Module Test

```python
"""Tests for <Module>/module_name.py"""
import ast
from .conftest import ROOT, find_function

class TestMyFunction:
    def test_some_implementation_detail(self):
        source = (ROOT / "Module" / "module_name.py").read_text()
        tree = ast.parse(source)
        func = find_function(tree, "my_function")
        assert func is not None
        src = ast.unparse(func)
        # Assert implementation details
```
