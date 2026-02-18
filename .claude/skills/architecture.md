# Architecture Deep Dive

## System Overview

Report-Summarizer is a multi-agent LangGraph orchestration system with nested subgraphs. The system uses a **Planning -> Research -> Writing -> Review -> Refinement** pipeline, with human-in-the-loop feedback gates.

## Graph Architecture

### Main Report Graph (`report_writer.py`)

The main graph is built by `ReportGraphBuilder` and has two levels: a **main graph** (report-level) and a **section subgraph** (per-section processing).

```
START
  -> generate_report_plan        # Planning phase: queries -> search -> section generation
  -> human_feedback              # interrupt() for user approval/feedback
  -> build_section_with_web_research  # Parallel section processing (Send API)
  -> route                       # Format completed sections
  -> [conditional] should_refine
     -> refine_sections          # If curr_refine_iteration < refine_iteration
        -> (re-triggers build_section_with_web_research via Send)
     -> gather_complete_section  # If refinement complete
  -> write_final_sections        # Non-research sections (intro, conclusion) via Send
  -> compile_final_report        # Assemble all sections
  -> END
```

### Section Subgraph (nested inside main graph)

```
START
  -> generate_queries     # Generate section-specific search queries
  -> search_db            # Hybrid search: local DB + agentic web search
  -> write_section        # Write content + grade quality
     -> [conditional]
        -> END            # If grade=pass or max_search_depth reached
        -> search_db      # If grade=fail, loop with follow-up queries
```

### Agentic Search Graph (`agentic_search.py`)

Standalone subgraph invoked by `search_db` in the section subgraph:

```
START
  -> get_searching_budget         # Determine iteration budget (currently hardcoded to 1)
  -> perform_web_search           # Selenium/Bing search with URL deduplication
  -> filter_and_format_results    # Async quality scoring (1-5, threshold >2)
  -> compress_raw_content         # LLM-based content compression
  -> aggregate_final_results      # Format results with source metadata
  -> check_searching_results      # Grade sufficiency
     -> [conditional]
        -> END                    # If pass or max iterations reached
        -> perform_web_search     # If fail, loop with follow-up queries
```

## State Management

### State Types (defined in `State/state.py`)

| State Type | Scope | Purpose |
|---|---|---|
| `ReportStateInput` | Graph input | `topic`, `refine_iteration` |
| `ReportStateOutput` | Graph output | `final_report` |
| `ReportState` | Main graph | Full report workflow state |
| `SectionState` | Section subgraph | Per-section research/writing state |
| `SectionOutputState` | Section output | `completed_sections` only |
| `AgenticSearchState` | Search subgraph | Web search iteration state |

### Custom Reducers

**`clearable_list_reducer`** (used by `completed_sections`):
- Normal list append via `left + right`
- Special `"__CLEAR__"` sentinel resets the list to empty
- Used during refinement loops to clear completed sections before re-research

**`operator.add`** (used by `feedback_on_report_plan`, `queries_history`):
- Standard list concatenation accumulator

### State Flow Patterns

1. **Send() API**: Used for dynamic parallel task spawning
   - `human_feedback` -> sends each research section to `build_section_with_web_research`
   - `refine_sections` -> re-sends refined sections with new queries
   - `initiate_final_section_writing` -> sends non-research sections to `write_final_sections`

2. **Command() API**: Used for conditional routing with state updates
   - `write_section` returns `Command(goto=END)` or `Command(goto="search_db")`
   - `check_searching_results` returns `Command(goto=END)` or `Command(goto="perform_web_search")`
   - `human_feedback` returns `Command(goto="generate_report_plan")` or `Command(goto=[Send(...)])`

3. **interrupt()**: Used in `human_feedback` node for human-in-the-loop
   - Returns `bool` (True = approve) or `str` (feedback text for plan revision)

## Checkpointing & Persistence

- **SQLite-based**: `SqliteSaver` for sync, `AsyncSqliteSaver` for async
- Default: `checkpoints.sqlite` in project root
- Enables: pause/resume workflows, state inspection, long-running task recovery
- Thread-safe: `check_same_thread=False`

## Builder Pattern

`ReportGraphBuilder` class:
- Lazy builds and caches both sync (`get_graph()`) and async (`get_async_graph()`) graphs
- `_build_section_graph()` creates the section subgraph (shared)
- `_build_main_graph(section_graph)` creates the main graph wrapping the section graph
- Accepts custom checkpointers for testing

`AgenticSearchGraphBuilder` class:
- Same lazy build pattern
- Module-level singleton: `agentic_search_graph = AgenticSearchGraphBuilder().get_graph()`

## Concurrency Model

- **Parallel section processing**: via `Send()` API in main graph
- **Async LLM calls**: `asyncio.gather()` with `Semaphore(2)` for rate limiting
- **Thread pools**: for blocking I/O in Selenium driver operations
- **Async graph execution**: `graph.astream()` / `graph.ainvoke()` for non-blocking workflows

## Key Architectural Decisions

1. **Nested subgraphs** instead of flat graph: section processing is isolated, enabling parallel execution and cleaner state management
2. **Tool-based structured output**: LLM outputs are constrained via `@tool` decorators with `tool_choice="required"`, ensuring parseable responses
3. **Fallback model pattern**: every LLM call has a primary + backup model with automatic failover via `with_fallbacks()`
4. **Hybrid retrieval**: vector (ChromaDB) + keyword (BM25) ensemble ensures both semantic and exact-match coverage
5. **Content expansion**: small-to-large chunking strategy with `track_expanded_context()` preserves surrounding context for better LLM comprehension
