# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-08-09

### Added
- Implemented full asynchronous architecture to resolve event loop conflicts and improve performance.
- Added `AsyncSqliteSaver` support in `report_writer.py` for async checkpoint management.
- Introduced `get_async_graph()` method in `ReportGraphBuilder` for creating async-compatible LangGraph instances.
- Added async versions of all main execution scripts with proper `AsyncSqliteSaver` integration.
- Implemented robust tool call validation in `Utils/utils.py` with automatic retry mechanism for required tool calls.

### Changed
- **BREAKING**: Converted `search_db` function in `report_writer.py` to async, now uses `await agentic_search_graph.ainvoke()`.
- **BREAKING**: Converted `filter_and_format_results` function in `agentic_search.py` to async with proper `await asyncio.gather()` for concurrent processing.
- **BREAKING**: Converted `compress_raw_content` function in `agentic_search.py` to async with parallel processing using `await asyncio.gather()` for improved performance.
- Enhanced `ReportGraphBuilder` to support both synchronous and asynchronous graph instances.
- Enhanced `call_llm` and `call_llm_async` functions in `Utils/utils.py` to use LiteLLM fallbacks parameter instead of manual try-catch for model switching.
- Improved `call_llm` and `call_llm_async` functions to automatically validate and retry tool calls when `tool_choice="required"` is set, with up to 3 retry attempts.
- Simplified error handling in `report_writer.py` by removing manual try-catch blocks for tool call failures, now relying on robust validation in `call_llm` functions.


## [0.2.1] - 2025-08-08

### Added
- Added final content refinement logic to `gather_complete_section` function in `report_writer.py` to ensure report consistency and integration.
- Introduced `content_refinement_formatter` tool in `Tools/tools.py` for formatting refined content output.
- Added `content_refinement_instructions` prompt template to both `Prompt/industry_prompt.py` and `Prompt/technical_research_prompt.py` for final content polishing.

### Changed
- Enhanced `gather_complete_section` function to perform content-only refinement based on full report context without modifying descriptions or generating new queries.
- Updated imports in `report_writer.py` to include the new content refinement formatter.

## [0.2.0] - 2025-08-03

### Added
- Implemented an iterative refinement process in `report_writer.py`. After the initial draft, each report section's description and content are refined based on the context of all other sections.
- Added new prompts to `Prompt/industry_prompt.py` specifically for refining section descriptions and content, aiming for greater coherence and detail.
- Introduced `refine_iteration` and `curr_refine_iteration` to `State/state.py` to manage the refinement loop.

### Changed
- Modified the main graph in `report_writer.py` to conditionally loop back to the research and writing phase after each refinement cycle, controlled by the `refine_iteration` config.
- The `completed_sections` state is now cleared before re-running the writing phase in a refinement iteration to prevent mixing old and new content.

## [0.1.0] - 2025-07-23

### Added
- Created a separate LangGraph for agentic web search in `agentic_search.py`.
- Introduced asynchronous processing in the web search graph to improve performance of search quality checks.

### Changed
- Refactored `report_writer.py` to delegate web search tasks to the new agentic search graph.
- Moved the `call_llm` function to `Utils/utils.py` for better code reuse and created an asynchronous version `call_llm_async`.