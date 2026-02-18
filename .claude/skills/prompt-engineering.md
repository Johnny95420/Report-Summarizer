# Prompt Engineering Guide

## Prompt File Locations

| File | Purpose | Lines |
|---|---|---|
| `Prompt/industry_prompt.py` | Report generation prompts (planning, writing, grading, refinement) | ~567 |
| `Prompt/agentic_search_prompt.py` | Search workflow prompts (budget, filtering, compression, grading) | ~221 |

## Prompt Structure Conventions

### XML-Style Tags

All prompts use XML-style tags for structured sections. Tags **must be properly closed** (enforced by `test_prompts.py`):

```python
"""
<Task>
Your goal is to ...
</Task>

<Context>
{context}
</Context>

<Evaluation Criteria>
...
</Evaluation Criteria>
"""
```

Common tag names used across prompts:
- `<Task>` / `</Task>` - Primary instructions
- `<Context>` / `</Context>` - Source material or background
- `<Topic>` / `</Topic>` - Report/section topic
- `<Report organization>` / `</Report organization>` - Structure template
- `<Feedback>` / `</Feedback>` - Human feedback
- `<Quality Checks>` / `</Quality Checks>` - Validation criteria
- `<Current Time>` / `</Current Time>` - Injected timestamp
- `<Section topic>` / `</Section topic>` - Section-specific topic
- `<Source material>` / `</Source material>` - Retrieved sources
- `<Query>` / `</Query>` - Search query context
- `<Document>` / `</Document>` - Document content for evaluation
- `<Language Rules>` / `</Language Rules>` - Language selection rules
- `<Query Format>` / `</Query Format>` - Search query format rules

### Current Time Injection

Every prompt in `industry_prompt.py` appends a dynamic timestamp:

```python
import datetime
time = datetime.datetime.now()
curr_date = datetime.datetime.strftime(time, format="%Y/%m/%d")

prompt_template = (
    """... prompt body ..."""
    + f"<Current Time> {curr_date} </Current Time>"
)
```

This is evaluated at **module import time**, not per-call. The date is in `YYYY/MM/DD` format.

### Format Placeholders

Prompts use Python `.format()` with named placeholders:

```python
system_instructions = section_writer_instructions.format(
    section_title=section.name,
    section_topic=section.description,
    context=source_str,
    section_content=section.content or "",
    follow_up_queries=formatted_queries,
)
```

All format calls are verified by tests to include all required keywords.

## Prompt Templates Reference

### Report Planning Phase

**`report_planner_query_writer_instructions`**
- Purpose: Generate search queries for planning phase
- Placeholders: `{topic}`, `{report_organization}`, `{number_of_queries}`, `{feedback}`
- Output: List of search queries via `queries_formatter` tool
- Language: Traditional Chinese

**`report_planner_instructions`**
- Purpose: Generate report section plan (name, description, research flag)
- Placeholders: `{topic}`, `{report_organization}`, `{context}`, `{feedback}`
- Output: List of sections via `section_formatter` tool
- Key requirement: descriptions must be 150-300 words with full background context

### Section Research Phase

**`query_writer_instructions`**
- Purpose: Generate section-specific search queries
- Placeholders: `{topic}`, `{number_of_queries}`
- Output: Keyword queries via `queries_formatter` tool
- Format: `[Entity] [Concept] [Time?]` (3-8 words, max 12)
- Language rules: Traditional Chinese for Taiwan topics, English for global topics

**`section_writer_instructions`**
- Purpose: Write a report section from sources
- Placeholders: `{section_title}`, `{section_topic}`, `{context}`, `{section_content}`, `{follow_up_queries}`
- Output: Plain text (no tool binding)
- Word limit: 100-1000 (enforced by prompt, excludes title/sources/tables)
- Key rules:
  - Start with bold key point
  - Use `##` for section title only
  - Inline citations with exact source title matching
  - End with `### Sources` section
  - Language: Traditional Chinese
  - STRICT source integrity: zero tolerance for fabricated sources

**`section_grader_instructions`**
- Purpose: Grade section quality, generate follow-up queries
- Placeholders: `{section_topic}`, `{section}`, `{queries_history}`
- Output: grade (pass/fail) + follow_up_queries via `feedback_formatter` tool
- Evaluation dimensions:
  1. Technical accuracy
  2. Financial correctness
  3. Investment analysis depth
  4. Quantitative data support
  5. Source citation integrity (CRITICAL - fabricated sources = automatic fail)
- Max 3 follow-up queries, prioritized: targeted > drill-down > exploratory
- Must check queries history to avoid semantic duplicates

### Refinement Phase

**`refine_section_instructions`**
- Purpose: Refine section using full report context, generate new queries
- Placeholders: `{section_name}`, `{section_description}`, `{section_content}`, `{full_context}`, `{number_of_queries}`
- Output: refined_description + refined_content + new_queries via `refine_section_formatter` tool
- Anti-narrowing principles: preserve all analytical dimensions, deepen don't replace
- Description output: additions only (not full rewrite)
- Cross-section integrity: remove misplaced content, use cross-references

**`content_refinement_instructions`**
- Purpose: Final polish for publication readiness
- Placeholders: `{section_name}`, `{section_content}`, `{full_context}`
- Output: refined_content via `content_refinement_formatter` tool
- Final stage: zero tolerance for hallucination, no new facts allowed

**`final_section_writer_instructions`**
- Purpose: Write non-research sections (intro, conclusion) from completed report
- Placeholders: `{section_title}`, `{section_topic}`, `{context}`
- Output: Plain text
- Word limit: 200-1000
- Introduction: `#` for title, no structural elements, no sources
- Conclusion: `##` for title, may include comparison tables

### Agentic Search Prompts

**`iteration_budget_instruction`**
- Purpose: Determine search iteration budget (1-3)
- Placeholder: `{query_list}`
- Output: budget integer via `searching_budget_formatter` tool
- Note: currently bypassed in code (hardcoded to 1)

**`query_rewriter_instruction`**
- Purpose: Rewrite queries for keyword-based search optimization
- Placeholder: `{queries_to_refine}`
- Output: refined queries via `queries_formatter` tool

**`results_filter_instruction`**
- Purpose: Score search result relevance (1-5)
- Placeholders: `{query}`, `{document}`
- Output: score via `quality_formatter` tool

**`results_compress_instruction`**
- Purpose: Compress raw web content while preserving key details
- Placeholders: `{query}`, `{document}`
- Output: summary_content via `summary_formatter` tool
- Key rules: preserve source metadata exactly, use thematic grouping, no tables (lists only)

**`searching_results_grader`**
- Purpose: Determine if search results are sufficient
- Placeholders: `{query}`, `{context}`
- Output: grade + follow_up_queries via `searching_grader_formatter` tool

## Rules for Modifying Prompts

### Must-Follow Rules

1. **Length units**: Always use "words" not "tokens" (enforced by tests)
2. **XML tags**: Always close XML-style tags properly (enforced by tests)
3. **Format placeholders**: If you add a new placeholder, update ALL `.format()` call sites
4. **Language consistency**: Maintain Traditional Chinese for financial/Taiwan topics, English for global
5. **Source integrity rules**: Never weaken the strict source citation rules
6. **Current Time**: Keep the `curr_date` injection at the end of `industry_prompt.py` templates

### Import Rules for `report_writer.py`

All 8 prompt variables must be **explicitly imported** (no wildcard `*`). This is enforced by `TestExplicitPromptImports` in `test_report_writer.py`:

```python
from Prompt.industry_prompt import (
    content_refinement_instructions,
    final_section_writer_instructions,
    query_writer_instructions,
    refine_section_instructions,
    report_planner_instructions,
    report_planner_query_writer_instructions,
    section_grader_instructions,
    section_writer_instructions,
)
```

`agentic_search.py` uses wildcard import `from Prompt.agentic_search_prompt import *` (acceptable for that module).

### Domain Context

The prompts are tuned for **financial/investment analysis** targeting:
- Taiwan stock market analysis (Traditional Chinese)
- US macro economics (English)
- Futures trading analysis
- Industry research (semiconductor, tech sectors)

The persona is consistently framed as a **senior institutional research analyst** (J.P. Morgan Asset Management standard).
