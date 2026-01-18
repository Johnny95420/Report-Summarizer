# Prompt Optimization Design Document

**Date**: 2025-01-18
**Version**: 1.0
**Status**: Approved

## Overview

This document outlines the optimization strategy for the research report generation prompts, specifically `Prompt/industry_prompt.py` and `Prompt/agentic_search_prompt.py`.

## Problems Identified

1. **Query descriptions are overly verbose** - Generated queries are complete sentences rather than effective search keywords
2. **Prompts are too long and difficult to maintain** - Instructions are scattered, terminology is inconsistent, and interactions between sections are hard to trace
3. **Refine descriptions become overly narrow** - The `refine_section_instructions` reduces multi-dimensional analysis to single focus areas
4. **Lack of financial domain specialization** - Prompts are too general for stock/macro/futures analysis

## Design Principles

1. **YAGNI** - Remove over-descriptive instructions, keep only what materially affects behavior
2. **Single Responsibility** - Each instruction block does one thing, reducing cross-interference
3. **Preserve All Critical Rules** - Do NOT delete: fact-checking rules, source integrity rules, format requirements, writing style guidelines
4. **Reorganize over Delete** - Restructure for clarity while maintaining all essential content

## Solution Components

### 1. Query Format Optimization

**Current (Problem)**:
```
❌ "台積電 2023 年第四季 N3 製程良率對毛利率的影響分析以及對 2024 年資本支出的指導意義"
```

**Optimized**:
```
✅ "台積電 N3 良率 毛利率 2023 Q4"
✅ "TSMC N3 yield gross margin Q4 2023"
```

**Rules**:
- Use KEYWORDS, not complete sentences
- Format: `[Entity] [Key Concept] [Timeframe if relevant]`
- Length: 3-8 tokens maximum
- No conjunctions (and/or/but)

**Domain-specific templates**:

| Domain | Template | Example |
|--------|----------|---------|
| Taiwan Stock | `{公司} {關鍵字} {年度/季度}` | `"台積電 資本支出 2024 指引"` |
| Industry Chain | `{產品} 供應鏈 廠商` | `"AI伺服器 供應鏈 台廠"` |
| Macro Indicators | `{指標} {國家} {時間}` | `"美國聯邦基金利率 2024 走勢"` |
| Futures/Derivatives | `{商品} 期貨 {合約} {時間}` | `"台指期 未平倉 2024/01"` |

### 2. Anti-Narrowing Principles for Refine

**Problem**: Refine reduces multi-dimensional descriptions to single focus.

**Solution**: Add `ANTI-NARROWING PRINCIPLES` block:

```python
<ANTI-NARROWING PRINCIPLES>
- **Dimension Preservation**: When refining descriptions, you MUST preserve ALL analytical dimensions from the original
  * Examples of dimensions: market background, financial metrics, competitive landscape, risk factors
  * DO NOT reduce multi-dimensional analysis to single focus

- **Deepen, Don't Replace**: Your refinement should ADD specificity to existing dimensions, not remove them
  * ❌ Bad: "Focus only on N3 yield improvement"
  * ✅ Good: "Analyze N3 yield trends (historical + current), impact on gross margins, customer adoption rates, and competitive comparison with Samsung GAA"

- **Missing Dimension Detection**: Identify dimensions that are MISSING from the original and ADD them

- **Output Format**: For refined_description, output ONLY the additions - do not repeat the original
</ANTI-NARROWING PRINCIPLES>
```

### 3. Shared Modules Structure

Extract repeated content into shared modules:

```
Prompt/
├── shared/
│   ├── source_integrity.py    # Shared source citation rules
│   ├── format_rules.py        # Shared format requirements
│   ├── tone_guidelines.py     # Shared writing style guidelines
│   ├── language_rules.py      # Shared language protocol (TC/EN)
│   └── query_format.py        # Shared query format rules
├── industry_prompt.py         # Main prompt (imports shared modules)
└── agentic_search_prompt.py   # Main prompt (imports shared modules)
```

### 4. Language Rules Module

**Shared across all query-generating instructions**:

```python
LANGUAGE_RULES = """
<Language Protocol>
Determine the query language based on the subject's geographic scope:

**Traditional Chinese**: Use when the topic is EXCLUSIVELY related to:
- Taiwan (台灣)
- Taiwan-listed companies (台股上市櫃)
- Taiwan-specific policies, regulations, or economic data

**English**: Use when the topic involves:
- United States, Europe, or international markets
- Global or comparative analysis
- Multi-country regions (APAC, ASEAN, etc.)
- Non-Taiwan Asian markets (Japan, Korea, Hong Kong, China)

**Mixed**: If analyzing Taiwan in a global context, use Traditional Chinese for Taiwan-specific queries and English for international/comparative queries.
</Language Protocol>
"""

LANGUAGE_RULES_SHORT = """
<Language Rules>
- Taiwan-only: Traditional Chinese
- Global/US/Europe/Asia: English
</Language Rules>
"""
```

### 5. Improved Prompt Structure

Add clear section boundaries:

```python
prompt_instructions = """...

═══════════════════════════════════════════════════════════
SECTION 1: TASK DEFINITION
═══════════════════════════════════════════════════════════

<Task>...</Task>

═══════════════════════════════════════════════════════════
SECTION 2: INPUT CONTEXT
═══════════════════════════════════════════════════════════

<Topic>...</Topic>
...
"""
```

Add version tracking:

```python
"""
PROMPT VERSION: 2.0
LAST UPDATED: 2025-01-18
CHANGES:
- v2.0: Added ANTI-NARROWING PRINCIPLES to refine_section_instructions
- v2.0: Extracted shared modules (language_rules, query_format, source_integrity)
- v2.0: Optimized query generation to use keyword format (3-8 tokens)
- v1.9: Added strict source integrity rules
- v1.8: Initial version
"""
```

## Implementation Plan

### Phase 1: Core Refactoring (Immediate)

| Step | File | Changes |
|------|------|---------|
| 1.1 | `shared/language_rules.py` | [NEW] Language protocol module |
| 1.2 | `shared/query_format.py` | [NEW] Query format rules (keyword, 3-8 tokens) |
| 1.3 | `shared/source_integrity.py` | [NEW] Extracted source integrity rules |
| 1.4 | `agentic_search_prompt.py` | Rewrite `query_rewriter_instruction` - keyword format |
| 1.5 | `agentic_search_prompt.py` | Update `searching_results_grader` - query format rules |
| 1.6 | `industry_prompt.py` | Update `section_grader_instructions` - use shared modules |
| 1.7 | `industry_prompt.py` | Add `ANTI-NARROWING PRINCIPLES` to `refine_section_instructions` |
| 1.8 | Both files | Add version header and section boundaries |

### Phase 2: Testing

```python
test_cases = [
    "台積電 N3 製程分析",        # Taiwan stock
    "美國 2024 年利率預測",      # Macro economics
    "台指期 未平倉分析",         # Futures/derivatives
]

acceptance_criteria = [
    "Query length < 8 tokens",
    "Refined description maintains multi-dimensional coverage",
    "Source citations 100% from provided sources",
    "Language rules applied correctly (TC/EN)",
]
```

### Phase 3: Modular Refactoring (Future)

- Create `shared/` directory structure
- Refactor main prompts to import shared modules
- Add financial domain knowledge module (deferred)

## Summary of Changes

| Problem | Solution | Impact |
|---------|----------|--------|
| Query too verbose | Keyword format, 3-8 tokens | Better search results |
| Refine narrows scope | ANTI-NARROWING PRINCIPLES | Maintains comprehensive coverage |
| Hard to maintain | Shared modules, clear sections | Easier updates |
| Terminology inconsistency | Unified terms | Reduced confusion |
| Language rules repeated | Extracted to `language_rules.py` | Single source of truth |
| Grader generates bad queries | Apply query format rules | Consistent query quality |

## Deferred Items

- Financial domain specialization (stock/macro/futures specific knowledge)
- Full modular refactoring with imports
- Automated testing framework
