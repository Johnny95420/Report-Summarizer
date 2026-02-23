# Design: compress_raw_content Smart Pass-Through

**Date:** 2026-02-23
**Branch:** feature/question-centric-agent-flow
**Status:** Approved

## Problem

`compress_raw_content` in `subagent/agentic_search.py` calls the LLM for every filtered result,
regardless of content size. Profiling shows it consumes ~66s (~17%) of total pipeline time.

## Solution: Size-Based Pass-Through (Approach B)

### Core Logic

```
for each result in filtered_web_results:
    if len(raw_content) < 5000:
        → pass through unchanged (no LLM call)
    else:
        → LLM compress (existing behavior)
```

**Threshold:** 5000 chars — matches the `selenium_api_search` chunking threshold.
Results already chunked at ≥5000 chars will always arrive as `_partN` chunks (<5000 chars)
and pass through directly without an LLM call.

### Secondary Changes

- **Model selection:** Remove `select_model_based_on_tokens` — always use `MODEL_NAME` /
  `BACKUP_MODEL_NAME` (long documents deserve the primary model for quality compression).
- **Semaphore:** Increase from 2 → 4, since far fewer LLM calls occur after pass-through logic.

## Expected Impact

From profiling: compress ~66s total. If ~70–80% of results are <5000 chars (typical after filter),
remaining LLM calls drop proportionally → expected ~15–20s. Overall pipeline savings: ~45–50s.

## Files Changed

- `subagent/agentic_search.py`: `compress_raw_content` function
