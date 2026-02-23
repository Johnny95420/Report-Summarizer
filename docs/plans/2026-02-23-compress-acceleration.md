# compress_raw_content Smart Pass-Through Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Skip LLM compression for results whose `raw_content` is < 5000 chars; compress only for ≥ 5000 chars.

**Architecture:** `compress_raw_content` already iterates over `filtered_web_results`. We add a size check before scheduling the async LLM task: short results are appended directly to `final_results`, long results go through the existing LLM path. Model selection simplifies to always `MODEL_NAME` / `BACKUP_MODEL_NAME`.

**Tech Stack:** Python `asyncio`, LangGraph, LiteLLM via `call_llm_async`

---

### Task 1: Write failing tests for pass-through behavior

**Files:**
- Modify: `tests/test_agentic_search.py`

**Step 1: Add two tests — one for short content, one for long content**

Find the test file and add at the end:

```python
@pytest.mark.asyncio
async def test_compress_passthrough_short_content(mock_call_llm_async):
    """Results with raw_content < 5000 chars are passed through without any LLM call."""
    short_result = {
        "title": "Short Article",
        "content": "brief",
        "url": "https://example.com/short",
        "raw_content": "x" * 100,  # well under 5000
    }
    state = {
        "queries": ["test query"],
        "followed_up_queries": [],
        "filtered_web_results": [{"results": [short_result]}],
    }
    result = await compress_raw_content(state)
    compressed = result["compressed_web_results"]
    # raw_content must be unchanged
    assert compressed[0]["results"][0]["raw_content"] == short_result["raw_content"]
    # No LLM call should have been made
    mock_call_llm_async.assert_not_called()


@pytest.mark.asyncio
async def test_compress_llm_called_for_long_content(mock_call_llm_async):
    """Results with raw_content >= 5000 chars trigger the LLM compression path."""
    long_result = {
        "title": "Long Article",
        "content": "detailed",
        "url": "https://example.com/long",
        "raw_content": "y" * 6000,  # over 5000
    }
    state = {
        "queries": ["test query"],
        "followed_up_queries": [],
        "filtered_web_results": [{"results": [long_result]}],
    }
    result = await compress_raw_content(state)
    # LLM must have been called exactly once
    mock_call_llm_async.assert_called_once()
```

You need to check what `mock_call_llm_async` fixture is already defined as in `conftest.py`.
Run `grep -n "mock_call_llm_async\|call_llm_async" tests/conftest.py` to confirm.
If it's not there, check `tests/test_agentic_search.py` for an existing fixture.

**Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_agentic_search.py::test_compress_passthrough_short_content \
       tests/test_agentic_search.py::test_compress_llm_called_for_long_content \
       -v
```

Expected: FAIL (current code always calls LLM regardless of content size)

---

### Task 2: Implement the pass-through logic

**Files:**
- Modify: `subagent/agentic_search.py` lines ~241–321

**Step 1: Replace `compress_content_with_metadata` and the semaphore setup**

Current code in `compress_raw_content` (lines 247–285):

```python
# Create semaphore to limit concurrent compression operations
semaphore = asyncio.Semaphore(2)

async def compress_content_with_metadata(query_idx: int, result_idx: int, query: str, result: dict):
    """Compress content with semaphore control and metadata preservation."""
    async with semaphore:
        try:
            document = (
                f"Title:{result['title']},Brief Content:{result['content']},Full Content:{result['raw_content']}"
            )
            system_instruction = results_compress_instruction.format(query=query, document=document)

            # Select appropriate model based on raw_content token length
            model_name, backup_model_name = select_model_based_on_tokens(result["raw_content"])

            compressed_result = await call_llm_async(
                model_name,
                backup_model_name,
                ...
            )

            return query_idx, result_idx, result, compressed_result

        except Exception as e:
            logger.error(...)
            return query_idx, result_idx, result, None
```

Replace with:

```python
# Increase semaphore: fewer LLM calls after pass-through means higher concurrency is safe
semaphore = asyncio.Semaphore(4)

_COMPRESS_CHAR_THRESHOLD = 5000

async def compress_content_with_metadata(query_idx: int, result_idx: int, query: str, result: dict):
    """Compress content with semaphore control and metadata preservation.

    Short content (< _COMPRESS_CHAR_THRESHOLD chars) is passed through unchanged to avoid
    unnecessary LLM calls. Only long content goes through LLM compression.
    """
    # Pass-through: short content needs no compression
    if len(result["raw_content"]) < _COMPRESS_CHAR_THRESHOLD:
        return query_idx, result_idx, result, "passthrough"

    async with semaphore:
        try:
            document = (
                f"Title:{result['title']},Brief Content:{result['content']},Full Content:{result['raw_content']}"
            )
            system_instruction = results_compress_instruction.format(query=query, document=document)

            compressed_result = await call_llm_async(
                MODEL_NAME,
                BACKUP_MODEL_NAME,
                prompt=[SystemMessage(content=system_instruction)]
                + [
                    HumanMessage(
                        content="Please help me to summary every piece of document directly, indirectly, potentially, or partially related to the query."
                    )
                ],
                tool=[summary_formatter],
                tool_choice="required",
            )

            return query_idx, result_idx, result, compressed_result

        except Exception as e:
            logger.error(f"Content compression failed for result {result.get('url', 'unknown')}: {e}")
            return query_idx, result_idx, result, None
```

**Step 2: Update the result-processing loop to handle the `"passthrough"` sentinel**

Find the loop starting at line ~294 and update the `if compressed_result is not None:` branch:

```python
    for compression_result in compression_results:
        if isinstance(compression_result, Exception):
            logger.error(f"Compression exception: {compression_result}")
            continue

        query_idx, result_idx, original_result, compressed_result = compression_result

        if compressed_result == "passthrough":
            # Short content: use original result unchanged
            final_results[query_idx]["results"].append(original_result)
            continue

        if compressed_result is not None:
            # Process successful LLM compression
            try:
                summary_content = ""
                for tool_call in compressed_result.tool_calls:
                    summary_content += tool_call["args"]["summary_content"] + "====" + "\n\n"
            except (IndexError, KeyError, TypeError) as e:
                logger.error("Failed to parse compression result: %s", e)
                final_results[query_idx]["results"].append(original_result)
                continue

            new_result = copy.deepcopy(original_result)
            new_result["raw_content"] = summary_content
            final_results[query_idx]["results"].append(new_result)
            successful_compressions += 1
        else:
            # LLM call failed: fall back to original content
            final_results[query_idx]["results"].append(original_result)
```

**Step 3: Run the two new tests**

```bash
pytest tests/test_agentic_search.py::test_compress_passthrough_short_content \
       tests/test_agentic_search.py::test_compress_llm_called_for_long_content \
       -v
```

Expected: both PASS

**Step 4: Run full test suite**

```bash
pytest tests/ -x -v
```

Expected: all tests pass (same count as before)

**Step 5: Commit**

```bash
git add subagent/agentic_search.py tests/test_agentic_search.py
git commit -m "perf: skip LLM compression for raw_content < 5000 chars

- Pass through results with raw_content < _COMPRESS_CHAR_THRESHOLD (5000 chars) unchanged
- Remove select_model_based_on_tokens from compress path; always use MODEL_NAME
- Increase compression semaphore from 2 → 4 (fewer LLM calls, higher safe concurrency)
- Expected: compress stage ~15-20s vs previous ~66s

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Manual smoke test (optional but recommended)

**Step 1: Run main block with 1 iteration and verify timing**

```bash
cd /root/pdf_parser
python subagent/agentic_search.py "" 1
```

Check the printed timing table. `compress_raw_content` should be noticeably faster than the
previous ~66s baseline. If most results are < 5000 chars, expect near-zero time for that node.

**Step 2: Run 3 iterations and verify Sources accumulation is intact**

```bash
python subagent/agentic_search.py "" 3
```

Verify that the final `[answer]` section contains multiple Sources entries (≥ 5 distinct URLs)
and the answer text cites `[1][2]...` inline. This confirms the pass-through results reach
`aggregate_final_results` and `synthesize_answer` correctly.
