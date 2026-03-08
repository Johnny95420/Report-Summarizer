"""Langfuse observability utilities for LangGraph nodes.

Provides a small, unified API so callers never touch OTel or Langfuse internals
directly. All context-propagation edge cases (threads, asyncio.gather, subgraphs)
are handled here.

Public API
----------
langfuse_node(fn, ...)     — wrap a LangGraph node (sync or async)
traced(name)               — context manager span (sync `with` only)
traced_thread(fn, *a, **kw)— run sync fn in thread WITH parent span propagated
observe                    — re-exported from langfuse for @observe on entry points
get_langfuse_callback()    — LangChain CallbackHandler linked to current span

Usage in graph builders:
    from Utils.langfuse_tracing import langfuse_node
    builder.add_node("my_node", langfuse_node(my_node_fn))

    # Disambiguate parallel sections:
    builder.add_node("orchestration", langfuse_node(orchestration, state_key="section.name"))

Entry points (top-level traces):
    from Utils.langfuse_tracing import observe

    @observe(name="report_writer")
    async def run():
        ...

Thread boundary (sync graph in async node):
    from Utils.langfuse_tracing import traced_thread
    result = await traced_thread(graph.invoke, initial_state, config)

Manual span (e.g. inside asyncio.gather closures):
    from Utils.langfuse_tracing import traced
    async def _do_web():
        with traced("agentic_web_search"):
            return await agentic_search_graph.ainvoke(...)
"""
import asyncio
import functools
import inspect
import sys as _sys

# ---------------------------------------------------------------------------
# Conditional imports — preserve test patches across reload()
# ---------------------------------------------------------------------------
if "observe" not in vars():
    from langfuse import observe  # type: ignore[assignment]  # noqa: PLC0415
if "get_client" not in vars():
    from langfuse import get_client  # type: ignore[assignment]  # noqa: PLC0415
if "CallbackHandler" not in vars():
    from langfuse.langchain import CallbackHandler  # type: ignore[assignment]  # noqa: PLC0415


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _get_dynamic_span_name(span_name: str, state_key: str | None, args) -> str:
    """Build a span name, optionally appending a value from state for disambiguation.

    When state_key is provided and the first positional arg (state dict) contains it,
    the span name becomes "span_name [value]", e.g. "orchestration [半導體分析]".
    """
    if not state_key or not args:
        return span_name
    state = args[0]
    if isinstance(state, dict):
        obj = state
        for part in state_key.split("."):
            if isinstance(obj, dict):
                obj = obj.get(part)
            else:
                obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return f"{span_name} [{obj}]"
    return span_name


def _client():
    """Get the Langfuse client from module namespace (respects test patches)."""
    return _sys.modules[__name__].get_client()


# ---------------------------------------------------------------------------
# Public: langfuse_node — wrap LangGraph node functions
# ---------------------------------------------------------------------------
def langfuse_node(fn, name: str | None = None, state_key: str | None = None):
    """Wrap a LangGraph node function with a Langfuse span.

    Works for sync and async nodes. Uses start_as_current_span so child calls
    (LLM, sub-agents) automatically nest under this span.

    Args:
        fn: Node function (sync or async).
        name: Span name override. Defaults to fn.__name__.
        state_key: Dot-delimited key into state dict to append to span name
                   (e.g. "section.name" → "orchestration [半導體分析]").
    """
    span_name = name if name is not None else fn.__name__

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            dynamic_name = _get_dynamic_span_name(span_name, state_key, args)
            with _client().start_as_current_span(name=dynamic_name):
                return await fn(*args, **kwargs)
        return async_wrapper
    else:
        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            dynamic_name = _get_dynamic_span_name(span_name, state_key, args)
            with _client().start_as_current_span(name=dynamic_name):
                return fn(*args, **kwargs)
        return sync_wrapper


# ---------------------------------------------------------------------------
# Public: traced — context manager for manual spans
# ---------------------------------------------------------------------------
def traced(name: str):
    """Create a Langfuse span as a context manager (sync `with` only).

    Use this for wrapping arbitrary code blocks, e.g. inside asyncio.gather closures:

        async def _do_web():
            with traced("agentic_web_search"):
                return await agentic_search_graph.ainvoke(...)

    The returned context manager only supports `with`, not `async with`.
    For async nodes, use langfuse_node() instead.
    """
    return _client().start_as_current_span(name=name)


# ---------------------------------------------------------------------------
# Public: traced_thread — run sync fn in thread with span propagation
# ---------------------------------------------------------------------------
async def traced_thread(fn, *args, **kwargs):
    """Run a sync function in asyncio.to_thread with OTel context propagated.

    Solves the problem where asyncio.to_thread breaks Langfuse span nesting
    because the worker thread gets an empty OTel context.

    Usage:
        result = await traced_thread(graph.invoke, initial_state, config)

    Instead of:
        # BAD: spans inside graph.invoke become orphans
        result = await asyncio.to_thread(graph.invoke, initial_state, config)
    """
    from opentelemetry import context as otel_context

    parent_ctx = otel_context.get_current()

    def _run():
        token = otel_context.attach(parent_ctx)
        try:
            return fn(*args, **kwargs)
        finally:
            otel_context.detach(token)

    return await asyncio.to_thread(_run)


# ---------------------------------------------------------------------------
# Public: get_langfuse_callback — LangChain integration
# ---------------------------------------------------------------------------
def get_langfuse_callback():
    """Return a LangChain CallbackHandler linked to the current Langfuse trace.

    Returns None when called outside a @observe context to avoid creating
    spurious root traces for standalone LLM calls.
    """
    client = _client()
    if not client.get_current_trace_id():
        return None
    return _sys.modules[__name__].CallbackHandler()
