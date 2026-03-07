"""Langfuse observability utilities for LangGraph nodes.

Langfuse 3.x API:
  - observe() imported from langfuse (not langfuse.decorators)
  - CallbackHandler() with no args auto-inherits current @observe trace context
  - get_langfuse_callback() returns None if not inside a trace (avoids spurious root traces)

Usage in graph builders:
    from Utils.langfuse_tracing import langfuse_node
    builder.add_node("my_node", langfuse_node(my_node_fn))

Entry points (top-level traces):
    from langfuse import observe

    @observe(name="agentic_search")
    async def _run():
        ...
"""
import functools
import inspect
import sys as _sys

# Conditional imports: only bind these names if they are not already present in the module
# namespace. This ensures that test patches applied via unittest.mock.patch(...) before an
# importlib.reload() call are preserved — reload() re-executes the module body in the SAME
# namespace without clearing it first, so a pre-existing attribute (set by patch) survives
# as long as the module code does not unconditionally overwrite it.
if "observe" not in vars():
    from langfuse import observe  # type: ignore[assignment]  # noqa: PLC0415
if "get_client" not in vars():
    from langfuse import get_client  # type: ignore[assignment]  # noqa: PLC0415
if "CallbackHandler" not in vars():
    from langfuse.langchain import CallbackHandler  # type: ignore[assignment]  # noqa: PLC0415


def langfuse_node(fn, name: str | None = None):
    """Wrap a LangGraph node function with a Langfuse span.

    Args:
        fn: Node function (sync or async). May take (state,) or (state, config, ...).
            Also works with callable instances (e.g. ToolNode).
        name: Span name override. Defaults to fn.__name__.

    Returns:
        Wrapped function with identical call signature.
    """
    span_name = name if name is not None else fn.__name__
    # Read 'observe' from the current module namespace at call time so that test patches
    # applied to Utils.langfuse_tracing.observe are respected.
    _observe = _sys.modules[__name__].observe

    if inspect.iscoroutinefunction(fn):
        @_observe(name=span_name)
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            return await fn(*args, **kwargs)
        return async_wrapper
    else:
        @_observe(name=span_name)
        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return sync_wrapper


def get_langfuse_callback():
    """Return a LangChain CallbackHandler linked to the current Langfuse trace.

    Returns None when called outside a @observe context to avoid creating
    spurious root traces for standalone LLM calls.

    Usage in call_llm / call_llm_async:
        handler = get_langfuse_callback()
        callbacks = [handler] if handler else []
        model.invoke(prompt, config={"callbacks": callbacks})
    """
    _mod = _sys.modules[__name__]
    client = _mod.get_client()
    if not client.get_current_trace_id():
        return None
    return _mod.CallbackHandler()
