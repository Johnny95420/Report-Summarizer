"""Tests for Utils/langfuse_tracing.py — langfuse_node wrapper and get_langfuse_callback."""
import asyncio
from unittest.mock import MagicMock, patch

import pytest


def test_langfuse_node_wraps_sync_function():
    """langfuse_node wraps a sync function, preserving __name__ and behavior."""
    with patch("Utils.langfuse_tracing.observe", lambda **kw: (lambda fn: fn)):
        from importlib import reload
        import Utils.langfuse_tracing
        reload(Utils.langfuse_tracing)
        from Utils.langfuse_tracing import langfuse_node

        def my_node(state):
            return {"result": state["x"] + 1}

        wrapped = langfuse_node(my_node)
        assert wrapped({"x": 5}) == {"result": 6}
        assert wrapped.__name__ == "my_node"


def test_langfuse_node_wraps_async_function():
    """langfuse_node wraps an async function, preserving __name__ and behavior."""
    with patch("Utils.langfuse_tracing.observe", lambda **kw: (lambda fn: fn)):
        from importlib import reload
        import Utils.langfuse_tracing
        reload(Utils.langfuse_tracing)
        from Utils.langfuse_tracing import langfuse_node

        async def my_async_node(state):
            return {"result": state["x"] * 2}

        wrapped = langfuse_node(my_async_node)
        result = asyncio.run(wrapped({"x": 3}))
        assert result == {"result": 6}
        assert wrapped.__name__ == "my_async_node"


def test_langfuse_node_passes_extra_args():
    """langfuse_node passes extra args (e.g. config: RunnableConfig) through unchanged."""
    with patch("Utils.langfuse_tracing.observe", lambda **kw: (lambda fn: fn)):
        from importlib import reload
        import Utils.langfuse_tracing
        reload(Utils.langfuse_tracing)
        from Utils.langfuse_tracing import langfuse_node

        def node_with_config(state, config):
            return {"val": config["key"]}

        wrapped = langfuse_node(node_with_config)
        assert wrapped({"x": 1}, {"key": "ok"}) == {"val": "ok"}


def test_langfuse_node_uses_function_name_as_span_name():
    """langfuse_node calls observe(name=fn.__name__)."""
    observed_kwargs = []

    def fake_observe(**kwargs):
        observed_kwargs.append(kwargs)
        return lambda fn: fn

    with patch("Utils.langfuse_tracing.observe", fake_observe):
        from importlib import reload
        import Utils.langfuse_tracing
        reload(Utils.langfuse_tracing)
        from Utils.langfuse_tracing import langfuse_node

        def perform_web_search(state):
            return {}

        langfuse_node(perform_web_search)
        assert any(kw.get("name") == "perform_web_search" for kw in observed_kwargs)


def test_langfuse_node_accepts_name_override():
    """langfuse_node uses explicit name= override when provided."""
    observed_kwargs = []

    def fake_observe(**kwargs):
        observed_kwargs.append(kwargs)
        return lambda fn: fn

    with patch("Utils.langfuse_tracing.observe", fake_observe):
        from importlib import reload
        import Utils.langfuse_tracing
        reload(Utils.langfuse_tracing)
        from Utils.langfuse_tracing import langfuse_node

        def tool_node_fn(state):
            return {}

        langfuse_node(tool_node_fn, name="tools")
        assert any(kw.get("name") == "tools" for kw in observed_kwargs)


def test_get_langfuse_callback_returns_handler_inside_trace():
    """get_langfuse_callback returns a CallbackHandler when inside a trace."""
    mock_client = MagicMock()
    mock_client.get_current_trace_id.return_value = "trace-123"
    mock_handler = MagicMock()

    with patch("Utils.langfuse_tracing.get_client", return_value=mock_client), \
         patch("Utils.langfuse_tracing.CallbackHandler", return_value=mock_handler):
        from importlib import reload
        import Utils.langfuse_tracing
        reload(Utils.langfuse_tracing)
        from Utils.langfuse_tracing import get_langfuse_callback

        result = get_langfuse_callback()
        assert result is mock_handler


def test_get_langfuse_callback_returns_none_outside_trace():
    """get_langfuse_callback returns None when not inside a trace (avoids spurious traces)."""
    mock_client = MagicMock()
    mock_client.get_current_trace_id.return_value = None

    with patch("Utils.langfuse_tracing.get_client", return_value=mock_client):
        from importlib import reload
        import Utils.langfuse_tracing
        reload(Utils.langfuse_tracing)
        from Utils.langfuse_tracing import get_langfuse_callback

        result = get_langfuse_callback()
        assert result is None
