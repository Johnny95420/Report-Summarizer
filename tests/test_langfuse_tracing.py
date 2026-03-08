"""Tests for Utils/langfuse_tracing.py — langfuse_node wrapper and get_langfuse_callback."""
import asyncio
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_client():
    """Create a mock Langfuse client with start_as_current_span returning a context manager."""
    mock_client = MagicMock()
    mock_span_ctx = MagicMock()
    mock_span_ctx.__enter__ = MagicMock(return_value=mock_span_ctx)
    mock_span_ctx.__exit__ = MagicMock(return_value=False)
    mock_client.start_as_current_span.return_value = mock_span_ctx
    return mock_client


def test_langfuse_node_wraps_sync_function():
    """langfuse_node wraps a sync function, preserving __name__ and behavior."""
    mock_client = _make_mock_client()
    with patch("Utils.langfuse_tracing.get_client", return_value=mock_client):
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
    mock_client = _make_mock_client()
    with patch("Utils.langfuse_tracing.get_client", return_value=mock_client):
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
    mock_client = _make_mock_client()
    with patch("Utils.langfuse_tracing.get_client", return_value=mock_client):
        from importlib import reload
        import Utils.langfuse_tracing
        reload(Utils.langfuse_tracing)
        from Utils.langfuse_tracing import langfuse_node

        def node_with_config(state, config):
            return {"val": config["key"]}

        wrapped = langfuse_node(node_with_config)
        assert wrapped({"x": 1}, {"key": "ok"}) == {"val": "ok"}


def test_langfuse_node_uses_function_name_as_span_name():
    """langfuse_node passes fn.__name__ to start_as_current_span."""
    mock_client = _make_mock_client()
    with patch("Utils.langfuse_tracing.get_client", return_value=mock_client):
        from importlib import reload
        import Utils.langfuse_tracing
        reload(Utils.langfuse_tracing)
        from Utils.langfuse_tracing import langfuse_node

        def perform_web_search(state):
            return {}

        wrapped = langfuse_node(perform_web_search)
        wrapped({})
        mock_client.start_as_current_span.assert_called_once_with(name="perform_web_search")


def test_langfuse_node_accepts_name_override():
    """langfuse_node uses explicit name= override when provided."""
    mock_client = _make_mock_client()
    with patch("Utils.langfuse_tracing.get_client", return_value=mock_client):
        from importlib import reload
        import Utils.langfuse_tracing
        reload(Utils.langfuse_tracing)
        from Utils.langfuse_tracing import langfuse_node

        def tool_node_fn(state):
            return {}

        wrapped = langfuse_node(tool_node_fn, name="tools")
        wrapped({})
        mock_client.start_as_current_span.assert_called_once_with(name="tools")


def test_langfuse_node_state_key_appends_section_name():
    """langfuse_node with state_key appends the value from state to span name."""
    mock_client = _make_mock_client()
    with patch("Utils.langfuse_tracing.get_client", return_value=mock_client):
        from importlib import reload
        import Utils.langfuse_tracing
        reload(Utils.langfuse_tracing)
        from Utils.langfuse_tracing import langfuse_node

        section = MagicMock()
        section.name = "半導體分析"

        def my_node(state):
            return {}

        wrapped = langfuse_node(my_node, state_key="section.name")
        wrapped({"section": section})
        mock_client.start_as_current_span.assert_called_once_with(name="my_node [半導體分析]")


def test_langfuse_node_state_key_missing_falls_back():
    """langfuse_node with state_key falls back to base name when key not in state."""
    mock_client = _make_mock_client()
    with patch("Utils.langfuse_tracing.get_client", return_value=mock_client):
        from importlib import reload
        import Utils.langfuse_tracing
        reload(Utils.langfuse_tracing)
        from Utils.langfuse_tracing import langfuse_node

        def my_node(state):
            return {}

        wrapped = langfuse_node(my_node, state_key="section.name")
        wrapped({"other_key": "value"})
        mock_client.start_as_current_span.assert_called_once_with(name="my_node")


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
