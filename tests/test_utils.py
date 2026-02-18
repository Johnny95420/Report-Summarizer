"""Tests for Utils/utils.py"""

import ast
from pathlib import Path

from .conftest import ROOT, find_function


class TestContentExtractorQuery:
    """ContentExtractor.query() must deepcopy docs and append the expanded copy."""

    def test_uses_deepcopy_and_return_res(self):
        source = (ROOT / "Utils" / "utils.py").read_text()
        tree = ast.parse(source)

        func = find_function(tree, "query")
        assert func is not None, "ContentExtractor.query not found"

        src = ast.unparse(func)
        assert "deepcopy" in src, "query() must use deepcopy"
        assert "return_res" in src, "query() must use return_res variable"
        assert "info.append(return_res)" in src, (
            "query() must append return_res (the expanded copy)"
        )
