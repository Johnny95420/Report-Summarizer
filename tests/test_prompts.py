"""Tests for Prompt/ template files."""

import re

import pytest

from .conftest import ROOT


class TestXMLTags:
    def test_follow_up_query_format_rules_closed(self):
        source = (ROOT / "Prompt" / "agentic_search_prompt.py").read_text()
        assert "</Follow-up Query Format Rules>" in source, "Closing tag </Follow-up Query Format Rules> not found"


class TestLengthUnits:
    """Prompt files should use 'words' not 'tokens' for length specifications."""

    PROMPT_FILES = [
        "Prompt/industry_prompt.py",
        "Prompt/agentic_search_prompt.py",
    ]

    @pytest.mark.parametrize("prompt_file", PROMPT_FILES)
    def test_no_tokens_unit_in_prompts(self, prompt_file):
        source = (ROOT / prompt_file).read_text()
        # Match patterns like "100 tokens", "50-100 tokens" — code var names are fine
        matches = re.findall(r"\b\d+[\s-]*tokens?\b", source, re.IGNORECASE)
        assert len(matches) == 0, f'{prompt_file} still uses "tokens" as length unit: {matches}'
