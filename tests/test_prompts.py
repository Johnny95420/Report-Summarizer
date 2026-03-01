"""Tests for Prompt/ template files."""

import re

import pytest

from .conftest import ROOT


class TestXMLTags:
    def test_follow_up_query_format_rules_closed(self):
        source = (ROOT / "Prompt" / "agentic_search_prompt.py").read_text()
        assert "</Follow-up Query Format Rules>" in source, "Closing tag </Follow-up Query Format Rules> not found"


class TestWordLimits:
    """Key prompts must use the correct word-count ranges."""

    def test_refine_section_instructions_word_limit(self):
        source = (ROOT / "Prompt" / "industry_prompt.py").read_text()
        # Should specify 500-2000 words, not the old 100-1000
        assert "500-2000" in source, (
            "refine_section_instructions / content_refinement_instructions should use 500-2000 word range"
        )
        assert "100-1000" not in source, (
            "Old word limit '100-1000' still present in industry_prompt.py"
        )

    def test_section_writer_word_limit(self):
        source = (ROOT / "Prompt" / "industry_prompt.py").read_text()
        # section_writer_instructions must not use old 100-1000 range
        assert "100-1000" not in source, (
            "Old word limit '100-1000' still present; update to 500-2000"
        )


class TestLengthUnits:
    """Prompt files should use 'words' not 'tokens' for length specifications."""

    PROMPT_FILES = [
        "Prompt/industry_prompt.py",
        "Prompt/agentic_search_prompt.py",
        "Prompt/document_qa_prompt.py",
    ]

    @pytest.mark.parametrize("prompt_file", PROMPT_FILES)
    def test_no_tokens_unit_in_prompts(self, prompt_file):
        source = (ROOT / prompt_file).read_text()
        # Match patterns like "100 tokens", "50-100 tokens" — code var names are fine
        matches = re.findall(r"\b\d+[\s-]*tokens?\b", source, re.IGNORECASE)
        assert len(matches) == 0, f'{prompt_file} still uses "tokens" as length unit: {matches}'
