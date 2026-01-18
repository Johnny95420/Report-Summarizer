"""
Shared language protocol rules for prompt instructions.

This module provides consistent language selection rules across all query-generating
instructions in the research report system.

Version: 1.0
Last Updated: 2025-01-18
"""

# Full version with detailed explanations
LANGUAGE_RULES_FULL = """
<Language Protocol>
Determine the query language based on the subject's geographic scope:

**Traditional Chinese**: Use when the topic is EXCLUSIVELY related to:
- Taiwan (台灣)
- Taiwan-listed companies (台股上市櫃)
- Taiwan-specific policies, regulations, or economic data
- Taiwan industry analysis or supply chain

**English**: Use when the topic involves:
- United States, Europe, or international markets
- Global or comparative analysis
- Multi-country regions (APAC, ASEAN, etc.)
- Non-Taiwan Asian markets (Japan, Korea, Hong Kong, China)
- International companies (TSMC's global operations, Apple, Nvidia, etc.)

**Mixed Strategy**: If analyzing Taiwan in a global context:
- Use Traditional Chinese for Taiwan-specific queries
- Use English for international/comparative queries
</Language Protocol>
"""

# Simplified version for token-sensitive contexts
LANGUAGE_RULES_SHORT = """
<Language Rules>
- Taiwan-only topics: Traditional Chinese
- Global/US/Europe/Asia or international topics: English
</Language Rules>
"""

# Ultra-compact version for very token-constrained prompts
LANGUAGE_RULES_MINIMAL = """
<Language> Taiwan-only: 繁體中文 | Global/Intl: English </Language>
"""

# Domain-specific language guidance
LANGUAGE_RULES_BY_DOMAIN = {
    "stock_tw": "Use Traditional Chinese for Taiwan-listed company analysis, Taiwan market news, and domestic investor sentiment. Use English for global competitive analysis, international supply chain, or US market impact.",
    "stock_global": "Use English for all non-Taiwan markets (US, Hong Kong, China, Japan, Korea, Europe).",
    "macro": "Use Traditional Chinese for Taiwan economic data (CPI, GDP, unemployment from Taiwan government). Use English for international macro data (Fed rates, US CPI, global trade).",
    "futures": "Use Traditional Chinese for Taiwan futures (台指期, 選擇權, 小台). Use English for international futures and commodities.",
    "tech": "Use English for technical specifications, semiconductor processes, and technology standards. Use Traditional Chinese for Taiwan companies' adoption/application.",
}
