"""
Shared query format rules for prompt instructions.

This module provides consistent query formatting guidelines across all
query-generating instructions in the research report system.

Version: 1.0
Last Updated: 2025-01-18
"""

# Core query format rules
QUERY_FORMAT_RULES = """
<Query Format Rules>
1. **Use KEYWORDS, not complete sentences**
   - ✅ Good: "台積電 N3 良率 2023 Q4"
   - ❌ Bad:  "請分析台積電在2023年第四季N3製程的良率表現"

2. **Standard Format**: [Entity] [Key Concept] [Timeframe if relevant]

3. **Length Limit**: Maximum 8 tokens per query
   - Tokens are space-separated words/numbers
   - Shorter queries generally perform better for search engines

4. **No Conjunctions**: Do not use and/or/but - use space separation only
   - ✅ Good: "台積電 資本支出 2024 指引"
   - ❌ Bad:  "台積電的資本支出以及2024年的指引"

5. **Search-Optimized Phrasing**: Frame for search engines, not human questions
   - ✅ Good: "Nvidia H100 規格 定位"
   - ❌ Bad:  "What are the specifications and positioning of Nvidia H100?"
</Query Format Rules>
"""

# Domain-specific query templates
QUERY_TEMPLATES_BY_DOMAIN = {
    "stock_tw": {
        "description": "Taiwan stock analysis queries",
        "templates": [
            "{公司名} {關鍵字} {年度/季度}",
            "{股票代號} {指標} {時間}",
            "{產業} 供應鏈 台廠",
        ],
        "examples": [
            "台積電 資本支出 2024 指引",
            "2330 營收 2023年12月",
            "AI伺服器 供應鏈 台廠",
        ]
    },
    "stock_global": {
        "description": "International stock analysis queries",
        "templates": [
            "{Company} {metric} {period}",
            "{ticker} earnings guidance {year}",
            "{sector} outlook {year}",
        ],
        "examples": [
            "Nvidia datacenter revenue Q4 2023",
            "AAPL buyback program 2024",
            "semiconductor outlook 2024",
        ]
    },
    "macro": {
        "description": "Macroeconomic indicator queries",
        "templates": [
            "{指標} {國家} {時間}",
            "{indicator} {country} forecast {year}",
            "Fed 利率 決策 {月份}",
        ],
        "examples": [
            "台灣 CPI 2023年12月",
            "US federal funds rate projection 2024",
            "Fed rate decision January 2024",
        ]
    },
    "futures": {
        "description": "Futures and derivatives queries",
        "templates": [
            "{商品} 期貨 {指標} {時間}",
            "{指數} 期貨 未平倉",
            "{選擇權} 履約價 持倉",
        ],
        "examples": [
            "台指期 未平倉 2024/01",
            "TXO put call ratio 2024",
            "VIX index correlation SPX",
        ]
    },
}

# Layered search strategy explanation
LAYERED_SEARCH_STRATEGY = """
<Layered Search Strategy>
When researching complex topics, generate queries across multiple layers:

**Layer 1 (Broad)**: General keywords for overview
- Example: "台積電 N3 製程 良率"

**Layer 2 (Focused)**: Specific aspects with more detail
- Example: "TSMC N3 yield improvement 2023"

**Layer 3 (Financial)**: Financial/market impact
- Example: "台積電 N3 毛利率 財報"

Each layer should use keyword format, not sentences.
</Layered Search Strategy>
"""

# Examples for demonstration (good vs bad)
QUERY_FORMAT_EXAMPLES = """
<Query Format Examples>

✅ Good Queries (Keyword-based):
- "台積電 N3 良率 2023 Q4"
- "美國 CPI 2023年12月"
- "台指期 未平倉 2024/01"
- "Nvidia H100 datacenter revenue"
- "AI晶片 供應鏈 台廠"

❌ Bad Queries (Sentence-based):
- "請分析台積電在2023年第四季N3製程的良率表現"
- "What was the US CPI data for December 2023?"
- "我需要找到台指期在2024年1月的未平倉量"
- "Can you show me Nvidia H100's revenue from datacenter segment?"
- "哪些台廠在AI晶片供應鏈中？"
</Query Format Examples>
"""

# Combined full instruction block
QUERY_FORMAT_INSTRUCTION_FULL = f"""
{QUERY_FORMAT_RULES}

{QUERY_FORMAT_EXAMPLES}
"""

# Compact version for token efficiency
QUERY_FORMAT_INSTRUCTION_SHORT = """
<Query Format>
- Use KEYWORDS, not sentences (3-8 tokens max)
- Format: [Entity] [Concept] [Time?]
- Examples: "台積電 N3 良率 2023 Q4" | "US CPI December 2023"
</Query Format>
"""
