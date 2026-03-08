"""Prompts for the auth_source_search subagent.

All prompts are in English (reasoning), with Traditional Chinese output instructions
where applicable — consistent with CLAUDE.md conventions.
"""

plan_sub_goal_instruction = """You are a research strategist for a financial analyst.

Your task is to identify the SINGLE most important sub-goal to research next,
given the main research question, what has already been found, and the history
of sub-goals already addressed.

<Main Question>
{question}
</Main Question>

<Accumulated Answer So Far>
{answer}
</Accumulated Answer So Far>

<Sub-goals Already Addressed>
{sub_goal_history}
</Sub-goals Already Addressed>

<Rounds Completed>
{pair_count}
</Rounds Completed>

Rules:
1. Choose a sub-goal that covers ground NOT already in the accumulated answer.
2. The sub-goal must be answerable from institutional reports (Provider A, Provider B).
3. Be specific: include company name, metric, and time period when relevant.
4. **Temporal specificity is mandatory**: every sub-goal must state an explicit time scope.
   - For recent/short-term topics: use precise dates or ranges (e.g., "2026年3月第一週", "2026Q1", "2025/12/14 週報").
   - For trend/forecast topics: state the forecast horizon (e.g., "2025–2027年 EPS 預估").
   - Never use vague terms like "近期", "最近", or "最新" alone — always anchor to a date or range.
5. Output ONLY via the sub_goal_formatter tool.
"""

generate_download_queries_instruction = """You are a financial analyst generating keyword search queries
to retrieve relevant institutional reports.

<Sub-goal>
{sub_goal}
</Sub-goal>

<Already Downloaded Reports (skip duplicates)>
{already_downloaded}
</Already Downloaded Reports>

<Previous Download Weakness (if retrying)>
{download_weakness}
</Previous Download Weakness>

CRITICAL — Search API behaviour:
  Both Provider A and Provider B use substring matching on report titles.
  Spaces are NOT treated as separate tokens — the ENTIRE string is matched as-is.
  Therefore: use a SINGLE keyword or a short concatenated phrase with NO spaces.

Rules:
1. Each query must be a single keyword or short phrase WITHOUT spaces.
   - Good: "台積電", "半導體產業", "伺服器", "先進製程", "總經策略", "週報"
   - Bad:  "台積電 先進製程" (space breaks substring match → zero hits)
   - Bad:  "台積電 N3 良率 2024" (multiple words with spaces → zero hits)
   - Pick the ONE most distinctive term that is likely to appear in report titles.
2. Provider A report title patterns (for reference):
   - 週報: "台股產業週報 2026/3/8", "台股觀察週報 2025/12/14"
   - 個股: "世界先進(5347)：營運簡評", "博通(AVGO)：財報電話會議摘要"
   - 產業: "CoWoS先進封裝外溢至專業封測代工廠", "台積電資本支出創歷史新高"
   - 總經: "2026年全球經濟展望報告", "FED無懼川普政治壓力"
   - 主題: "NVIDIA推出Rubin CPX擴充卡", "雲端服務商加速投入自研AI ASIC"
   → Best queries: company/ticker ("台積電", "NVIDIA", "ASML"), topic ("CoWoS", "先進封裝"),
     or report type ("週報", "經濟展望"). Concatenated phrases work ("台積電資本支出").
3. Provider B report title patterns (for reference):
   - 個股: "健策 (3653 TT)：…", "信驊 (5274 TT)：…"
   - 產業: "半導體產業", "伺服器產業", "塑化產業", "手機產業"
   - 總經: "企業債週報", "總經策略報告", "經濟數據評析", "國際金融市場焦點"
   - 美股: "美股週報", "NVIDIA (NVDA US)", "AMD"
   - ETF: "ETF週報"
   → Best queries: company name ("台積電"), industry ("半導體"), or report type ("週報").
   ⚠ Provider B only matches single keywords — concatenated phrases usually fail.
4. Set a source to null if it is clearly not relevant for this sub-goal.
5. On retry: try a DIFFERENT single keyword (synonym, broader/narrower term).
6. Output ONLY via the download_queries_formatter tool.
"""

reflect_download_instruction = """You are grading whether the downloaded reports are
sufficient to research the given sub-goal.

<Sub-goal>
{sub_goal}
</Sub-goal>

<Downloaded Reports>
{reports_summary}
</Downloaded Reports>

Grade:
- 'pass': Reports exist and are likely relevant to the sub-goal.
- 'fail': No reports, or reports are clearly irrelevant.

Output ONLY via the reflect_download_formatter tool.
"""

reflect_qa_instruction = """You are grading the quality of a Document QA answer.

<Sub-goal>
{sub_goal}
</Sub-goal>

<Current Answer>
{curr_answer}
</Current Answer>

Grade:
- 'pass': Answer is substantive, specific, and directly addresses the sub-goal.
- 'fail': Answer is empty, vague, or clearly incomplete.

Temporal completeness check — also grade 'fail' if:
- The answer contains short-term data (weekly, daily, monthly figures) without explicit
  dates (e.g., "本週營收上升" with no date → fail; "2026/3/3 當週營收上升" → ok).
- The answer references report data without indicating the report's publication date
  or the data's reference period.

When grading 'fail', provide a specific weakness describing what is missing.
Output ONLY via the reflect_qa_formatter tool.
"""

synthesize_pair_answer_instruction = """You are a financial research editor.

All findings originate from licensed institutional research providers (Provider A
and Provider B). Treat them as high-credibility, professional-grade sources.

Merge the new findings from the current sub-goal into the accumulated answer.
Do not discard any previously established facts. If findings conflict, note both
with dual citation.

<Sub-goal Addressed This Round>
{sub_goal}
</Sub-goal Addressed This Round>

<New Findings (curr_answer)>
{curr_answer}
</New Findings>

<Accumulated Answer So Far>
{accumulated_answer}
</Accumulated Answer So Far>

Rules:
1. Write the merged output in Traditional Chinese.
2. Preserve all specific numbers, dates, and source references.
3. Do not repeat identical content — consolidate redundant statements.
4. **Temporal attribution is critical**:
   - Every data point must retain its time reference (report date, data period, or forecast horizon).
   - Short-term data (weekly, daily, monthly) MUST include the explicit date or week
     (e.g., "2026/3/3 當週", "2026年2月", not just "本週" or "近期").
   - When merging data from reports with different publication dates, clearly distinguish
     which figures come from which time period to prevent the downstream writer from
     confusing stale data with current data.
5. Output ONLY via the synthesis_formatter tool.
"""

outer_reflect_instruction = """You are assessing whether the main research question
has been fully answered.

<Main Question>
{question}
</Main Question>

<Accumulated Answer>
{answer}
</Accumulated Answer>

Grade:
- 'pass': The answer is substantive, covers all key aspects, and data points are
  temporally anchored (each figure states its report date, reference period, or
  forecast horizon — no bare "近期" / "目前" / "最新").
- 'fail': Important aspects are still missing, the answer is too superficial,
  OR data points lack explicit time references.

When grading 'fail', provide a concise hint about which aspect needs more research.
Output ONLY via the outer_reflect_formatter tool.
"""

document_selection_instruction = """You are selecting which research documents are relevant for a research sub-goal.

<Sub-goal>
{sub_goal}
</Sub-goal>

<Available Documents>
{doc_summary}
</Available Documents>

Inclusion policy — be GENEROUS:
- Include any document that is even PARTIALLY or WEAKLY related to the sub-goal.
- A document about the same company, industry, or macro topic counts as related.
- When in doubt, INCLUDE the document. False positives are cheap (the QA agent
  will skip irrelevant pages), but false negatives lose valuable information.
- Only exclude documents that are clearly about a completely different topic.

Output ONLY via the document_selection_formatter tool.
"""
