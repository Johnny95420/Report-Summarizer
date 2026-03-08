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
2. The sub-goal must be answerable from institutional reports (InvestAnchor, Yunta).
3. Be specific: include company name, metric, and time period when relevant.
4. Output ONLY via the sub_goal_formatter tool.
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

Rules:
1. Generate SHORT keyword queries (3–8 words, e.g.: "台積電 N3 良率 2024 Q4").
2. For InvestAnchor: focused on Taiwan stocks / macro research.
3. For Yunta: focused on Taiwan equity / sector reports.
4. Set a source to null if it is clearly not relevant for this sub-goal.
5. On retry: change the keywords to address the weakness described above.
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

When grading 'fail', provide a specific weakness describing what is missing.
Output ONLY via the reflect_qa_formatter tool.
"""

synthesize_pair_answer_instruction = """You are a financial research editor.

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
4. Output ONLY via the synthesis_formatter tool.
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
- 'pass': The answer is substantive and covers all key aspects of the question.
- 'fail': Important aspects are still missing or the answer is too superficial.

When grading 'fail', provide a concise hint about which aspect needs more research.
Output ONLY via the outer_reflect_formatter tool.
"""
