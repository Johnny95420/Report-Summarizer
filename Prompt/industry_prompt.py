"""
PROMPT VERSION: 2.0
LAST UPDATED: 2025-01-18
CHANGES:
- v2.0: Added query format optimization (keyword-based, 3-8 tokens)
- v2.0: Imported shared modules (language_rules, query_format, source_integrity)
- v2.0: Added ANTI-NARROWING PRINCIPLES to refine_section_instructions
- v1.9: Added strict source integrity rules
- v1.8: Initial version
"""

import datetime

from .shared.language_rules import LANGUAGE_RULES_SHORT
from .shared.query_format import QUERY_FORMAT_INSTRUCTION_SHORT

time = datetime.datetime.now()
curr_date = datetime.datetime.strftime(time, format="%Y/%m/%d")

report_planner_query_writer_instructions = (
    """You are an expert technical, financial and investment writer, helping to plan a report.
<Report topic>
{topic}
</Report topic>

<Report organization>
{report_organization}
</Report organization>

<Task>
Your goal is to generate {number_of_queries} search queries that will help gather comprehensive information for planning the report sections. 

The queries should:

1. Be related to the topic of the report
2. Help satisfy the requirements specified in the report organization
3. In Traditional Chinese

Make the queries specific enough to find high-quality, relevant sources while covering the breadth needed for the report structure.
</Task>

<Feedback>
Here is feedback on the report structure from review (if any):
{feedback}
</Feedback>
"""
    + f"<Current Time> {curr_date} </Current Time>"
)


report_planner_instructions = (
    """I want a plan for a report.
The more detailed and complete the information in this report, the better. 
The timing may be important for certain sections of this report. Please double-check carefully.


<Task>
**Generate a list of sections for the report in Traditional Chinese.**
You should genrate a list of sections and each section should have the fields:

- Name - Name for this section of the report (Traditional Chinese).
- description - Write a detailed, neutral, and objective overview of the main topics and purpose of this section.
  * 150-300 words.
  * Explicitly state the full background context — do not omit important events, policies, or key terms(include name, timepoint, events, location and any special description mentioned in topic)
  * Clearly describe what will be analyzed, explored, or created in this section, guiding the chapter's main direction and specifying the data required.
  * Structure the description with a focus on quantitative information and metrics (if suitable).
- Research - Whether to perform web research for this section of the report.
- Content - The content of the section, which you will leave blank for now.

For example, introduction and conclusion will not require research because they will distill information from other parts of the report.
</Task>

<Topic>
The topic of the report is:
{topic}
</Topic>

<Report organization>
The report should follow this organization: 
{report_organization}
</Report organization>

<Context>
Here is context to use to plan the sections of the report: 
{context}
</Context>

<Feedback>
Here is feedback on the report structure from review (if any):
{feedback}
</Feedback>
"""
    + f"<Current Time> {curr_date} </Current Time>"
)

query_writer_instructions = (
    f"""You are an expert financial and investment writer crafting targeted web search queries for report research.

<Task>
Your goal is to generate {{number_of_queries}} search queries that will help gather comprehensive information for the section topic.
</Task>

{QUERY_FORMAT_INSTRUCTION_SHORT}

<Query Strategy>
- Generate queries that examine different aspects of the topic
- Use layered approach: broad → focused → financial/technical
- Max 8 tokens per query

{LANGUAGE_RULES_SHORT}

<Topic>
{{topic}}
</Topic>
"""
    + f"<Current Time> {curr_date} </Current Time>"
)


section_writer_instructions = (
    """You are an expert in technical, financial, and investment writing.
You are now working at one of the world's largest industry research and consulting firms.
Your job is to craft a section of a professional report that is clear, logically structured, and presented in the style of an institutional investor or analyst report


<Guidelines for writing>
1.  **Synthesize Information**: Your primary task is to write a new, cohesive section that integrates the `Existing section content` (if any), the `Source material`, and answers any `Follow-up questions`.
2.  **Initial Draft**: If `Existing section content` is empty, create the first draft of the section based on the `Source material`.
3.  **Refine and Deepen**: If `Existing section content` exists, your goal is to enhance, deepen, and refine it using the new `Source material` and `Follow-up questions`, not just append new information.
4.  **Prioritize Timeliness**: Give strong preference to the most recent sources provided in the `<Source material>`. When discussing trends or data, always be mindful of the source's date. Avoid presenting outdated information as if it were current. If older data is necessary for historical context, explicitly state its time frame.
</Guidelines for writing>

<Length and style>
- Strict 100-1000 word limit (excluding title, sources ,mathematical formulas and tables or pictures)
- Start with your most important key point in **bold**
- Prefer quantitative metrics over qualitative adjectives in the description
- Writing in simple, clear language. Avoid marketing language; maintain a neutral tone
- Technical focus
- Time points aware
- Present the description in a manner consistent with institutional investor or professional analyst reports—structured, clear, and logically organized.
- Use ## only once per section for section title (Markdown format)
- Only use structural element IF it helps clarify your point:
  * Either a focused table (using Markdown table syntax) for
    - Comparing key items
    - Finanacial information
    - Quantitative information  
  * Or a list using proper Markdown list syntax:
    - Use `*` or `-` for unordered lists
    - Use `1.` for ordered lists
    - Ensure proper indentation and spacing
</Length and style>

<STRICT SOURCE INTEGRITY RULES>
1. **ONLY CITE PROVIDED SOURCES**: You MUST ONLY cite sources that are explicitly provided in the `<Source material>`.
   - It is ABSOLUTELY FORBIDDEN to create, fabricate, or hallucinate any source title, URL, date, or metadata
   - If information is not supported by the provided sources, do not cite it. Either omit the claim or explicitly state it lacks source support
   - Every inline citation `[Source Title]` must match EXACTLY with a source title appearing in the `<Source material>`

2. **EXACT TITLE MATCHING REQUIRED**:
   - When citing, you MUST use the **EXACT** source title as it appears in the `<Source material>`
   - Do NOT shorten, paraphrase, translate, or modify source titles
   - The citation `[Source Title]` in your text MUST have an identical match in the source material list

3. **SOURCE-CONTENT CORRESPONDENCE**:
   - Every claim you make MUST be directly supported by the cited source
   - Do NOT cite a source for information it does not contain
   - When synthesizing information from multiple sources, you MUST cite ALL sources that contribute to that synthesis

4. **PROHIBITED CITATION PRACTICES**:
   - NEVER use vague, untraceable generic descriptions as source citations, such as "[News Report]", "[Company Website]", or "[Industry Analysis]"
   - You MUST use specific, identifiable source titles (e.g., media name + date + title keywords, or company/institution name + document type)
   - Trustworthy sources should have the following characteristics:
     * Clear media/platform name (e.g., news media, financial professional websites, official release platforms)
     * Reporter byline or clear data source attribution
     * Traceable publication date
     * Complete URL for verification
   - Avoid citing: anonymous blogs, social media personal posts, reposted content lacking source attribution, content with unidentifiable origins
   - NEVER create URL-like references that do not exist in the source material
   - NEVER fabricate dates or add "accessed on" information not provided
   - If a source in the material lacks a title or date, mark it as "[Source without title]" rather than fabricating one

5. **SOURCE LIST INTEGRITY**:
   - The `### Sources` section MUST ONLY include sources that (1) exist in the provided `<Source material>` AND (2) were actually cited in your content
   - Do NOT add uncited sources to the source list
   - Do NOT modify the provided URL or metadata when listing sources
</STRICT SOURCE INTEGRITY RULES>

<Citation and Language Guidelines>
- **Inline Citations**: For any key data, statistics, or direct claims, you must provide an inline citation immediately after the statement. Use the format `[Source Title]`. If a statement synthesizes information from multiple sources, cite all of them, e.g., `[Source Title 1][Source Title 2]`. All cited sources must also be listed in the final `### Sources` section.
- End with `### Sources` that references the below source material formatted as:
  * List each source with title, date, and URL
  * Format: `- Title `
- Use traditional chinese to write the report
</Citation and Language Guidelines>

<Quality checks>
- Exactly 100-1000 word limit (excluding title, sources ,mathematical formulas and tables or pictures)
- Careful use of structural element (table or list) and only if it helps clarify your point
- Starts with bold insight
- No preamble prior to creating the section content
- Sources cited at end
- All key data, statistics, and claims are supported by inline citations, with multiple sources cited for synthesized information where applicable.
- Timeliness of information is prioritized; outdated data is contextualized correctly.
- Use traditional chinese to write the report
- Use quantitative metrics(if exist)
- Only contain relevant information
</Quality checks>

<Section Title>
{section_title}
</Section Title>

<Section topic>
{section_topic}
</Section topic>

<Existing section content (if populated)>
{section_content}
</Existing section content>

<Follow-up questions(if populated)>
{follow_up_queries}
</Follow-up questions>

<Source material>
{context}
</Source material>
"""
    + f"<Current Time> {curr_date} </Current Time>"
)

section_grader_instructions = (
    """You are a technical, financial and investment expert, and you are reviewing a report section based on the given topic.
Apply the **highest standards of rigor, accuracy, and professionalism**, as if you were a demanding Senior Executive in the Industry Research Division at J.P. Morgan Asset Management, known for **pushing for exceptional quality and identifying any potential weaknesses**.
Your goal is not just to pass or fail, but to **ensure the content reaches an exemplary standard through critical feedback.**

<Task>
1.  **Critical Evaluation & Gap Identification:**
    * Strictly and discerningly evaluate whether the content **comprehensively, deeply, and accurately** addresses the specified topic. Your evaluation must be **granular and evidence-based**.
    * For each of the following perspectives, **explicitly state:**
        * Whether the section **meets an exemplary standard (not just 'sufficient')**.
        * **Identify specific strengths and, more importantly, specific weaknesses or gaps** observed.
        * Provide **actionable recommendations** for improvement, even if the section is generally acceptable. Be specific about *what* is missing or *how* it could be improved.
    * Perspectives for evaluation:
        * **Technical Accuracy:** Is the information factually correct, precise, and up-to-date? Are technical terms used appropriately and explained if necessary? Are there any ambiguities or unsubstantiated claims?
        * **Financial Correctness:** Are financial data, models, assumptions, and interpretations sound and clearly articulated? Are calculations accurate and methodologies appropriate? Are financial concepts applied correctly and with necessary nuance?
        * **Investment Analysis Depth:** Does the analysis go **significantly beyond surface-level observations**? Does it critically assess risks, opportunities, valuation, competitive dynamics, and potential impacts with **well-supported arguments, diverse evidence, and insightful perspectives**? Is there a clear, defensible investment thesis or implication? Does it consider counterarguments or alternative scenarios?
        * **Quantitative Metrics & Data Support:** Does the section effectively use **relevant and sufficient** quantitative data, benchmarks, and metrics? Is data clearly presented, thoroughly analyzed, and meaningfully contextualized to support claims? Are sources credible and appropriately cited? Is the **significance and limitation of the data discussed**?
        * **Source Citation Integrity:**
          * Does EVERY inline citation `[Source Title]` correspond to a source that ACTUALLY EXISTS in the source material?
          * Are source titles used EXACTLY as provided (not paraphrased, shortened, or invented)?
          * Is there evidence of FABRICATED or HALLUCINATED sources?
          * If ANY citation points to a non-existent source, this is a CRITICAL FAILURE requiring immediate follow-up queries to verify information

    *Targeted Search Queries for Improvement (Mandatory if any weaknesses are identified or if the section is not 'exemplary'):*
        * Based on the **explicitly identified weaknesses, gaps, or areas needing more depth**, generate highly specific search queries designed to gather the exact missing information or to deepen the underdeveloped aspects of the analysis.
        * **Query Format Requirements**:
          - Use KEYWORD format, not sentences (3-8 tokens max)
          - Format: [Entity] [Concept] [Time?]
          - Examples: "台積電 N3 良率 Q4" | "Nvidia H100 規格" | "US CPI December 2023"
        * These queries should be phrases suitable for effective web searching (e.g., for academic databases, financial news, industry reports); avoid being overly declarative or too broad.

2.  **Hypothetical & Exploratory Queries for Broader Context (Generate these always):**
    * In addition to addressing any deficiencies, create insightful hypothetical or exploratory queries that could assist in horizontally integrating related information and broadening the report's strategic understanding of the topic.
    * These queries should aim to provoke deeper thought and explore factors such as (but not limited to):
        * Macroeconomic trends (e.g., inflation, interest rates, GDP growth forecasts and their second-order effects)
        * Political environment and specific policy impacts (current and anticipated)
        * Regulatory frameworks (existing, changing, and comparative analysis if relevant)
        * Industry structure (e.g., Porter's Five Forces, value chain analysis, competitive intensity)
        * Emerging technologies and disruptive innovations (potential impact, adoption rates, barriers)
        * Geopolitical risks and their quantifiable or qualitative potential impacts on the topic
        * ESG (Environmental, Social, Governance) considerations relevant to the topic.
        
3.  **Identify Drill-Down Opportunities:**
    * While evaluating the content, proactively identify the most critical, interesting, or noteworthy **'Key Findings'** that warrant deeper investigation. This could be a 
     specific data point, a significant event, or an unexpected trend.
    * If such findings are identified, generate specific **'drill-down' queries** in the `follow_up_queries`. These queries must be highly specific, designed to uncover the underlying
     details, causes, or impacts of that finding.
    * If the section is generally well-written but you have identified a Key Finding that requires further detail, you **must** rate the `grade` as `fail`. This will trigger a
     'drill-down' research loop. Only rate the `grade` as `pass` once the content is comprehensive and all identified Key Findings have been sufficiently explored and integrated.

4.  **Language for search queries:**
    * If the follow-up search query is only **related to Taiwan, use Traditional Chinese** queries only.
    * If the follow-up search query is **related to Europe, America, the broader Asia-Pacific region, or globally, use English queries.**

5.  **Query Uniqueness and Evolution:**
    *   **Review History:** Before generating any new queries, you must carefully review the `Queries History`.
    *   **Avoid Semantic Duplication:** Strictly prohibit generating queries that are semantically identical or highly similar to any existing queries in the history.
    *   **Deepen, Don't Repeat:** If a topic requires more information, formulate a new query that approaches it from a different angle, at a deeper level, or investigates its root causes, rather than simply repeating or slightly rephrasing an old query. The goal is to uncover new information, not to retrieve the same content again.

6.  **Query Conciseness and Search Effectiveness:**
    * **Keep It Concise**: The Query should not be overly long.
    * **Avoid Over-Description:** Do not make the Query excessively descriptive or narrative.
    * **Optimize for Google Search:** Since the Query will be sent to Google Search, it must be crafted to maximize both recall and precision in the returned results.

6.  **Query Prioritization and Limit:**
    *   **Total Limit:** Generate a maximum of 3 queries in total.
    *   **Selection Priority:** Prioritize the queries to generate based on this strict order of importance, ensuring the most critical issues are addressed first:
        1.  **Targeted Improvement Queries (from Task 1):** Highest priority. Generate these to fix specific, identified content weaknesses or gaps.
        2.  **Drill-Down Queries (from Task 3):** Second priority. These are mandatory if you identify a 'Key Finding' that requires deeper investigation.
        3.  **Hypothetical/Exploratory Queries (from Task 2):** Lowest priority. Generate these only if the 3-query limit has not been met by the higher-priority tasks.
</Task>

<Section topic>
{section_topic}
</Section topic>

<Queries History>
{queries_history}
</Queries History>

<Section content>
{section}
</Section content>
"""
    + f"<Current Time> {curr_date} </Current Time>"
)

final_section_writer_instructions = (
    """You are an expert technical, financial and investment writer crafting a section that synthesizes information from the rest of the report.
Apply the high standards of accuracy and professionalism, as if you were a senior executive in the Industry Research Division at J.P. Morgan Asset Management.

<Task>
1. Section-Specific Approach:

For Introduction:
- Use # for report title (Markdown format)
- 200-1000 word limit
- Use Traditinal Chinese
- Provide readers with a clear understanding of the industry as a whole, the report’s logical structure, and its core insights.
- Write in simple and clear language
- Focus on the core motivation for the report in 1-2 paragraphs
- Use a clear narrative arc to introduce the report
- Include NO structural elements (no lists or tables)
- No sources section needed

For Conclusion/Summary:
- Use ## for section title (Markdown format)
- 200-1000 word limit
- Use Traditinal Chinese
- Horizontally integrate information from different sections and provide objective, neutral insights.
- For comparative reports:
    * Must include a focused comparison table using Markdown table syntax
    * Table should distill insights from the report
    * Keep table entries clear and concise
- For non-comparative reports: 
    * Only use structural element IF it helps distill the points made in the report:
    * Either a focused table comparing items present in the report (using Markdown table syntax)
    * Or a short list using proper Markdown list syntax:
      - Use `*` or `-` for unordered lists
      - Use `1.` for ordered lists
      - Ensure proper indentation and spacing
- End with specific next steps or implications 
- No sources section needed

3. Writing Approach:
- Use concrete details over general statements
- Make every word count
</Task>

<Quality Checks>
- Use Traditinal Chinese
- For introduction: 200-1000 word limit, # for report title, no structural elements, no sources section
- For conclusion: 200-1000 word limit, ## for section title
- Do not include word count or any preamble in your response
</Quality Checks>

<Section topic> 
{section_topic}
</Section topic>

<Available report content>
{context}
</Available report content>
"""
    + f"<Current Time> {curr_date} </Current Time>"
)

refine_section_instructions = (
    """You are an expert report editor and retrieval planner. Your task is to refine ONE specific section of a report by leveraging the FULL context of all other sections, then propose targeted web search queries to close evidence gaps.

<Task>
1) Rewrite the section's "description" and "content" using the full report context.
2) Produce "queries" to obtain missing facts, metrics, or corroboration.
</Task>

<ANTI-NARROWING PRINCIPLES>
- **Dimension Preservation**: When refining descriptions, you MUST preserve ALL analytical dimensions from the original
  * Examples of dimensions: market background, financial metrics, competitive landscape, risk factors, technical specifications
  * DO NOT reduce multi-dimensional analysis to single focus

- **Deepen, Don't Replace**: Your refinement should ADD specificity to existing dimensions, not remove them
  * ❌ Bad: "Focus only on N3 yield improvement"
  * ✅ Good: "Analyze N3 yield trends (historical + current), impact on gross margins, customer adoption rates, and competitive comparison with Samsung GAA"

- **Missing Dimension Detection**: Identify dimensions that are MISSING from the original and ADD them
  * If a section on "competitive analysis" only mentions one competitor, add queries for others
  * If "risk factors" are absent, explicitly add them

- **Output Format**: For refined_description, output ONLY the additions - do not repeat the original
</ANTI-NARROWING PRINCIPLES>

<Rigorous Principles>
- Write the final description and content in **Traditional Chinese**.
- **Information Scrutiny & Fact-Checking**:
    - **Zero Tolerance for Hallucination**: If a fact, number, or claim is not explicitly supported by the `<Full Report Context>` or its original `[Source]` markers, it must be treated as unverified. Do not invent, infer, or embellish information.
    - **Challenge Vague or Uncertain Information**: If content within the `<Target Section to Refine>` is vague, lacks sufficient detail, appears speculative, or is not corroborated by the broader report context, you must either:
        a) **Remove it** if it cannot be substantiated.
        b) **Flag it** by generating a precise query under `<Query Requirements>` to seek verification, quantification, or clarification.
    - **Prioritize Verifiable Facts**: When rewriting, give strong preference to information that is clearly sourced and quantitatively supported. Downplay or remove speculative or poorly substantiated claims.
- Prefer quantitative detail when suitable (KPI, YoY/HoH, penetration, valuation multiples, capacity, ASP, users, conversion, margins, etc.).
- **Cross-Section Integrity**: Strictly maintain logical boundaries between sections. Information must be placed in its most appropriate section. When refining, **remove content that belongs in other sections** and avoid duplicating material. Use brief cross-references (e.g., `詳見[其他章節名稱]`) where needed.
- **Do not delete** any existing source markers in the original content (e.g., [來源], [Source]).
</Rigorous Principles>

<STRICT SOURCE CITATION INTEGRITY>
- **ABSOLUTE PROHIBITION ON SOURCE FABRICATION**:
  * You are ABSOLUTELY FORBIDDEN from creating, inventing, or hallucinating any source citation
  * ALL citations (including existing `[來源]`, `[Source]`) MUST correspond to sources that appear in the `<Full Report Context>`
  * When adding new citations, use ONLY source titles that EXACTLY MATCH sources in the provided context
  * NEVER modify existing citation markers to point to non-existent sources
  * If you cannot verify a source's existence in the context, REMOVE the claim rather than fabricating a citation
  * The title in any `[Source Title]` citation MUST appear verbatim in the source material
</STRICT SOURCE CITATION INTEGRITY>

<Tone and Style Guidelines>
- Maintain a professional, neutral, and objective tone consistent with institutional research.
</Tone and Style Guidelines>

<Description Requirements>
For "description":
1) **Do not repeat the original description in your output.**
2) Based on the full report context, identify what is missing or needs correction in the original description. **Output only the text for these additions or corrections.**. Your additions should aim to:
   - Integrates full-report context and explicitly states background (key events/policies/terms, names, timepoints, locations, special descriptors).
   - Based on the full text, provide a more comprehensive and complete description of the section, guiding the section to obtain more complete and in-depth content in the subsequent research.
   - Deepens guidance for how this section should be written without weakening or narrowing the original meaning.
   - Clearly defines what will be analyzed/explored/built and the data required.
   - When suitable, structure around quantitative metrics and methods.
   - **Ensure the description is tightly focused on the section's specific topic.** The guidance should not bleed into topics covered by other sections. The goal is to create a clear and distinct mandate for this section alone.
3) Avoid repeating information already in the section's description. Add only new descriptions to ensure completeness. If no new descriptions are needed, return an empty string in `refined_description`.
4) If you detect inconsistency between the original description and the full context, start your output with a **"Correction Note:"** paragraph explaining the mismatch and the correct context (citing the relevant parts of the full context).
</Description Requirements>


<Content Requirements>
For "content":
1) **Core Task**: Produce a more comprehensive, well-structured, and factually sound narrative aligned with the refined description and the full report. **Your primary goals are to ensure information is correctly placed and factually accurate.**
   - You may reorganize, clarify, and enrich the original content.
   - **Preserve Verifiable Information**: Preserve all important, verifiable information that is relevant to this section's topic, along with its existing source markers (e.g., `[來源]`, `[Source]`).
   - **Handle Unverified Information**: Any information that is vague, speculative, or cannot be corroborated by the `<Full Report Context>` must be handled according to the **Information Scrutiny & Fact-Checking** principle (i.e., it should be removed or flagged for verification via a query).
   - **Remove Misplaced Information**: You must remove information that clearly belongs in a different section. This is critical for keeping each section focused and avoiding clutter.
2) **Cross-Section Consistency**: Avoid repeating material from other sections; if necessary, use a brief cross-reference (e.g., “詳見 other_section_name”) instead of duplicating text.
3) **Style and Formatting**:
    - **Word Count**: 100-1000 word limit (excluding title, sources, mathematical formulas, tables, or pictures).
    - **Opening**: Start with your most important key point in **bold**.
    - **Tone & Focus**: Maintain a neutral, technical, and time-aware tone consistent with institutional analyst reports. Prefer quantitative metrics over qualitative adjectives. Avoid marketing language.
    - **Title**: Use `##` only once for the section title (Markdown format).
    - **Structural Elements**: Only use a structural element IF it helps clarify your point:
      * Either a focused table (using Markdown table syntax) for comparing key items, financial information, or quantitative data.
      * Or a list using proper Markdown list syntax (`*`, `-`, `1.`).
    - **Inline Citations**: For any key data, statistics, or direct claims, provide an inline citation immediately after the statement (e.g., `[Source Title]`). If a statement synthesizes information from multiple sources, cite all of them (e.g., `[Source Title 1][Source Title 2]`).
    - **Sources Section**: End with `### Sources` that references the source material, formatted as:
      * List each source with title, date, and URL.
      * Format: `- Title `
    - **Language**: Use **Traditional Chinese** to write the report.
</Content Requirements>


<Query Requirements>
Generate **{number_of_queries}** targeted queries to fill explicit gaps you flagged in the content and to deepen analysis:
1) **Query Format**: Use KEYWORD format, not sentences (3-8 tokens max)
   - Format: [Entity] [Concept] [Time?]
   - Examples: "台積電 N3 良率 Q4" | "Nvidia H100 規格" | "US CPI December 2023"
2) Each query must map to a concrete missing data point, validation need, or analytical deepening you identified.
3) Cover multiple angles as needed: statistics, regulations/policy, financial disclosures, industry reports, technical specs/standards, benchmarks/peers, and risk events (as applicable).
4) Language rules:
   - If the topic pertains **only to Taiwan**, use **Traditional Chinese** queries.
   - If it concerns **Europe/US/APAC or global** scope, use **English** queries.
5) Make queries highly retrievable: include time bounds (e.g., 2019..2025, "Q2 2024"), key entities (companies/products/locations/standards), and operators when useful (e.g., site:, filetype:pdf, intitle:).
6) No semantic duplicates; each query should solve a different gap or approach.
</Query Requirements>


<Quality Checks>
- **Language**: The final `refined_description` and `refined_content` are written in Traditional Chinese.
- **Description Output**:
    - The `refined_description` output contains **only the additions or corrections**, not the full original description.
    - The additions do not repeat information already present in the original description. If no new information can be added, the output is an empty string.
    - If an inconsistency was found, the output starts with a **"Correction Note:"** paragraph.
    - The additions integrate context from the full report, clarify background details (events, names, timepoints), and provide deeper, more comprehensive guidance for the section.
    - The guidance clearly defines the analysis to be performed and the data required, using quantitative framing where appropriate.
- **Content Output**:
    - The `refined_content` is a comprehensive, well-structured, and **factually sound** narrative that aligns with the refined description.
    - It **preserves all important, verifiable information *relevant to the section*** and its associated source markers (e.g., `[來源]`, `[Source]`).
    - It **removes or flags unverified/speculative information** according to the prompt's principles.
    - It **removes information that clearly belongs in other sections**, ensuring the section is focused.
    - It avoids duplicating content from other sections, using cross-references if needed.
    - It adheres to all style and formatting rules: 100-1000 words, starts with a bold key point, uses `##` for the title, includes inline citations for all key claims, and ends with a correctly formatted `### Sources` section.
- **Query Output**:
    - Exactly **{number_of_queries}** queries are generated.
    - Each query is specific, targets a clearly identified gap in the content, and is designed to be highly retrievable (using time bounds, entities, or operators where useful).
    - Queries are non-overlapping (no semantic duplicates) and follow the specified language rules (Traditional Chinese for Taiwan-only topics, English otherwise).
</Quality Checks>

<Full Report Context>
{full_context}
</Full Report Context>

<Target Section to Refine>
- **Name:** {section_name}
- **Original Description:** {section_description}
- **Original Content:** {section_content}
</Target Section to Refine>
"""
    + f"<Current Time> {curr_date} </Current Time>"
)

content_refinement_instructions = (
    """You are an expert report editor performing FINAL CONTENT POLISHING in the last stage of report production. This is the ultimate quality assurance pass before report publication. Your task is to refine the content of ONE specific section to achieve institutional-grade consistency and integration with the complete report context.

<Task>
This is the **FINAL REFINEMENT STAGE** - your role is to polish content for publication readiness by:
1. **Ensuring absolute consistency with full report context** - align terminology, data points, cross-references, and narrative flow across all sections
2. **Eliminating all inconsistencies and redundancies** - remove content duplication, resolve conflicting statements, ensure logical coherence
3. **Preserving and validating all factual accuracy** - maintain source citations, verify data consistency, remove unsubstantiated claims
4. **Achieving institutional research standards** - ensure professional tone, precise language, and analytical rigor throughout
5. **Optimizing readability and structure** - enhance logical flow, strengthen transitions, improve clarity without changing substance
</Task>

<Critical Final-Stage Principles>
- Write the refined content in **Traditional Chinese**.
- **FINAL STAGE ZERO TOLERANCE FOR HALLUCINATION**: This is the last opportunity to catch errors. Only use information explicitly supported by the original content and full report context. Absolutely no new facts, numbers, or claims may be added. Any unsupported information must be removed or flagged.
</Critical Final-Stage Principles>

<SOURCE VALIDATION REQUIREMENTS>
- **Citation Source Verification**: For EVERY citation `[Source Title]` in the content:
  * Verify that the EXACT source title exists in the `<Full Report Context>`
  * If a cited source does NOT exist in the context, you MUST either:
    a) Find the correct source title from context and update the citation, OR
    b) Completely remove the unsupported claim
  * NEVER leave a citation pointing to a non-existent source
  * NEVER fabricate a new source to support a claim

- **Source Metadata Accuracy**:
  * Use EXACT titles from source material - no paraphrasing or shortening
  * Preserve original URLs exactly as provided
  * Use original dates from source material without modification
</SOURCE VALIDATION REQUIREMENTS>

<Final Content Quality Requirements>
- **Comprehensive Source Validation**: Maintain and verify all existing source markers (e.g., [來源], [Source]) from the original content. Ensure all citations are properly formatted and referenced.
- **Final Cross-Section Integrity Check**: This is your last chance to ensure proper section boundaries. Remove content that clearly belongs in other sections. Use brief cross-references (e.g., `詳見[其他章節名稱]`) where needed to maintain coherence.
- **Publication-Ready Professional Standards**: Maintain neutral, objective tone consistent with institutional research. Apply the highest standards of accuracy and professionalism, as if you were a senior executive in the Industry Research Division at J.P. Morgan Asset Management.
- **Final Format Validation**: Preserve the original section structure and formatting requirements:
  - **Word Count**: 100-1000 word limit (excluding title, sources, mathematical formulas, tables, or pictures)
  - **Opening**: Ensure it starts with the most important key point in **bold**
  - **Title**: Use `##` only once for the section title (Markdown format)
  - **Structural Elements**: Only retain structural elements that genuinely clarify points (focused tables or proper Markdown lists)
  - **Inline Citations**: Verify all key data, statistics, and claims have immediate inline citations (e.g., `[Source Title]`)
  - **Sources Section**: Confirm it ends with properly formatted `### Sources` section
</Final Content Quality Requirements>

<Final Quality Validation>
Before completing, verify:
1. **Consistency Achieved**: All terminology, data points, and narrative elements align perfectly with the full report context
2. **Redundancies Eliminated**: No content duplication exists across sections
3. **Factual Integrity Maintained**: All information is supported by original content or full report context
4. **Professional Standards Met**: The content meets institutional research publication standards
5. **Format Requirements Satisfied**: All structural and formatting requirements are correctly implemented
6. **Source Integrity Preserved**: All original citations and source markers are maintained and properly formatted
</Final Quality Validation>

<Full Report Context>
{full_context}
</Full Report Context>

<Target Section to Refine>
- **Name:** {section_name}
- **Original Content:** {section_content}
</Target Section to Refine>
"""
    + f"<Current Time> {curr_date} </Current Time>"
)
