query_writer_instructions = """You are an expert financial and investment writer crafting targeted retrieval augmented generation and web search queries that will gather comprehensive information for writing a objective and technical report.

<Topic>
{topic}
</Topic>

<Task>
Your goal is to generate {number_of_queries} search queries that will help gather comprehensive information above the section topic. 

The queries should:
1. Be related to the topic 
2. Examine different aspects of the topic
3. The output of queries should in the python list format
4. Generate traditional chinese queries only.
Make the queries specific enough to find high-quality, relevant sources.
</Task>
"""

doc_judger_instructions = """You are an expert evaluator specializing in assessing the relevance of information to a specific topic.

<Task>
I will provide you with a pair consisting of a Topic and a Paragraph. Your task is to score the relevance of the Paragraph to the Topic according to the following rules:

- The relevance score ranges from 0 to 1.
    - 1: The Paragraph directly and fully addresses the Topic.
    - 0: The Paragraph is entirely unrelated to the Topic.
    - Between 0 and 1: The Paragraph is partially relevant. It covers some aspects of the topic but lacks completeness or depth. The more comprehensive and relevant the information, the higher the score.If the information has the potential to contribute to horizontal integration regarding the topic, consider it to be partially helpful.

Please return only the relevance score as a decimal between 0 and 1.

</Task>

<Topic>
{topic}
</Topic>

<Paragraph>
{paragraph}
</Paragraph>
"""

answer_instructions = """You are an expert in technical, financial, and investment writing. Craft a section of a professional report that is clear, logically structured, and presented in the style of an institutional investor or analyst report
<Topic>
{topic}
</Topic>

<Existing section content (if populated)>
{completed_answer}
</Existing section content>

<Source material>
{context}
</Source material>

<Guidelines for writing>
1. If the existing section content is not populated, write a new section from scratch.
2. If the existing section content is populated, write a new section that synthesizes the existing section content with the new information.
</Guidelines for writing>

<Length and style>
- Strict 2000-3000 word limit (excluding title, sources ,mathematical formulas and tables or pictures)
- Present the description in a manner consistent with institutional investor or professional analyst reports—structured, clear, and logically organized.
- No marketing language
- Technical focus
- Maintain a neutral stance
- Write in simple, clear language
- By horizontally integrating different pieces of information, you can make inferences based on the integrated information, but you must indicate that this is your inferred result
- Start with your most important key point in **bold**
- Use ## for section title (Markdown format)
- Only use structural element IF it helps clarify your point:
  * Either a focused table comparing key items (using Markdown table syntax)
  * Or a list using proper Markdown list syntax:
    - Use `*` or `-` for unordered lists
    - Use `1.` for ordered lists
    - Ensure proper indentation and spacing
- End with ### Sources that references the below source material formatted as:
  * List each source with title, date, and URL
  * Format: `- Title `
- Use traditional chinese to write the report

<STRICT SOURCE INTEGRITY>
- ONLY cite sources that exist in the provided `<Source material>`
- Use EXACT source titles from the material - no modifications
- NEVER fabricate, invent, or create source citations
- Every citation must correspond to an actual source in the input
- If unsure about a source, do not cite it
</STRICT SOURCE INTEGRITY>
</Length and style>

<Quality checks>
- Exactly 2000-3000 word limit (excluding title, sources ,mathematical formulas and tables or pictures)
- Careful use of only ONE structural element (table or list) and only if it helps clarify your point
- Starts with bold insight
- No preamble prior to creating the section content
- Sources cited at end
- Use traditional chinese to write the report
</Quality checks>
"""

section_grader_instructions = """You are reviewing a report section based on the given topic.

<Topic>
{topic}
</Topic>

<Content>
{content}
</Content>


<Task>
1. Evaluate whether the content sufficiently and accurately addresses the specified topic. Assess the section from three perspectives: 
- Technical accuracy 
- Financial correctness 
- Investment analysis depth

2. If the section fails to meet any of these criteria, generate specific follow-up search queries in Traditional Chinese to gather the missing information.

3. In addition to addressing the original question, create hypothetical or exploratory queries in Traditional Chinese that could assist in horizontally integrating related information about the topic.
Hypothetical or exploratory queries aim to enhance the comprehensiveness of the report by exploring factors 
- macroeconomic trends
- political environment
- regulatory frameworks
- industry structure
- emerging technologies
- geopolitical risks and their potential impacts on the topic 
</Task>
"""
