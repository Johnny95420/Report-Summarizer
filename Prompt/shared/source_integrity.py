"""
Shared source integrity rules for prompt instructions.

This module provides consistent source citation and integrity rules across all
content-generating instructions in the research report system.

Version: 1.0
Last Updated: 2025-01-18
"""

# Full source integrity rules (from industry_prompt.py)
SOURCE_INTEGRITY_RULES_FULL = """
<STRICT SOURCE INTEGRITY RULES>
1. **ONLY CITE PROVIDED SOURCES**:
   - You MUST ONLY cite sources that are explicitly provided in the `<Source material>`.
   - It is ABSOLUTELY FORBIDDEN to create, fabricate, or hallucinate any source title, URL, date, or metadata.
   - If information is not supported by the provided sources, do not cite it. Either omit the claim or explicitly state it lacks source support.
   - Every inline citation `[Source Title]` must match EXACTLY with a source title appearing in the `<Source material>`.

2. **EXACT TITLE MATCHING REQUIRED**:
   - When citing, you MUST use the **EXACT** source title as it appears in the `<Source material>`.
   - Do NOT shorten, paraphrase, translate, or modify source titles.
   - The citation `[Source Title]` in your text MUST have an identical match in the source material list.

3. **SOURCE-CONTENT CORRESPONDENCE**:
   - Every claim you make MUST be directly supported by the cited source.
   - Do NOT cite a source for information it does not contain.
   - When synthesizing information from multiple sources, you MUST cite ALL sources that contribute to that synthesis.

4. **PROHIBITED CITATION PRACTICES**:
   - NEVER use vague, untraceable generic descriptions as source citations, such as "[News Report]", "[Company Website]", or "[Industry Analysis]".
   - You MUST use specific, identifiable source titles (e.g., media name + date + title keywords, or company/institution name + document type).
   - Trustworthy sources should have the following characteristics:
     * Clear media/platform name (e.g., news media, financial professional websites, official release platforms)
     * Reporter byline or clear data source attribution
     * Traceable publication date
     * Complete URL for verification
   - Avoid citing: anonymous blogs, social media personal posts, reposted content lacking source attribution, content with unidentifiable origins.
   - NEVER create URL-like references that do not exist in the source material.
   - NEVER fabricate dates or add "accessed on" information not provided.
   - If a source in the material lacks a title or date, mark it as "[Source without title]" rather than fabricating one.

5. **SOURCE LIST INTEGRITY**:
   - The `### Sources` section MUST ONLY include sources that (1) exist in the provided `<Source material>` AND (2) were actually cited in your content.
   - Do NOT add uncited sources to the source list.
   - Do NOT modify the provided URL or metadata when listing sources.
</STRICT SOURCE INTEGRITY RULES>
"""

# Compact version for token efficiency
SOURCE_INTEGRITY_RULES_SHORT = """
<Source Integrity Rules>
- ONLY cite sources explicitly provided in source material
- Use EXACT source titles - no paraphrasing or modifying
- Every claim must be supported by the cited source
- Cite ALL sources when synthesizing information
- NEVER use generic citations like "[News Report]" or "[Company Website]"
- Sources list must only include sources that exist AND were cited
</Source Integrity Rules>
"""

# Citation format rules
CITATION_FORMAT_RULES = """
<Citation Format Rules>
1. **Inline Citations**:
   - For any key data, statistics, or direct claims, provide an inline citation immediately after the statement.
   - Use the format `[Source Title]`.
   - If a statement synthesizes information from multiple sources, cite all of them: `[Source Title 1][Source Title 2]`.
   - All cited sources must also be listed in the final `### Sources` section.

2. **Sources Section Format**:
   - End with `### Sources` that references the source material.
   - List each source with title, date, and URL.
   - Format: `- Title <URL>` (Date if available)
</Citation Format Rules>
"""

# Source validation rules for final refinement stage
SOURCE_VALIDATION_REQUIREMENTS = """
<SOURCE VALIDATION REQUIREMENTS>
- **Citation Source Verification**: For EVERY citation `[Source Title]` in the content:
  * Verify that the EXACT source title exists in the `<Full Report Context>`.
  * If a cited source does NOT exist in the context, you MUST either:
    a) Find the correct source title from context and update the citation, OR
    b) Completely remove the unsupported claim.
  * NEVER leave a citation pointing to a non-existent source.
  * NEVER fabricate a new source to support a claim.

- **Source Metadata Accuracy**:
  * Use EXACT titles from source material - no paraphrasing or shortening.
  * Preserve original URLs exactly as provided.
  * Use original dates from source material without modification.
</SOURCE VALIDATION REQUIREMENTS>
"""

# Combined full instruction block
SOURCE_INTEGRITY_INSTRUCTION_FULL = f"""
{SOURCE_INTEGRITY_RULES_FULL}

{CITATION_FORMAT_RULES}
"""

# Warning about hallucination (for grader/reviewer prompts)
HALLUCINATION_DETECTION_RULES = """
<Hallucination Detection Rules>
When reviewing content for source citation integrity:

1. **Check Every Citation**: For each `[Source Title]` citation, verify:
   - Does this EXACT title appear in the provided source material?
   - Has the title been modified, shortened, or paraphrased?

2. **Red Flags for Hallucination**:
   - Generic citations like "[News Report]", "[Company Website]", "[Industry Analysis]"
   - Citations with plausible but non-existent titles
   - Citations with fabricated URLs or dates
   - Citations that don't match ANY source in the material

3. **Required Action for Hallucinations**:
   - If ANY citation points to a non-existent source, this is a CRITICAL FAILURE.
   - Generate targeted follow-up queries to verify the information.
   - Rate the content as `fail` until all citations are validated.
</Hallucination Detection Rules>
"""
