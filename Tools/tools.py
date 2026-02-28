from typing import Literal

from langchain_core.tools import tool


@tool
def searching_grader_formatter(grade: Literal["pass", "fail"], follow_up_queries: list[str]) -> dict:
    """Evaluate search completeness and provide follow-up queries if needed.

    [Agent Instruction]
    Call this tool once after evaluating whether the current answer sufficiently addresses
    the research question. If grade="fail", provide 1-2 follow-up search queries as a list
    of strings. Each element must be a complete, standalone keyword-based search string
    (3-8 words; up to 12 for complex multi-concept queries) — NOT a sentence or explanation.
    If grade="pass", set follow_up_queries to an empty list [].

    Args:
        grade (Literal["pass", "fail"]): Whether the current answer sufficiently covers
            the research question and all its sub-questions.
        follow_up_queries (list[str]): List of 1-2 keyword-based search strings targeting
            specific missing information. Each element is a standalone query string,
            not a sentence or a single character. Empty list when grade="pass".

    Returns:
        dict: {"grade": str, "follow_up_queries": list[str]}
    """
    return {"grade": grade, "follow_up_queries": follow_up_queries}


@tool
def searching_budget_formatter(budget: int) -> dict:
    """Set the maximum number of search iterations for the current research task.

    [Agent Instruction]
    Call this tool once at the start of a research session. Assign a budget of 1, 2, or 3
    based on question complexity. The maximum allowed value is 3.

    Args:
        budget (int): Number of search-and-synthesize iterations to allow (1, 2, or 3).

    Returns:
        dict: {"budget": int}
    """
    return {"budget": budget}


@tool
def quality_formatter(score: int) -> dict:
    """Score the relevance of a search result document to the query.

    [Agent Instruction]
    Call this tool once per document-query pair. Use the 1-5 scale:
    5=Very Relevant, 4=Relevant, 3=Moderately Relevant, 2=Slightly Relevant, 1=Irrelevant.

    Args:
        score (int): Relevance score between 1 and 5.

    Returns:
        dict: {"score": int}
    """
    return {"score": score}


@tool
def summary_formatter(summary_content: str) -> dict:
    """Produce a structured summary of search result content relevant to the query.

    [Agent Instruction]
    Call this tool once per search result to output the extracted and compressed content.

    Args:
        summary_content (str): Structured summary of the document preserving all relevant
            facts, figures, source titles, URLs, and publication dates.

    Returns:
        dict: {"summary_content": str}
    """
    return {"summary_content": summary_content}


@tool
def queries_formatter(thought: str, queries: list[str]) -> dict:
    """Package a search strategy and list of queries into structured format.

    [Agent Instruction]
    Call this tool once after deciding which queries to run.
    The queries field must be a list of strings — not a single string or comma-separated values.
    Each element is a complete standalone keyword-based search string.

    Args:
        thought (str): Brief explanation of the search strategy (1-2 sentences).
        queries (list[str]): List of keyword-based search query strings.

    Returns:
        dict: {"search_queries": list[str]}
    """
    return {"search_queries": queries}


@tool
def question_formatter(question: str) -> dict:
    """Structure the main research question with embedded sub-questions.

    [Agent Instruction]
    Call this tool once after decomposing the section topic into a research question.
    The question field must follow the format: main question on line 1, followed by
    sub-questions as bullet points prefixed with "- " (at most 5 sub-questions).

    Args:
        question (str): Main research question followed by sub-questions as bullet points.

    Returns:
        dict: {"question": str}
    """
    return {"question": question}


@tool
def answer_formatter(answer: str) -> dict:
    """Format the synthesized answer with inline citations.

    [Agent Instruction]
    Call this tool once after synthesizing search results into a comprehensive answer.
    The answer must include inline citations [N] for each claim.
    Do NOT write a ### Sources section — it will be appended automatically from the
    Source Registry.

    Args:
        answer (str): Synthesized answer with inline [N] citations. No Sources section.

    Returns:
        dict: {"answer": str}
    """
    return {"answer": answer}


@tool
def feedback_formatter(grade: Literal["pass", "fail"], weakness: str) -> dict:
    """Evaluate a report section and describe any weaknesses requiring revision.

    [Agent Instruction]
    Call this tool once after reviewing the section content against its description.
    Set grade="pass" if the section is complete and well-supported; grade="fail" otherwise.
    When grade="fail", weakness must describe the specific gaps clearly enough to guide
    the next research question. When grade="pass", set weakness to "".

    Args:
        grade (Literal["pass", "fail"]): Whether the section meets quality requirements.
        weakness (str): Specific description of what is missing or insufficient.
            Empty string if grade="pass".

    Returns:
        dict: {"grade": str, "weakness": str}
    """
    return {"grade": grade, "weakness": weakness}


@tool
def section_formatter(name: str, description: str, research: bool, content: str) -> dict:
    """Define a report section with its metadata and initial content.

    [Agent Instruction]
    Call this tool once per section when planning the report structure.
    description should be 150-300 words covering the section's scope, required data,
    and analytical focus. Leave content as an empty string "" during planning.

    Args:
        name (str): Short display title for this report section.
        description (str): 150-300 word overview of the section's scope, purpose,
            and the data/metrics required.
        research (bool): Whether this section requires external web research.
        content (str): Section body text. Use empty string during planning phase.

    Returns:
        dict: {"name": str, "description": str, "research": bool, "content": str}
    """
    return {
        "name": name,
        "description": description,
        "research": research,
        "content": content,
    }


@tool
def refine_section_formatter(refined_description: str, refined_content: str, weakness: str) -> dict:
    """Provide refinements to a section's description and content using full report context.

    [Agent Instruction]
    Call this tool once after reviewing the section against the full report context.
    refined_description must contain ONLY the additions or corrections to the original —
    do NOT repeat the original description text. Return empty string "" if no changes are needed.
    refined_content is the fully rewritten section body (100-1000 words).
    weakness describes the single most impactful remaining research gap, or "" if complete.

    Args:
        refined_description (str): Additions or corrections to the original description only
            (delta, not full replacement). Empty string if the original is already complete.
        refined_content (str): Fully rewritten section content (100-1000 words).
        weakness (str): Specific actionable gap description, or empty string if fully resolved.

    Returns:
        dict: {"refined_description": str, "refined_content": str, "weakness": str}
    """
    return {
        "refined_description": refined_description,
        "refined_content": refined_content,
        "weakness": weakness,
    }


@tool
def content_refinement_formatter(refined_content: str) -> dict:
    """Provide the final polished version of section content for publication readiness.

    [Agent Instruction]
    Call this tool once during the final refinement stage. Do NOT add any new facts or
    claims not already present in the original content. Only polish for consistency,
    remove duplication, and verify that every inline citation [N] has a matching source entry.

    Args:
        refined_content (str): Final polished section content with verified citations only.

    Returns:
        dict: {"refined_content": str}
    """
    return {
        "refined_content": refined_content,
    }
