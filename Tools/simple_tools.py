from typing import Literal

from langchain_core.tools import tool


@tool
def queries_formatter(queries: list[str]):
    """Summary
    Put queries into list
    Args:
        queries:List[str]: the queries in a list
    """
    return {"queries": queries}


@tool
def scores_formatter(score: float):
    """Summary
    Get a score
    Args:
        score:int: the score of question / doc pair relevance
    """
    return score


@tool
def final_judge_formatter(grade: Literal["pass", "fail"], queries: list[str]):
    """
        get the grade of section report
    Args:
        grade: Literal["pass","fail"]: Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail').
        queries: List[str]: List of follow-up search queries.
    """
    return {"grade": grade, "queries": queries}
