import operator
from typing import Annotated, TypedDict


class AgenticSearchState(TypedDict):
    # Primary input: main question with embedded sub-questions
    question: str
    # Search queries generated internally by generate_queries_from_question
    queries: list[str]
    followed_up_queries: list[str]
    web_results: list[dict]
    filtered_web_results: list[dict]
    compressed_web_results: list[dict]
    # Current iteration's formatted search results (reset each round)
    materials: str
    # Iteratively updated answer with inline citations (persists across rounds)
    answer: str
    max_num_iterations: int
    curr_num_iterations: int
    num_queries: int
    url_memo: set[str]
    # Append-only registry of {title, url} for all quality-passed sources.
    # Position + 1 = stable citation number [N] across iterations.
    source_registry: Annotated[list[dict], operator.add]
