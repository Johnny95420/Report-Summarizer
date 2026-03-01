import operator
from typing import Annotated, NotRequired, TypedDict


class WebResult(TypedDict):
    title: str
    content: str
    raw_content: str
    url: str
    score: NotRequired[int]


class WebResultBatch(TypedDict):
    results: list[WebResult]


class AgenticSearchState(TypedDict):
    # Primary input: main question with embedded sub-questions
    question: str
    # Search queries generated internally by generate_queries_from_question
    queries: list[str]
    followed_up_queries: list[str]
    # Parallel lists — must all have len == len(queries); zip() silently truncates if they differ
    web_results: list[WebResultBatch]
    filtered_web_results: list[WebResultBatch]
    compressed_web_results: list[WebResultBatch]
    # Current iteration's formatted search results (reset each round)
    materials: str
    # Iteratively updated answer with inline citations (persists across rounds)
    answer: str
    max_num_iterations: int
    curr_num_iterations: int
    num_queries: int
    url_memo: list[str]
    # Append-only registry of {title, url} for all quality-passed sources.
    # Position + 1 = stable citation number [N] across iterations.
    source_registry: Annotated[list[dict], operator.add]
    # Search locale determined by query language; defaults: gl="tw", hl="zh-tw"
    gl: str
    hl: str
    # Search time range chosen by the LLM; "all" means no restriction
    # Valid values: "day" | "week" | "month" | "year" | "all"
    time_filter: str
