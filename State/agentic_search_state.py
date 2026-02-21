from typing import TypedDict


class AgenticSearchState(TypedDict):
    queries: list[str]
    followed_up_queries: list[str]
    web_results: list[dict]
    filtered_web_results: list[dict]
    compressed_web_results: list[dict]
    source_str: str
    max_num_iterations: int
    curr_num_iterations: int
    url_memo: set[str]
