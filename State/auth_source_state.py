import operator
from typing import Annotated, TypedDict


class AuthReportState(TypedDict):
    # -- Input --
    question: str

    # -- Budget / config --
    max_pairs: int
    max_download_reflections: int
    max_qa_reflections: int
    qa_budget: int

    # -- Sub-goal --
    sub_goal: str
    sub_goal_history: Annotated[list, operator.add]

    # -- Download phase --
    download_queries: dict  # {"investanchor": str | None, "yuanta": str | None}
    download_weakness: str
    download_reflection_count: int

    # -- Accumulated reports --
    downloaded_reports: list[dict]  # [{"name": str, "path": str, "source": str}]

    # -- QA phase --
    selected_reports: list[dict]
    curr_answer: str
    qa_weakness: str
    qa_reflection_count: int

    # -- Navigator state sidecar (single-run safety net, NOT checkpoint resume) --
    navigator_state_path: str

    # -- Output --
    answer: str

    # -- Round tracking --
    pair_count: int
