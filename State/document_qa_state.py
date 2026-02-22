from typing import TypedDict

from langgraph.graph import MessagesState


class FileReference(TypedDict):
    name: str
    path: str


class DocumentQAState(MessagesState):
    """State for the Document QA LangGraph sub-agent."""

    file_paths: list[FileReference]  # input
    question: str  # input
    answer: str  # output
    iteration: int  # budget tracking
    budget: int  # max iterations
    consecutive_errors: int  # track consecutive LLM failures
    consecutive_text_only: int  # track consecutive text-only (no tool calls) responses
