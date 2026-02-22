from langgraph.graph import MessagesState

from Tools.reader_models import FileReference


class DocumentQAState(MessagesState):
    """State for the Document QA LangGraph sub-agent."""

    file_paths: list[FileReference]  # input
    question: str  # input
    answer: str  # output
    iteration: int  # budget tracking
    budget: int  # max iterations
    consecutive_errors: int  # track consecutive LLM failures; force_answer at >= _CONSECUTIVE_ERROR_THRESHOLD
    consecutive_text_only: (
        int  # track consecutive text-only responses; force_answer at >= _CONSECUTIVE_TEXT_ONLY_THRESHOLD
    )
