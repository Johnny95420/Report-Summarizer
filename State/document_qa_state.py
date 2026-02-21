from langgraph.graph import MessagesState


class DocumentQAState(MessagesState):
    """State for the Document QA LangGraph sub-agent."""

    file_paths: list[dict]  # input: [{"name": ..., "path": ...}]
    question: str  # input
    answer: str  # output
    iteration: int  # budget tracking
    budget: int  # max iterations
