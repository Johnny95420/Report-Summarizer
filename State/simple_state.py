from typing import TypedDict
from langchain_core.documents import Document


class RAGStateInput(TypedDict):
    topic: str


class RAGState(TypedDict):
    topic: str
    queries: list
    information: list[Document]
    completed_answer: str
    final_answer: str
    iteration_times: int
