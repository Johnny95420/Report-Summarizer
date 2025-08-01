# %%
import operator
from typing import Annotated, List, Literal, TypedDict

from langchain.schema import Document
from pydantic import BaseModel, Field


# %%
class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    research: bool = Field(
        description="Whether to perform web research for this section of the report."
    )
    content: str = Field(description="The content of the section.")


class ReportStateInput(TypedDict):
    # Report topic
    topic: str


class ReportStateOutput(TypedDict):
    # Final report
    final_report: str


class ReportState(TypedDict):
    # Report topic
    topic: str
    # Feedback on the report plan
    feedback_on_report_plan: Annotated[list, operator.add]
    # List of report sections
    sections: list[Section]
    # Send() API key
    completed_sections: Annotated[list, operator.add]
    # String of any completed sections from research to write final sections
    report_sections_from_research: str
    # Final report
    final_report: str


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")


class Queries(BaseModel):
    queries: List[SearchQuery] = Field(description="List of search queries.")


class SectionState(TypedDict):
    # Report section
    section: Section
    # Number of search iterations done
    search_iterations: int
    # List of search queries
    search_queries: list[SearchQuery]
    # List of follow-up search queries
    follow_up_queries: list[SearchQuery]
    queries_history: Annotated[list, operator.add]
    # String of formatted source content from web search
    source_str: str
    # String of any completed sections from research to write final sections
    report_sections_from_research: str
    # Final key we duplicate in outer state for Send() API
    completed_sections: list[Section]


class SectionOutputState(TypedDict):
    # Final key we duplicate in outer state for Send() API
    completed_sections: list[Section]
