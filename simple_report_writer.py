import os
import pprint

from dotenv import load_dotenv

load_dotenv(".env")
from typing import Literal

import omegaconf
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from Prompt.simple_prompt import (answer_instructions, doc_judger_instructions,
                                  query_writer_instructions,
                                  section_grader_instructions)
from retriever import hybrid_retriever
from State.simple_state import RAGState, RAGStateInput
from Tools.simple_tools import (final_judge_formatter, queries_formatter,
                                scores_formatter)

config = omegaconf.OmegaConf.load("report_config.yaml")
MODEL_NAME = config["MODEL_NAME"]
VERIFY_MODEL_NAME = config["VERIFY_MODEL_NAME"]


def generate_queries(state: RAGStateInput, config: RunnableConfig):
    topic = state["topic"]
    configurable = config["configurable"]
    number_of_queries = configurable["number_of_queries"]

    system_instructions = query_writer_instructions.format(
        topic=topic, number_of_queries=number_of_queries
    )
    writer_model = ChatLiteLLM(model=MODEL_NAME, temperature=0.0)
    structure_model = writer_model.bind_tools(
        [queries_formatter], tool_choice="required"
    )
    results = structure_model.invoke(
        [SystemMessage(content=system_instructions)]
        + [HumanMessage(content="Generate relative queries on the provided topic.")]
    )
    queries = results.tool_calls[0]["args"]
    return queries_formatter.invoke(queries)


def search_relevance_doc(state: RAGState, config: RunnableConfig):
    queries = state["queries"]
    info = []
    for q in queries:
        if q == "":
            continue
        results = hybrid_retriever.invoke(q)
        for res in results:
            if res not in info:
                info.append(res)
    return {
        "queries": {},
        "information": info,
        "iteration_times": state.get("iteration_times", 0) + 1,
    }


def verify_relevance_doc(state: RAGState, config: RunnableConfig):
    topic = state["topic"]
    information = state["information"]
    judger_model = ChatLiteLLM(model=VERIFY_MODEL_NAME, temperature=0).bind_tools(
        tools=[scores_formatter], tool_choice="required"
    )
    final_docs = []
    for result in information:
        doc_judger = doc_judger_instructions.format(
            topic=topic,
            paragraph=result.page_content,
        )
        output = judger_model.invoke(
            [SystemMessage(content=doc_judger)]
            + [
                HumanMessage(
                    content="Please help me to clarify the relavance of question and paragraph."
                )
            ]
        )
        score = output.tool_calls[0]["args"]["score"]
        if score >= 0.2:
            final_docs.append(result)
    return {"information": final_docs}


def write_topic(state: RAGState, config: RunnableConfig):
    topic = state["topic"]
    information = state["information"]
    completed_answer = state.get("completed_answer", None)

    paragraphs = ""
    for res in information:
        paragraphs += f"Title : {res.metadata['path']}\n\n"
        paragraphs += f"Context : {res.page_content}\n\n"
        paragraphs += "=" * 80 + "\n\n"

    system_answer = answer_instructions.format(
        topic=topic, context=paragraphs, completed_answer=completed_answer
    )
    writer_model = ChatLiteLLM(model=MODEL_NAME, temperature=0.8)
    outputs = writer_model.invoke(
        [SystemMessage(content=system_answer)]
        + [HumanMessage(content="Please give me section report about this topic")]
    )
    return {"information": [], "completed_answer": outputs.content}


def check_completeness(
    state: RAGState, config: RunnableConfig
) -> Command[Literal[END, "search_relevance_doc"]]:
    topic = state["topic"]
    answer = state["completed_answer"]
    configurable = config["configurable"]
    max_search_depth = configurable["max_search_depth"]
    if state["iteration_times"] >= max_search_depth:
        return Command(update={"final_answer": answer}, goto=END)

    section_grader = section_grader_instructions.format(topic=topic, content=answer)
    grader_model = ChatLiteLLM(model=MODEL_NAME, temperature=0).bind_tools(
        tools=[final_judge_formatter], tool_choice="required"
    )
    outputs = grader_model.invoke(
        [SystemMessage(content=section_grader)]
        + [
            HumanMessage(
                content="Grade the report and consider follow-up questions for missing information"
            )
        ]
    )
    outputs = outputs.tool_calls[0]["args"]
    if outputs["grade"] == "pass":
        return Command(update={"final_answer": answer}, goto=END)
    else:
        return Command(
            update={"queries": outputs["queries"]}, goto="search_relevance_doc"
        )


builder = StateGraph(RAGState, input=RAGStateInput)
builder.add_node("generate_queries", generate_queries)
builder.add_node("search_relevance_doc", search_relevance_doc)
builder.add_node("verify_relevance_doc", verify_relevance_doc)
builder.add_node("write_topic", write_topic)
builder.add_node("check_completeness", check_completeness)

builder.add_edge(START, "generate_queries")
builder.add_edge("generate_queries", "search_relevance_doc")
builder.add_edge("search_relevance_doc", "verify_relevance_doc")
builder.add_edge("verify_relevance_doc", "write_topic")
builder.add_edge("write_topic", "check_completeness")
graph = builder.compile()
