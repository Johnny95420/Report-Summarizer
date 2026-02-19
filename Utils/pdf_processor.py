# %%
import asyncio
import json
import os
import re
from io import StringIO

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM
from markdown_it import MarkdownIt
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


# %%
@tool
def financial_metadata_formatter(
    date: str,
    investment_target: str,
    company_rating: str,
    price_target: str,
    report_highlights: str,
):
    """Summary
    This is a tool to format the response.
    The content of the response needs to include: report date, investment target, company rating, target price, and report highlights.
    Args:
        date: The publication date of the report. If not mentioned, please use "None".
        investment_target: The name of the company or industry. If not mentioned, please use "None".
        company_rating: Including "Buy", "Neutral", or "Sell". If not mentioned, please use "None".
        price_target: The target price for the stock. If not mentioned, please use "None".
        report_highlights: Please summarize the first 5000 words into a summary of approximately 100 words.
    """
    return {
        "date": date,
        "investment_target": investment_target,
        "company_rating": company_rating,
        "price_target": price_target,
        "report_highlights": report_highlights,
    }


async def table_summarization(model_name, file_name, context_heading, context_paragraph, table):
    system_instructions = """
    You are an expert at summarizing the key content of tables and describing their contents.

    <Table and Content>
    {table}
    </Table and Content>

    <Task>
    - Your task is to summarize the message that the table intends to convey.
    You do not need to excessively describe the numbers and detailed content within the table.
    However, you need to clearly express what information the table is primarily communicating and what topics the information covers (without needing to list every single item).
    - Your descriptive style should be similar to the following examples:
    1. This is a table that compares different models on the XXX task. The table includes four models: A, B, C, and D, and the aspects being compared are the alpha and beta tasks.
    2. This is a financial statement for a specific company for a given year and month. The financial statement includes the contents A, B, C, D, and E.
    3. This is a table comparing companies A, B, and C. The table's content includes a comparison of aspects such as alpha, beta, and gamma.
    - You should summarize table in Traditional Chinese.
    </Task>

    <Limit>
    - The word count should be under 500 words.
    - You should summarize the specific and important information mentioned in the table.
    - Do not over-emphasize detailed content, such as specific numbers.
    - The style needs to be suitable for Retrieval-Augmented Generation (RAG).
    - Summarize table in Traditional Chinese.
    </Limit>
    """
    table = (
        f"FileName:{file_name}"
        + "\n"
        + f"Context Heading:{context_heading}"
        + "\n"
        + f"Context Paragraph:{context_paragraph}"
        + "\n"
        + f"Table: {table}"
    )
    writer_model = ChatLiteLLM(model=model_name, temperature=0)
    output = await writer_model.ainvoke(
        [SystemMessage(content=system_instructions.format(table=table))]
        + [HumanMessage(content="Please help me to summarize this table into description for doing RAG.")]
    )
    return output.content


async def financial_report_metadata_extraction(model_name, file_name, content):
    system_instructions = """
    You are an expert in extracting metadata from investment reports.
    I will provide you with the first 5000 words of a report.
    Based on the content of these 5000 words, you need to extract the report date, investment target, company rating, target price, and report highlights for me.

    <Content>
    {content}
    </Content>

    <Task>
    Your task is to extract key information from the beginning of the report, including:
    - date: The publication date of the report. If not mentioned, please use "None".
    - investment_target: The name of the company or industry. If not mentioned, please use "None".
    - company_rating: Including "Buy", "Neutral", or "Sell". If not mentioned, please use "None".
    - price_target: The target price for the stock. If not mentioned, please use "None".
    - report_highlights: Please summarize the first 5000 words into a summary of approximately 100 words.
    - All the content above should be provided in Traditional Chinese in your response.
    </Task>

    <Limit>
    - Use Traditional Chinese.
    - If you can not get the information please feedback this term in None. Do not generate it by your self.
    </Limit>
    """
    content = f"FileName:{file_name}" + "\n" + f"Content: {content}"
    tool_model = ChatLiteLLM(model=model_name, temperature=0).bind_tools(
        [financial_metadata_formatter], tool_choice="required"
    )
    output = await tool_model.ainvoke(
        [SystemMessage(content=system_instructions.format(content=content))]
        + [HumanMessage(content="Please help me to summarize this table into description for doing RAG.")]
    )
    try:
        return output.tool_calls[0]["args"]
    except Exception:
        output = await tool_model.ainvoke(
            [SystemMessage(content=system_instructions.format(content=content))]
            + [HumanMessage(content="Please help me to summarize this table into description for doing RAG.")]
        )
        return output.tool_calls[0]["args"]


metadata_extraction = financial_report_metadata_extraction
keys = [
    "date",
    "investment_target",
    "company_rating",
    "price_target",
    "report_highlights",
]


class PDFProcessor:
    def __init__(
        self,
        files: list[str],
        target_folder: str,
        do_extract_table: bool = True,
        model_name: str = "deepseek/deepseek-chat",
    ):

        self.files = files
        self.target_folder = target_folder
        os.makedirs(self.target_folder, exist_ok=True)
        self.do_extract_table = do_extract_table
        self.model_name = model_name
        self.converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )

    def convert_pdf_to_text(self, file):
        rendered = self.converter(file)
        text, _, _ = text_from_rendered(rendered)
        pattern = r"!\[\]\((.*?)\)"
        text = re.sub(pattern, "", text)
        return text

    def extract_table(self, md_text):
        md = MarkdownIt().enable("table")
        tokens = md.parse(md_text)
        lines = md_text.splitlines()
        results = []
        for i, token in enumerate(tokens):
            if token.type == "table_open":
                context_heading = None
                context_paragraph = None

                for j in range(i - 1, -1, -1):
                    if tokens[j].type == "heading_open":
                        if j + 1 < len(tokens) and tokens[j + 1].type == "inline":
                            context_heading = tokens[j + 1].content
                        break

                for j in range(i - 1, -1, -1):
                    if tokens[j].type == "paragraph_open":
                        if j + 1 < len(tokens) and tokens[j + 1].type == "inline":
                            context_paragraph = tokens[j + 1].content
                        break

                table_content = ""
                if token.map and len(token.map) == 2:
                    start_line, end_line = token.map
                    table_content = "\n".join(lines[start_line:end_line])
                    html_table = md.render(table_content)
                    pd.read_html(StringIO(html_table))[0].to_dict(orient="records")
                    results.append(
                        {
                            "context_heading": context_heading,
                            "context_paragraph": context_paragraph,
                            "table": table_content,
                        }
                    )
        return results

    async def summarize_table(self, file_name, context_heading, context_paragraph, table):
        summary_text = await table_summarization(self.model_name, file_name, context_heading, context_paragraph, table)
        return summary_text

    async def metadata_extraction(self, file_name, content):
        metadata = await metadata_extraction(self.model_name, file_name, content)
        return metadata

    async def parse(self):
        for f in self.files:
            name_w_extension = os.path.basename(f)
            name, _ = os.path.splitext(name_w_extension)
            text = self.convert_pdf_to_text(f)
            metadata = await self.metadata_extraction(name, text[:5000])
            metadata["full_content"] = text
            with open(
                f"{self.target_folder}/{name}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(metadata, f, ensure_ascii=False)

            if self.do_extract_table:
                tables = self.extract_table(text)
            if len(tables) == 0:
                continue

            tasks = [self.summarize_table(name, **table_data) for table_data in tables]
            summaries = await asyncio.gather(*tasks)

            for idx, (table, summary_txt) in enumerate(zip(tables, summaries)):
                table["summary"] = summary_txt
                for key in keys:
                    table[key] = metadata[key]
                with open(
                    f"{self.target_folder}/{name}_table_{idx}.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(table, f, ensure_ascii=False)

    def run_parse(self):
        asyncio.run(self.parse())
