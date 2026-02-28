# %%
import asyncio
import os

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM

from Tools.tools import queries_formatter
from Utils.utils import call_search_engine, web_search_deduplicate_and_format_sources


# %%
@tool
def background_knowledge_formatter(hashtags: list[str], knowledge: str):
    """Summary
    This is a tool to format the response.
    You need to put generated key hot words in a python list and knowledge in markfown style string(around 300~500 words)
    Args:
        hashtags:List[str] = Some generated key hot words about knowledge
        knowledge:str = information about certain topic
    """
    return {"hashtags": hashtags, "knowledge": knowledge}


async def get_background_knowledge(model_name: str, key_word: str):
    system_instruction = """Please help me generate three queries for the keyword .
    These queries are intended to help me better understand the keyword and assist me in writing a 300-500 word short column

    <Key Word>
    {key_word}
    </Key Word>

    <Task>
    Your task is to generate three to five queries to help me conduct online research for a given keyword.
    Please ensure that the generated queries are in the same language as the input keyword.
    For example:
        A keyword in Traditional Chinese should generate queries in Traditional Chinese.
        A keyword in English should generate queries in English.
    </Task>
    """
    tool_model = ChatLiteLLM(model=model_name, temperature=0).bind_tools([queries_formatter], tool_choice="required")
    output = await tool_model.ainvoke(
        [SystemMessage(content=system_instruction.format(key_word=key_word))]
        + [HumanMessage(content="Please help me to find relevant information.")]
    )
    web_results = call_search_engine(output.tool_calls[0]["args"]["queries"], True)
    source_str = web_search_deduplicate_and_format_sources(web_results, 5000, True)

    system_instruction = f"""You are an experienced copywriter who retains important information and summarizes key points and hashtags.
    The following is my relevant background knowledge. Please assist me in synthesizing the information below into a 300-500 word column.

    <knowledge>
    {source_str}
    </knowledge>

    <Task>
    Your task is to write a column based on the knowledge and summarize the important keywords into python list hashtags.
    <Task>

    <Limit>
    If the content of the information is in Chinese, please write the column in Chinese.
    If it is in English, please use English for the writing.
    </Limit>
    """
    tool_model = ChatLiteLLM(model=model_name, temperature=0).bind_tools(
        [background_knowledge_formatter], tool_choice="required"
    )
    output = await tool_model.ainvoke(
        [SystemMessage(content=system_instruction)]
        + [HumanMessage(content="Please help me to summarize knowledge into column and hashtags.")]
    )
    return output.tool_calls[0]["args"]


async def model_refine_transcription(model_name: str, transcription_text: str, background_knowledge: str | None):
    system_instruction = """
    Your task is to transform the verbatim transcript sampled from the audio model into readable content that adheres to Markdown format.
    The content in the Transcription represents the verbatim transcript sampled from the audio.
    Knowledge represents knowledge related to the speech content, which may not directly relate to the speech but can be used to correct errors in the transcript.

    <Transcription>
    {transcription}
    </Transcription>

    <Knowledge>
    {knowledge}
    </Knowledge>

    <Task>
    1. Your task is to revise the transcript into readable, fluent content in Markdown format.
    2. You must ensure that the translated content does not omit any meaningful information, and only redundant words such as interjections or pauses can be removed.
    3. If the content can be structured using a table, please help me convert it into a table, ensuring that no important information is omitted.
    4. Language
       - If the transcript is in Chinese, the output should be in Traditional Chinese.
       - If the transcript is in English, the output should be in English.
    5. Do not output in Simplified Chinese.
    </Task>

    <Guideline>
    - Please assist with formatting and adding punctuation to create readable content in Markdown format.
    - Ensure the fluency of the sentences.
    - Must ensure that the translated content does not omit any meaningful information, and only redundant words such as interjections or pauses can be removed.
    </Guideline>
    """
    model_name = "deepseek/deepseek-chat"
    tool_model = ChatLiteLLM(model=model_name, temperature=0)
    output = await tool_model.ainvoke(
        [
            SystemMessage(
                content=system_instruction.format(transcription=transcription_text, knowledge=background_knowledge)
            )
        ]
        + [HumanMessage(content="Please help me to adjust this paragraph into suitable content and format.")]
    )
    return output.content


class AudioTranscription:
    def __init__(
        self,
        files: list[str],
        target_path: str,
        audio_model_params: dict | None = None,
        llm_refine: bool = True,
        llm_model_name: str | None = "deepseek/deepseek-chat",
    ):
        self.files = files
        self.target_path = target_path
        self.audio_model = None
        if audio_model_params is None:
            audio_model_params = {
                "model": "iic/SenseVoiceSmall",
                "vad_model": "fsmn-vad",
                "vad_kwargs": {"max_single_segment_time": 30000},
                "device": "cuda:0",
            }
        self.get_audio_model(audio_model_params)

        self.llm_refine = llm_refine
        self.llm_model_name = llm_model_name

    def get_audio_model(self, params: dict):
        self.audio_model = AutoModel(**params)

    def get_audio_context(self, audio_path: str, hotword: list | None = None):
        res = self.audio_model.generate(
            input=audio_path,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=120,
            merge_vad=True,
            merge_length_s=30,
            hotword=hotword,
        )
        text = rich_transcription_postprocess(res[0]["text"])
        return text

    async def parse(self):
        if not self.llm_refine:
            audio_results = [self.get_audio_context(f) for f in self.files]
            return audio_results

        tasks = []
        for f in self.files:
            _, name_w_extension = os.path.split(f)
            name, _ = os.path.splitext(name_w_extension)
            tasks.append(get_background_knowledge(self.llm_model_name, f"介紹 {name}"))
        informations = await asyncio.gather(*tasks)

        audio_results = [
            self.get_audio_context(f, information["hashtags"]) for f, information in zip(self.files, informations)
        ]

        refine_tasks = [
            model_refine_transcription(self.llm_model_name, text, information["knowledge"])
            for text, information in zip(audio_results, informations)
        ]
        results = await asyncio.gather(*refine_tasks)
        for f, res in zip(self.files, results):
            _, name_w_extension = os.path.split(f)
            name, _ = os.path.splitext(name_w_extension)
            with open(f"{self.target_path}/{name}.md", "w") as f:
                f.write(res)

    def run_parse(self):
        asyncio.run(self.parse())
