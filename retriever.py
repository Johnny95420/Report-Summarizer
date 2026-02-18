# %%
import os
import glob
import json

import omegaconf
import torch
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger("Retriever")
logger.setLevel(logging.ERROR)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# %%
def process_date(information):
    date = information["date"]
    if date is None or date == "None":
        logger.critical(file)
        logger.critical(
            "Can not get date information by first 5000 strngs. Try to parse the date information by file name"
        )
        try:
            date = name.split("-")[-1]
            if "_" in date:
                date = date.split("_")[0]

        except Exception as e:
            logger.critical(
                "Can not get date information by file name. Set date to None"
            )
            date = "None"
    return date


def process_document(name, date, information):
    if "table" in information:
        if len(information["table"]) >= 100000:
            logger.critical(
                f"File:{file}.\n\n The length of table string longer than 100000"
            )
            return None

        metadata = {
            "path": name,
            "date": date,
            "context_heading": (
                information["context_heading"]
                if information["context_heading"]
                else "None"
            ),
            "context_paragraph": (
                information["context_paragraph"]
                if information["context_paragraph"]
                else "None"
            ),
            "summary": information["summary"],
            "table": information["table"],
        }
        doc = Document(
            information["summary"],
            metadata=metadata,
        )

    else:
        doc = Document(
            information["full_content"],
            metadata={
                "path": name,
                "content": information["full_content"],
                "date": date,
            },
        )
    return doc


# %%
import pathlib

_HERE = pathlib.Path(__file__).parent
config = omegaconf.OmegaConf.load(_HERE / "retriever_config.yaml")
files = []
if config["raw_file_path"] is not None:
    for f in config["raw_file_path"]:
        files.extend(glob.glob(f"{f}/*.*"))

    documents, table_documents = [], []
    for file in files:
        name = os.path.split(file)[-1]
        name, extension = os.path.splitext(name)
        with open(file, "rb") as f:
            information = json.load(f)

        date = process_date(information)
        doc = process_document(file, date, information)

        if doc is None:
            continue

        if "table" in doc.metadata:
            table_documents.append(doc)

        else:
            documents.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["split_chunk_size"],
        chunk_overlap=config["split_chunk_overlap"],
        separators=["\n\n\n\n", "\n\n\n", "\n\n", "\n"],
    )
    doc_splits = text_splitter.split_documents(documents) + table_documents
    embeddings = HuggingFaceEmbeddings(model_name=config["embedding_model"])
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embeddings,
    )
    torch.cuda.empty_cache()
    bm25_retriever = BM25Retriever.from_documents(doc_splits)
    bm25_retriever.k = config["top_k"]
    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            vectorstore.as_retriever(search_kwargs={"k": config["top_k"]}),
            bm25_retriever,
        ],
        weights=config["hybrid_weight"],
    )
else:
    hybrid_retriever = None
