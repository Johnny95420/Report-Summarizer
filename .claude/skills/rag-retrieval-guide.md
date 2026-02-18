# RAG Retrieval System

## Overview

The retrieval system (`retriever.py`) implements a hybrid search strategy combining **vector similarity** (ChromaDB + HuggingFace embeddings) and **keyword matching** (BM25) via LangChain's `EnsembleRetriever`. This provides both semantic understanding and exact-match precision.

## Architecture

```
User Query
    |
    v
EnsembleRetriever (hybrid_retriever)
    |
    +---> ChromaDB Vector Search (weight: 0.4)
    |         - HuggingFace BAAI/bge-m3 embeddings
    |         - Cosine similarity
    |         - top_k results
    |
    +---> BM25 Keyword Search (weight: 0.6)
              - Okapi BM25 algorithm
              - Token-based matching
              - top_k results
    |
    v
Merged & Deduplicated Results
    |
    v
Context Expansion (small-to-large)
    |
    v
Formatted Source String
```

## Configuration (`retriever_config.yaml`)

```yaml
raw_file_path:
  - "/path/to/preprocessed_data/"    # Directories containing JSON files
split_chunk_size: 1500               # Characters per chunk
split_chunk_overlap: 250             # Overlap between chunks
embedding_model: "BAAI/bge-m3"      # Multilingual embedding model
top_k: 5                            # Results per retriever
hybrid_weight: [0.4, 0.6]           # [vector_weight, bm25_weight]
```

## Data Pipeline

### 1. Preprocessing

Raw files (PDF, audio) are preprocessed into JSON format by `preprocess_files.py` and `Utils/pdf_processor.py`:

**Text document JSON schema:**
```json
{
  "date": "2024-01-15",
  "full_content": "The complete markdown text..."
}
```

**Table document JSON schema:**
```json
{
  "date": "2024-01-15",
  "context_heading": "Section heading",
  "context_paragraph": "Surrounding paragraph",
  "summary": "LLM-generated table summary",
  "table": "| col1 | col2 |\n| --- | --- |\n| ... |"
}
```

### 2. Document Loading (`retriever.py`)

At module import time, if `raw_file_path` is not None:

1. Glob all files from configured paths
2. Parse each JSON file
3. Extract date (fallback: parse from filename)
4. Create `Document` objects with metadata:
   - Text docs: `metadata = {path, content, date}`
   - Table docs: `metadata = {path, date, context_heading, context_paragraph, summary, table}`
5. Skip tables > 100,000 characters

### 3. Chunking

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=250,
    separators=["\n\n\n\n", "\n\n\n", "\n\n", "\n"],
)
doc_splits = text_splitter.split_documents(documents) + table_documents
```

**Note**: Table documents are NOT split - they're added whole to preserve tabular structure.

### 4. Indexing

```python
# Vector store
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vectorstore = Chroma.from_documents(doc_splits, collection_name="rag-chroma", embedding=embeddings)
torch.cuda.empty_cache()  # Free GPU memory after embedding

# BM25 index
bm25_retriever = BM25Retriever.from_documents(doc_splits)
bm25_retriever.k = top_k

# Ensemble
hybrid_retriever = EnsembleRetriever(
    retrievers=[vectorstore.as_retriever(search_kwargs={"k": top_k}), bm25_retriever],
    weights=[0.4, 0.6],
)
```

## Query Pipeline

### `search_relevance_doc()` in `report_writer.py`

```python
def search_relevance_doc(queries):
    seen = set()
    info = []
    for q in queries:
        results = hybrid_retriever.invoke(q)
        for res in results:
            if res.page_content in seen:
                continue
            seen.add(res.page_content)
            if "table" in res.metadata:
                info.append(res)                    # Tables used as-is
            else:
                expanded = track_expanded_context(   # Expand text context
                    res.metadata["content"],
                    res.page_content,
                    forward_capacity=1500,
                    backward_capacity=1000,
                )
                return_res = deepcopy(res)           # MUST deepcopy
                return_res.metadata["content"] = expanded
                info.append(return_res)
    return info
```

### Context Expansion Strategy

The "small-to-large" chunking strategy (inspired by Ilya Rice's award-winning approach):

1. **Small chunks** are used for retrieval (better precision)
2. **Large context** is reconstructed for LLM consumption (better comprehension)

```python
def track_expanded_context(original_context, critical_context, forward_capacity=10000, backward_capacity=2500):
    start_idx = original_context.find(critical_context)
    # Expand backward by backward_capacity chars to nearest \n\n boundary
    # Expand forward by forward_capacity chars to nearest \n\n boundary
    return expanded_context
```

The expansion snaps to paragraph boundaries (`\n\n`) to avoid cutting mid-sentence.

## ContentExtractor (Dynamic Retrieval)

`Utils/utils.py` contains a `ContentExtractor` class for dynamic, runtime indexing of web search results:

```python
class ContentExtractor:
    def __init__(self, temp_dir, k=3):
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        # Initialize with placeholder document
        self.vectorstore = Chroma.from_documents(docs, collection_name="temp_data", embedding=embeddings)
        self.bm25_retriever = BM25Retriever.from_documents(docs)
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[vectorstore.as_retriever(search_kwargs={"k": 3}), bm25_retriever],
            weights=[0.8, 0.2],  # Note: different weights from main retriever
        )
```

**Key differences from main retriever**:
- Weights: `[0.8, 0.2]` (vector-heavy) vs `[0.4, 0.6]` (BM25-heavy for main)
- Chunk size: 300 (vs 1500 for main)
- Chunk overlap: 50 (vs 250 for main)
- Purpose: index large web results at runtime, not preprocessed local data

**Usage flow**:
1. Large web results (>5000 chars) are saved to temp files
2. `content_extractor.update(large_files)` - chunks and indexes new files
3. `content_extractor.query(query)` - retrieves most relevant chunks with context expansion

## Result Formatting

### Local DB Results

```python
def format_search_results_with_metadata(results):
    # For table documents:
    #   Source [path]: Report Date, Context Heading, Context Paragraph, Summary, Table Content
    # For text documents:
    #   Source [path]: Report Date, Source Content
```

### Web Results

```python
def web_search_deduplicate_and_format_sources(search_response, include_raw_content):
    # Deduplicate by URL (highest score wins)
    # Format: Source [title]: URL, Most relevant content, [raw_content if included]
```

### Combined Source String

In `search_db()`, local and web results are joined:
```python
source_str = local_results + "===\n\n" + web_results
```

## PDF Processing (`Utils/pdf_processor.py`)

The `PDFProcessor` class handles PDF-to-JSON conversion:

1. **PDF to Markdown**: Uses `marker-pdf` library
2. **Metadata extraction**: LLM-based extraction of date, rating, price target using tool binding (`financial_report_metadata_extraction`)
3. **Table summarization**: LLM summarizes each table with context (`table_summarization`)
4. **Output**: Structured JSON with full content, tables (as separate entries), and metadata

## Embedding Model

**BAAI/bge-m3** (recommended):
- Multilingual: supports Traditional Chinese, English, and 100+ languages
- Dense + sparse + multi-vector retrieval capabilities
- Optimized for RAG applications
- GPU-accelerated via PyTorch/CUDA

Memory note: `torch.cuda.empty_cache()` is called after embedding to free GPU memory.
