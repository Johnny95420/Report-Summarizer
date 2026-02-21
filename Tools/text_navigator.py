import json
import logging
import os
import pathlib

import omegaconf
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.tools import BaseTool, tool
from langchain_huggingface import HuggingFaceEmbeddings

from Tools.reader_models import BaseReaderDocument, PDFReaderDocument, SearchResult, sanitize_name

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "retriever_config.yaml"
_cfg = omegaconf.OmegaConf.load(_CONFIG_PATH)
_DEFAULT_TOP_K: int = int(_cfg.get("navigator_top_k", 5))
_DEFAULT_PERSIST_DIR: str = str(_PROJECT_ROOT / "navigator_tmp")

logger = logging.getLogger("TextNavigator")
logger.setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Module-level embedding singleton
# ---------------------------------------------------------------------------
_embedding_model: HuggingFaceEmbeddings | None = None
_embedding_model_name: str = ""


def get_embedding_model(model_name: str = "Qwen/Qwen3-Embedding-0.6B") -> HuggingFaceEmbeddings:
    global _embedding_model, _embedding_model_name
    if _embedding_model is None or _embedding_model_name != model_name:
        _embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        _embedding_model_name = model_name
    return _embedding_model


# ---------------------------------------------------------------------------
# AgentDocumentReader
# ---------------------------------------------------------------------------
class AgentDocumentReader:
    """Long-lived navigator that reads pre-processed ReaderDocument JSON files.

    Typical agent workflow:
        reader = AgentDocumentReader()          # create once
        reader.open_document("doc1.json")       # load file 1
        reader.semantic_search("some query")
        reader.go_to_page(5)
        reader.update_bookmark("key_section")
        reader.open_document("doc2.json")       # switch file (state reset)
    """

    def __init__(
        self,
        persist_dir: str = _DEFAULT_PERSIST_DIR,
        embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    ):
        self._persist_dir = persist_dir
        self._embedding_model_name = embedding_model_name
        self._document: BaseReaderDocument | None = None
        self._vectorstore: Chroma | None = None
        self._bm25: BM25Retriever | None = None
        self._current_page: int = 0
        self._current_path: str | None = None
        # Bookmarks survive document switches: {label: (path, doc_name, page_id)}
        self._bookmarks: dict[str, tuple[str, str, int]] = {}

    # ------------------------------------------------------------------
    # Document lifecycle
    # ------------------------------------------------------------------

    def open_document(self, path: str) -> str:
        """Load a pre-processed ReaderDocument JSON from *path*.

        Resets navigation cursor but preserves all bookmarks across document switches.
        Returns a status summary string. If *path* is already open, returns current
        state without reloading (dedup check).
        """
        import json

        # Dedup: skip reload if same document is already open
        if self._current_path == path and self._document is not None:
            return (
                f"[Already open: {self._document.name} | {len(self._document.pages)} pages"
                f" | page {self._current_page}]"
            )

        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        if "highlights" in raw and "tables" in raw:
            doc = PDFReaderDocument.load(path)
        else:
            doc = BaseReaderDocument.load(path)

        self._document = doc
        self._current_page = 0
        self._current_path = path
        # NOTE: bookmarks are intentionally NOT cleared here — they persist across documents

        # Build or load vectorstore
        os.makedirs(self._persist_dir, exist_ok=True)
        persist_path = os.path.join(self._persist_dir, sanitize_name(doc.name))

        embeddings = get_embedding_model(self._embedding_model_name)
        if os.path.exists(persist_path):
            self._vectorstore = Chroma(
                persist_directory=persist_path,
                embedding_function=embeddings,
            )
        else:
            self._vectorstore = Chroma.from_documents(
                documents=doc.pages,
                embedding=embeddings,
                persist_directory=persist_path,
            )

        # Build BM25 index
        self._bm25 = BM25Retriever.from_documents(doc.pages)

        return f"Opened: {doc.name} | date: {doc.date} | {len(doc.pages)} pages"

    def close_document(self) -> None:
        """Clear all state and free resources."""
        self._document = None
        self._vectorstore = None
        self._bm25 = None
        self._current_page = 0
        self._current_path = None
        self._bookmarks = {}

    # ------------------------------------------------------------------
    # Status / metadata
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        self._require_open()
        return {
            "current_page": self._current_page,
            "total_pages": len(self._document.pages),
        }

    def get_metadata(self) -> dict:
        self._require_open()
        doc = self._document
        word_count = sum(len(p.page_content.split()) for p in doc.pages)
        return {
            "name": doc.name,
            "date": doc.date,
            "word_count": word_count,
            "has_outline": bool(doc.outlines),
        }

    def get_outline(self) -> list[dict]:
        self._require_open()
        return self._document.outlines

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def peek_page(self, page: int) -> str:
        """Return first 500 chars of *page* without moving the cursor."""
        self._require_open()
        self._validate_page(page)
        return self._document.pages[page].page_content[:500]

    def go_to_page(self, page: int) -> str:
        """Move cursor to *page* and return full page content."""
        self._require_open()
        self._validate_page(page)
        self._current_page = page
        return self._document.pages[page].page_content

    def next_page(self) -> str:
        """Advance cursor by one and return full page content."""
        self._require_open()
        next_idx = self._current_page + 1
        self._validate_page(next_idx)
        self._current_page = next_idx
        return self._document.pages[next_idx].page_content

    def prev_page(self) -> str:
        """Move cursor back one and return full page content."""
        self._require_open()
        prev_idx = self._current_page - 1
        self._validate_page(prev_idx)
        self._current_page = prev_idx
        return self._document.pages[prev_idx].page_content

    # ------------------------------------------------------------------
    # Bookmarks
    # ------------------------------------------------------------------

    def update_bookmark(self, label: str) -> str:
        """Save current page under *label*. Bookmarks persist across document switches."""
        self._require_open()
        self._bookmarks[label] = (self._current_path, self._document.name, self._current_page)
        return f"Bookmarked page {self._current_page} of '{self._document.name}' as '{label}'"

    def show_bookmarks(self) -> dict:
        """Return all saved bookmarks as {label: {doc, page_id}}."""
        return {
            label: {"doc": name, "page_id": page_id}
            for label, (path, name, page_id) in self._bookmarks.items()
        }

    def go_to_bookmark(self, label: str) -> str:
        """Move cursor to the page saved under *label*, auto-switching document if needed."""
        if label not in self._bookmarks:
            raise KeyError(f"Bookmark '{label}' not found. Available: {list(self._bookmarks)}")
        path, _name, page_id = self._bookmarks[label]
        if path != self._current_path:
            self.open_document(path)
        self._current_page = page_id
        return self._document.pages[page_id].page_content

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def semantic_search(self, query: str, k: int = _DEFAULT_TOP_K) -> list[SearchResult]:
        """Vector similarity search. Returns up to *k* SearchResult objects."""
        self._require_open()
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")
            results = self._vectorstore.similarity_search_with_relevance_scores(query, k=k)
        return [self._make_result(doc, score) for doc, score in results]

    def keyword_search(self, query: str, k: int = _DEFAULT_TOP_K) -> list[SearchResult]:
        """BM25 keyword search. Returns up to *k* SearchResult objects. score is None (BM25 has no relevance score)."""
        self._require_open()
        self._bm25.k = k
        docs = self._bm25.invoke(query)
        return [self._make_result(doc, score=None) for doc in docs]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_open(self) -> None:
        if self._document is None:
            raise RuntimeError("No document is open. Call open_document(path) first.")

    def _validate_page(self, page: int) -> None:
        total = len(self._document.pages)
        if not (0 <= page < total):
            raise IndexError(f"Page {page} out of range [0, {total - 1}].")

    # TODO: if search results is a table. the page_content is table summary the original table is in the metadata. If agent do not open this page can not read the original table  
    def _make_result(self, doc, score: float | None) -> SearchResult:
        page_id = doc.metadata.get("page_id", 0)
        page_content = self._document.pages[page_id].page_content if page_id < len(self._document.pages) else ""
        return SearchResult(
            page_id=page_id,
            score=score,
            page_preview=page_content[:250],
        )

    # ------------------------------------------------------------------
    # LangChain / LangGraph tool binding
    # ------------------------------------------------------------------

    def get_tools(self) -> list[BaseTool]:
        """Return a list of LangChain tools bound to this reader instance.

        Usage::
            reader = AgentDocumentReader()
            tools = reader.get_tools()
            llm_with_tools = llm.bind_tools(tools)
            tool_node = ToolNode(tools)
        """
        _reader = self

        @tool
        def open_document(path: str) -> str:
            """Load a pre-processed ReaderDocument JSON file and prepare it for reading.

            Input:  path — absolute path to a JSON file inside reader_tmp/.
            Output: summary string "Opened: <name> | date: <date> | <N> pages",
                    or "[Already open: ...]" if the same file is already loaded (no reload).

            Agent Instruction:
            - Call this before any other tool; all operations fail if no document is open.
            - ONLY ONE DOCUMENT CAN BE ACTIVE AT A TIME. Opening a second document immediately
              replaces the first — you cannot hold two documents open simultaneously. Do NOT
              plan to open multiple documents in parallel; open them sequentially one at a time.
            - Calling it again on a different path switches documents — cursor resets to page 0.
            - Bookmarks are NOT cleared on document switch; they persist across all documents
              in the current session. Use show_bookmarks to check what was already marked in
              previously opened documents before starting a new search.
            - Calling with the same path that is already open is a no-op (returns current state).

            Args:
                path: Absolute path to a ReaderDocument JSON file in reader_tmp/.
            """
            return _reader.open_document(path)

        @tool
        def close_document() -> str:
            """Close the currently open document and release its vectorstore and BM25 index from memory.

            Input:  none.
            Output: confirmation string "Document closed.".

            Agent Instruction:
            - Call this after finishing with a document when processing many files in one task,
              to prevent memory from accumulating across documents.
            - After closing, open_document must be called again before any other tool.
            """
            _reader.close_document()
            return "Document closed."

        @tool
        def get_status() -> str:
            """Return the current navigation state as a JSON string.

            Input:  none.
            Output: JSON with keys: current_page (int), total_pages (int).

            Agent Instruction:
            - Call this whenever you are unsure of your current position to re-orient yourself.
            - Use total_pages before navigation to plan your reading range and avoid out-of-bounds errors.
            - To see saved bookmarks, call show_bookmarks separately.
            """
            return json.dumps(_reader.get_status(), ensure_ascii=False)

        @tool
        def get_metadata() -> str:
            """Return document metadata as a JSON string.

            Input:  none.
            Output: JSON with keys: name (str), date (str), word_count (int), has_outline (bool).

            Agent Instruction:
            - Call right after open_document to confirm the report date and size before committing
              to a full read; discard documents that are out of date or irrelevant.
            - If has_outline is true, call get_outline next to map the document structure
              before navigating.
            """
            return json.dumps(_reader.get_metadata(), ensure_ascii=False)

        @tool
        def get_outline() -> str:
            """Return the document outline as a JSON array for rapid structural overview.

            Input:  none.
            Output: JSON array where each element is one of:
                    - text chunk:  {page_id, headers: [{header_level, title}, ...]}
                    - table chunk: {page_id, table_summary}

            Agent Instruction:
            - Always call this before broad search or sequential navigation; use the outline to
              identify target sections, then jump directly with go_to_page instead of scanning
              page by page.
            - Header titles are truncated (suffix "...[truncated]") — use go_to_page for full text.
            - table_summary is a brief digest; use go_to_page to read the full table data.
            """
            return json.dumps(_reader.get_outline(), ensure_ascii=False)

        @tool
        def peek_page(page: int) -> str:
            """Preview the first 500 characters of a page without moving the cursor.

            Input:  page — zero-based page index.
            Output: first 500 characters of that page as a plain string.

            Agent Instruction:
            - Use after a search to quickly assess relevance before committing to go_to_page.
            - Safe to call freely — the cursor does not move, so current reading position is preserved.
            - If 500 characters are sufficient to answer the question, skip go_to_page entirely.

            Args:
                page: Zero-based page index (range: 0 to total_pages-1).
            """
            return _reader.peek_page(page)

        @tool
        def go_to_page(page: int) -> str:
            """Move the cursor to a specific page and return its full content.

            Input:  page — zero-based page index.
            Output: full text content of that page as a string.

            Agent Instruction:
            - The cursor stays on this page after the call; subsequent next_page / prev_page
              use it as the new reference point.
            - Pair with search results: use page_id from semantic_search or keyword_search,
              then call this tool to read the full page.
            - Check total_pages via get_status first; an out-of-range index raises IndexError.

            Args:
                page: Zero-based page index (range: 0 to total_pages-1).
            """
            return _reader.go_to_page(page)

        @tool
        def next_page() -> str:
            """Advance the cursor by one page and return its full content.

            Input:  none.
            Output: full text content of the next page as a string.

            Agent Instruction:
            - Use for linear reading after locating a relevant section, e.g. reading several
              consecutive pages of data or narrative.
            - Raises IndexError if already on the last page; check current_page < total_pages-1
              via get_status before calling.
            """
            return _reader.next_page()

        @tool
        def prev_page() -> str:
            """Move the cursor back one page and return its full content.

            Input:  none.
            Output: full text content of the previous page as a string.

            Agent Instruction:
            - Use when the current page references context from the preceding page.
            - Raises IndexError if already on page 0; check current_page > 0 via get_status
              before calling.
            """
            return _reader.prev_page()

        @tool
        def semantic_search(query: str, k: int = _DEFAULT_TOP_K) -> str:
            """Search the document by vector similarity to find pages most semantically related to the query.

            Input:  query — natural language query string;
                    k     — number of results to return (default set in config).
            Output: JSON array of results, each with:
                    - page_id:      page index (pass directly to go_to_page or peek_page)
                    - score:        relevance score 0–1 (higher is more relevant)
                    - page_preview: first 250 characters of that page

            Agent Instruction:
            - Prefer this tool for conceptual or intent-based queries, e.g.
              "impact of TSMC US fab expansion on supply chain".
            - Results with score below 0.4 are weakly relevant; cross-check with keyword_search.
            - After receiving results, use peek_page on the top page_ids to confirm relevance
              before calling go_to_page for a full read.
            - Query language should match the document language (Traditional Chinese for this system).

            Args:
                query: Natural language search query.
                k: Number of results to return (default from config).
            """
            results = _reader.semantic_search(query, k)
            return json.dumps([r.model_dump() for r in results], ensure_ascii=False)

        @tool
        def keyword_search(query: str, k: int = _DEFAULT_TOP_K) -> str:
            """Search the document by BM25 keyword matching to find pages containing specific terms.

            Input:  query — keyword query string;
                    k     — number of results to return (default set in config).
            Output: JSON array of results, each with:
                    - page_id:      page index (pass directly to go_to_page or peek_page)
                    - score:        null (BM25 does not produce a normalised relevance score)
                    - page_preview: first 250 characters of that page

            Agent Instruction:
            - Best for exact proper nouns: company names, ticker symbols, technical terms
              (e.g. "帆宣", "HBM", "N3E").
            - score is always null and carries no ranking meaning beyond BM25 term frequency.
            - Use alongside semantic_search for full coverage: semantic_search captures intent,
              keyword_search captures exact terminology.

            Args:
                query: Keyword search query.
                k: Number of results to return (default from config).
            """
            results = _reader.keyword_search(query, k)
            return json.dumps([r.model_dump() for r in results], ensure_ascii=False)

        @tool
        def update_bookmark(label: str) -> str:
            """Save the current cursor page under a descriptive label for instant retrieval later.

            Input:  label — descriptive bookmark name string.
            Output: confirmation string "Bookmarked page <N> of '<doc>' as '<label>'".

            Agent Instruction:
            - Bookmark immediately after reading a page that answers part of the question
              (e.g. a key table, a target price, a risk section). The label IS your memory:
              a good label like 'nan_dian_eps_2026' lets you jump back without re-searching.
            - Bookmarks survive document switches — you can open another document and still
              return to a page in a previous document via go_to_bookmark.
            - Calling with the same label overwrites the previous entry for that label.
            - Use descriptive labels: 'tsmc_capex_table', 'risk_factors', 'target_price_340'.

            Args:
                label: Descriptive bookmark name (e.g. 'key_table', 'main_conclusion').
            """
            return _reader.update_bookmark(label)

        @tool
        def show_bookmarks() -> str:
            """Return all saved bookmarks as a JSON object {label: {doc, page_id}}.

            Input:  none.
            Output: JSON object mapping label strings to {doc: <document name>, page_id: <int>}.
                    Returns {} if no bookmarks have been set.
                    Bookmarks span all documents opened in the current session.

            Agent Instruction:
            - Call this at the START of each new sub-task or before any search to check
              whether a relevant page has already been bookmarked in any open document.
              If a matching label exists, use go_to_bookmark instead of searching again.
            - The label name is your key: if you saved 'target_price' earlier, check here
              before re-reading sections to find the same data.
            - Think of show_bookmarks as your personal index of already-confirmed findings.
            """
            return json.dumps(_reader.show_bookmarks(), ensure_ascii=False)

        @tool
        def go_to_bookmark(label: str) -> str:
            """Jump to the page saved under a bookmark label and return its full content.

            Input:  label — bookmark label previously saved with update_bookmark.
            Output: full text content of the bookmarked page as a string.
                    If the bookmark belongs to a different document than the one currently
                    open, that document is automatically loaded before jumping.

            Agent Instruction:
            - Preferred workflow: show_bookmarks → identify matching label → go_to_bookmark.
              This avoids redundant searches when you already confirmed the page in a
              previous iteration.
            - Use this to cross-reference pages across documents: bookmark a key metric in
              doc A, switch to doc B, then go_to_bookmark to pull back that metric for comparison.
            - Call show_bookmarks first if unsure of the exact label — a wrong label raises KeyError.

            Args:
                label: Bookmark label previously saved with update_bookmark.
            """
            return _reader.go_to_bookmark(label)

        return [
            open_document,
            get_status,
            get_metadata,
            get_outline,
            peek_page,
            go_to_page,
            next_page,
            prev_page,
            semantic_search,
            keyword_search,
            update_bookmark,
            show_bookmarks,
            go_to_bookmark,
        ]
