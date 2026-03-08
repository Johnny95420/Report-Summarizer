"""Auth Source Search LangGraph subagent.

Downloads institutional reports (InvestAnchor, Yuanta) and answers research
questions via Document QA with a shared AgentDocumentReader navigator.

File strategy: two-layer directory.
  - Global cache: reader_tmp/ (persistent, used for cache hits)
  - Per-run dir:  reader_tmp/auth_run_{uuid}/ (symlinks, deleted after run)
"""

import json
import logging
import os
import pathlib
import re

import omegaconf

_HERE = pathlib.Path(__file__).parent.parent
_config = omegaconf.OmegaConf.load(_HERE / "report_config.yaml")
MODEL_NAME = _config["MODEL_NAME"]
BACKUP_MODEL_NAME = _config["BACKUP_MODEL_NAME"]
LIGHT_MODEL_NAME = _config["LIGHT_MODEL_NAME"]
BACKUP_LIGHT_MODEL_NAME = _config["BACKUP_LIGHT_MODEL_NAME"]

_retriever_cfg = omegaconf.OmegaConf.load(_HERE / "retriever_config.yaml")
_READER_TMP_DIR = str(_retriever_cfg.get("reader_tmp_dir", "reader_tmp"))

logger = logging.getLogger("AuthSourceSearch")

_NO_ANSWER_SENTINEL = "[auth_source_search: 未能從機構報告中找到相關資訊]"


# ---------------------------------------------------------------------------
# _sanitize_qa_answer
# ---------------------------------------------------------------------------
_INVALID_ANSWER_MARKERS = [
    "[Unable to extract answer from documents]",
    "[LLM failed",
    "[BUDGET EXHAUSTED",
    "[Answer submitted.",
]
_PLANNING_PATTERN = re.compile(r"^\s*\[[ x]?\]\s+\d+\.", re.MULTILINE)


def _sanitize_qa_answer(answer: str) -> str:
    """Return empty string when the answer is a Document QA fallback artifact."""
    if not answer or not answer.strip():
        return ""
    for marker in _INVALID_ANSWER_MARKERS:
        if marker in answer:
            return ""
    if _PLANNING_PATTERN.search(answer):
        return ""
    return answer


# ---------------------------------------------------------------------------
# Navigator state persistence helpers (single-run safety net)
# ---------------------------------------------------------------------------
def _save_navigator_state(navigator, path: str) -> None:
    """Persist navigator bookmarks and current path to a JSON sidecar file."""
    bookmarks_serializable = {label: [bm.path, bm.doc_name, bm.page_id] for label, bm in navigator._bookmarks.items()}
    state = {
        "bookmarks": bookmarks_serializable,
        "last_open_path": navigator._current_path,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _restore_navigator_state(navigator, path: str) -> None:
    """Restore navigator bookmarks from a JSON sidecar file."""
    if not os.path.exists(path):
        return
    from Tools.text_navigator import Bookmark

    with open(path, encoding="utf-8") as f:
        saved = json.load(f)

    navigator._bookmarks = {label: Bookmark(*values) for label, values in saved.get("bookmarks", {}).items()}
    last_path = saved.get("last_open_path")
    if last_path and os.path.exists(last_path):
        navigator.open_document(last_path)
