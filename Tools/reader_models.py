import hashlib
import json
import re

from langchain_core.documents import Document
from pydantic import BaseModel

_UNSAFE_CHARS = re.compile(r'[/\\:*?"<>|\s]+')


def sanitize_name(name: str) -> str:
    """Replace path-unsafe chars with underscores; collapse consecutive underscores.
    Truncate to 200 chars using first-150 + md5 suffix if too long.
    """
    sanitized = _UNSAFE_CHARS.sub("_", name).strip("_")
    if len(sanitized) > 200:
        digest = hashlib.md5(sanitized.encode()).hexdigest()[:8]
        sanitized = sanitized[:150] + "_" + digest
    return sanitized


class SearchResult(BaseModel):
    page_id: int
    score: float | None   # None for keyword search (BM25 has no relevance score)
    page_preview: str     # first 250 chars of that page

    class Config:
        arbitrary_types_allowed = True


class BaseReaderDocument(BaseModel):
    date: str | None
    name: str
    outlines: list[dict]
    pages: list[Document]

    class Config:
        arbitrary_types_allowed = True

    def save(self, path: str) -> None:
        """Serialize to JSON — Document objects serialized via model_dump()."""
        data = self.model_dump()
        data["pages"] = [p.model_dump() for p in self.pages]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BaseReaderDocument":
        """Load from JSON, reconstructing Document objects from dicts."""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            data["pages"] = [Document(**d) for d in data["pages"]]
            return cls(**data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ValueError(f"Failed to load '{path}': {e}. Try deleting the cached file and re-preprocessing.") from e


class PDFReaderDocument(BaseReaderDocument):
    highlights: str
    tables: list[Document]

    def save(self, path: str) -> None:
        data = self.model_dump()
        data["pages"] = [p.model_dump() for p in self.pages]
        data["tables"] = [t.model_dump() for t in self.tables]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "PDFReaderDocument":
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            data["pages"] = [Document(**d) for d in data["pages"]]
            data["tables"] = [Document(**d) for d in data["tables"]]
            return cls(**data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ValueError(f"Failed to load '{path}': {e}. Try deleting the cached file and re-preprocessing.") from e
