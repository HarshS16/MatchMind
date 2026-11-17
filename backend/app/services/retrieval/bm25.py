"""BM25 lexical baseline retrieval over resume text."""

import math
import re
import uuid
from collections import Counter

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """In-memory BM25 index. Rebuilt on demand from candidate texts."""

    def __init__(self) -> None:
        self._doc_ids: list[uuid.UUID] = []
        self._texts: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._built = False

    def build(self, doc_ids: list[uuid.UUID], texts: list[str]) -> None:
        self._doc_ids = list(doc_ids)
        self._texts = texts
        tokenized = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        self._built = True

    def search(self, query: str, top_k: int = 20) -> list[tuple[uuid.UUID, float]]:
        if not self._built or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._doc_ids[i], float(scores[i])) for i in order if scores[i] > 0]

    @property
    def size(self) -> int:
        return len(self._doc_ids)
