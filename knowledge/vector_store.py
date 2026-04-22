"""JSON-backed vector store for transition knowledge and user feedback.

Uses sklearn TF-IDF + cosine similarity instead of ChromaDB to avoid
native dependency issues.  The knowledge base is small enough (~dozens of
entries) that this is instant.

Stores two kinds of entries:
  - Rules:    static transition knowledge seeded at init
  - Feedback: user ratings after each mix, accumulated over time
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

logger = logging.getLogger("dj.knowledge")

_PERSIST_PATH = Path("data/knowledge.json")


class TransitionKnowledgeBase:
    """TF-IDF vector store backed by a single JSON file."""

    def __init__(self, persist_path: Path | str = _PERSIST_PATH) -> None:
        self._path = Path(persist_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, dict] = {}  # id -> {text, metadata}
        if self._path.exists():
            self._entries = json.loads(self._path.read_text(encoding="utf-8"))
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self._dirty = False  # tracks whether index needs rebuild

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        self._path.write_text(
            json.dumps(self._entries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _rebuild_index(self) -> None:
        """Rebuild TF-IDF matrix from all stored documents."""
        if not self._entries:
            self._vectorizer = None
            self._tfidf_matrix = None
            return
        docs = [e["text"] for e in self._entries.values()]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._tfidf_matrix = self._vectorizer.fit_transform(docs)
        self._dirty = False

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_rule(self, text: str, metadata: dict | None = None) -> None:
        """Upsert a transition rule (idempotent — same text won't duplicate)."""
        doc_id = "rule_" + hashlib.md5(text.encode()).hexdigest()[:10]
        self._entries[doc_id] = {
            "text": text,
            "metadata": {"type": "rule", **(metadata or {})},
        }
        self._dirty = True
        self._save()

    def add_feedback(
        self,
        text: str,
        strategy: str,
        rating: int,
        metadata: dict | None = None,
    ) -> None:
        """Store user feedback about a specific transition."""
        meta = {
            "type": "feedback",
            "strategy": strategy,
            "rating": rating,
            "timestamp": time.time(),
            **(metadata or {}),
        }
        doc_id = f"fb_{hashlib.md5((text + str(time.time())).encode()).hexdigest()[:10]}"
        self._entries[doc_id] = {"text": text, "metadata": meta}
        self._dirty = True
        self._save()
        logger.info("Stored feedback: %s, %d/5", strategy, rating)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(self, query_text: str, n_results: int = 5) -> dict:
        """Retrieve the most relevant knowledge entries for a query.

        Returns a dict matching ChromaDB's query result shape so the
        rag_decider code doesn't need to change:
            {documents: [[...]], metadatas: [[...]], distances: [[...]]}
        """
        if not self._entries:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        if self._dirty or self._vectorizer is None:
            self._rebuild_index()

        query_vec = self._vectorizer.transform([query_text])
        dists = cosine_distances(query_vec, self._tfidf_matrix)[0]  # (n_docs,)

        n = min(n_results, len(self._entries))
        top_indices = np.argsort(dists)[:n]

        ids = list(self._entries.keys())
        documents = []
        metadatas = []
        distances = []
        for idx in top_indices:
            entry = self._entries[ids[idx]]
            documents.append(entry["text"])
            metadatas.append(entry["metadata"])
            distances.append(float(dists[idx]))

        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def count(self) -> int:
        return len(self._entries)
