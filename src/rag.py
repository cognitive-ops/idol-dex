"""RAG layer — FAISS index + metadata storage."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from .config import settings
from .embedder import embedder


class Document:
    """Single indexed document."""

    def __init__(self, doc_id: str, title: str, metadata: dict, content: str):
        self.doc_id = doc_id
        self.title = title
        self.metadata = metadata  # {actors, date, plot, reviews, source_url}
        self.content = content  # full text for context

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "metadata": self.metadata,
            "content": self.content,
        }


class RAGStore:
    """FAISS-backed RAG store."""

    def __init__(self, index_path: str = settings.faiss_index_path,
                 metadata_path: str = settings.metadata_path):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.documents: dict[str, Document] = {}
        self.index = None
        self._load()

    def _load(self):
        """Load index + metadata from disk."""
        if self.index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path) as f:
                data = json.load(f)
                for doc_dict in data:
                    doc = Document(**doc_dict)
                    self.documents[doc.doc_id] = doc

    def add_documents(self, docs: list[Document]) -> None:
        """Add documents to index. Builds FAISS if first time."""
        embeddings = embedder.embed_batch([doc.content for doc in docs])
        embeddings_np = np.array(embeddings, dtype=np.float32)

        if self.index is None:
            dim = embedder.get_embedding_dim()
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings_np)
        for doc in docs:
            self.documents[doc.doc_id] = doc

        self._save()

    def search(self, query: str, k: int = settings.top_k) -> list[tuple[Document, float]]:
        """Search for top-k documents most similar to query."""
        if self.index is None or not self.documents:
            return []

        query_emb = np.array([embedder.embed_text(query)], dtype=np.float32)
        distances, indices = self.index.search(query_emb, min(k, len(self.documents)))

        results = []
        for i, distance in zip(indices[0], distances[0]):
            if i == -1:  # invalid index
                continue
            doc_list = list(self.documents.values())
            if i < len(doc_list):
                doc = doc_list[i]
                results.append((doc, float(distance)))

        return results

    def _save(self):
        """Persist index + metadata to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

        with open(self.metadata_path, "w") as f:
            json.dump([doc.to_dict() for doc in self.documents.values()], f, indent=2)


rag_store = RAGStore()
