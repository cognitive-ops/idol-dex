"""RAG layer — vector store abstraction. Backend picked via settings.vector_backend:
"faiss" (local index + metadata.json file, default for dev) or "qdrant" (docker-compose)."""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import faiss
import numpy as np

from .config import settings
from .embedder import embedder

logger = logging.getLogger(__name__)


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
            logger.debug("loading index=%s metadata=%s", self.index_path, self.metadata_path)
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path) as f:
                data = json.load(f)
                for doc_dict in data:
                    doc = Document(**doc_dict)
                    self.documents[doc.doc_id] = doc
            logger.info("loaded %d documents, index size=%d", len(self.documents), self.index.ntotal)
        else:
            logger.debug("no existing index/metadata found at %s / %s", self.index_path, self.metadata_path)

    def add_documents(self, docs: list[Document]) -> None:
        """Add documents to index. Builds FAISS if first time."""
        logger.debug("add_documents: embedding %d docs", len(docs))
        embeddings = embedder.embed_batch([doc.content for doc in docs])
        embeddings_np = np.array(embeddings, dtype=np.float32)

        if self.index is None:
            dim = embedder.get_embedding_dim()
            logger.debug("creating new FAISS IndexFlatL2 dim=%d", dim)
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings_np)
        for doc in docs:
            self.documents[doc.doc_id] = doc

        logger.info("added %d docs, index size now=%d", len(docs), self.index.ntotal)
        self._save()

    def search(self, query: str, k: int = settings.top_k) -> list[tuple[Document, float]]:
        """Search for top-k documents most similar to query."""
        logger.debug("search: query=%r k=%d", query, k)
        if self.index is None or not self.documents:
            logger.debug("search: empty index, returning no results")
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

        logger.debug("search: returned %d results", len(results))
        return results

    def _save(self):
        """Persist index + metadata to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

        with open(self.metadata_path, "w") as f:
            json.dump([doc.to_dict() for doc in self.documents.values()], f, indent=2)
        logger.debug("saved index=%s metadata=%s", self.index_path, self.metadata_path)


class QdrantStore:
    """Qdrant-backed RAG store — vectors + payload (metadata) live in Qdrant, no local files."""

    def __init__(self, url: str = settings.qdrant_url, collection: str = settings.qdrant_collection):
        from qdrant_client import QdrantClient

        self.collection = collection
        self.client = QdrantClient(url=url)
        logger.debug("QdrantStore init: url=%s collection=%s", url, collection)
        self._ensure_collection()

    def _ensure_collection(self):
        from qdrant_client.models import Distance, VectorParams

        if not self.client.collection_exists(self.collection):
            dim = embedder.get_embedding_dim()
            logger.info("creating qdrant collection=%s dim=%d", self.collection, dim)
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        else:
            logger.debug("qdrant collection=%s already exists", self.collection)

    @staticmethod
    def _point_id(doc_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))

    def add_documents(self, docs: list[Document]) -> None:
        """Upsert documents as points (vector + full payload) into Qdrant."""
        from qdrant_client.models import PointStruct

        logger.debug("add_documents: embedding %d docs", len(docs))
        embeddings = embedder.embed_batch([doc.content for doc in docs])

        points = [
            PointStruct(id=self._point_id(doc.doc_id), vector=emb, payload=doc.to_dict())
            for doc, emb in zip(docs, embeddings)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        logger.info("upserted %d docs into qdrant collection=%s", len(docs), self.collection)

    def search(self, query: str, k: int = settings.top_k) -> list[tuple[Document, float]]:
        """Search for top-k documents most similar to query."""
        logger.debug("search: query=%r k=%d", query, k)
        query_emb = embedder.embed_text(query)
        hits = self.client.search(collection_name=self.collection, query_vector=query_emb, limit=k)

        results = []
        for hit in hits:
            payload = hit.payload
            doc = Document(payload["doc_id"], payload["title"], payload["metadata"], payload["content"])
            distance = 1.0 - hit.score  # cosine score -> distance-like, lower = more relevant
            results.append((doc, distance))

        logger.debug("search: returned %d results", len(results))
        return results


def _build_store():
    if settings.vector_backend == "qdrant":
        logger.info("vector backend: qdrant (%s)", settings.qdrant_url)
        return QdrantStore()
    logger.info("vector backend: faiss (%s)", settings.faiss_index_path)
    return RAGStore()


rag_store = _build_store()
