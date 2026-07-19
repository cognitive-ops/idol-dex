"""Embedder — local sentence-transformers (no API calls)."""
from sentence_transformers import SentenceTransformer

from .config import settings


class Embedder:
    """Embed text using sentence-transformers."""

    def __init__(self, model_name: str = settings.embedding_model):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        """Embed single text string."""
        return self.model.encode(text, convert_to_tensor=False).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed batch of texts."""
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def get_embedding_dim(self) -> int:
        """Embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


embedder = Embedder()
