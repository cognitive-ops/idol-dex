"""Embedder — local sentence-transformers (no API calls)."""
import logging
import os
from sentence_transformers import SentenceTransformer

from .config import settings

logger = logging.getLogger(__name__)

# Set Hugging Face download timeout (env var must be set BEFORE importing HF libs)
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(settings.hf_hub_download_timeout)


class Embedder:
    """Embed text using sentence-transformers."""

    def __init__(self, model_name: str = settings.embedding_model):
        logger.info("loading embedding model: %s", model_name)
        logger.debug("(may take a few minutes on first run while downloading)")
        self.model = SentenceTransformer(model_name)
        logger.info("embedding model loaded: %s", model_name)

    def embed_text(self, text: str) -> list[float]:
        """Embed single text string."""
        logger.debug("embed_text: %d chars", len(text))
        return self.model.encode(text, convert_to_tensor=False).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed batch of texts."""
        logger.debug("embed_batch: %d texts", len(texts))
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def get_embedding_dim(self) -> int:
        """Embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


embedder = Embedder()
