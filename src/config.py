"""Config — env-driven settings."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """RAG chatbot config."""

    anthropic_api_key: str = ""
    model: str = "claude-opus-4-8"

    # Vector backend: "faiss" (local file-based, default for dev) or "qdrant" (docker-compose)
    vector_backend: str = "faiss"

    # FAISS
    faiss_index_path: str = "./data/faiss_index/index.faiss"
    metadata_path: str = "./data/metadata.json"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "imdb_docs"

    # Frontend
    api_base_url: str = "http://localhost:8000"

    # Web scraper (dataset file downloads)
    scraper_timeout: int = 30
    scraper_user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    # IMDb non-commercial datasets (https://datasets.imdbws.com/, free TSV dumps —
    # titles/ratings/cast/people, no plot summaries). See
    # https://developer.imdb.com/non-commercial-datasets/
    imdb_dataset_base_url: str = "https://datasets.imdbws.com"
    imdb_data_dir: str = "./data/imdb"
    imdb_max_titles: int = 300       # top-N titles by vote count to ingest
    imdb_min_votes: int = 10000      # quality floor so obscure/junk titles are skipped
    imdb_title_types: tuple[str, ...] = ("movie", "tvSeries", "tvMovie", "tvMiniSeries")
    imdb_cast_per_title: int = 5     # top-billed cast members pulled per title

    # RAG
    embedding_model: str = "all-MiniLM-L6-v2"  # sentence-transformers local model
    hf_hub_download_timeout: int = 60  # Hugging Face download timeout (seconds)
    top_k: int = 5
    max_context_length: int = 3000

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()
