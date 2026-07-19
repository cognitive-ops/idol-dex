"""Config — env-driven settings."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """RAG chatbot config."""

    anthropic_api_key: str = ""
    model: str = "claude-3-5-sonnet-20241022"

    # FAISS
    faiss_index_path: str = "./data/faiss_index/index.faiss"
    metadata_path: str = "./data/metadata.json"

    # Web scraper
    scraper_timeout: int = 30
    scraper_user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    # RAG
    embedding_model: str = "all-MiniLM-L6-v2"  # sentence-transformers local model
    hf_hub_download_timeout: int = 60  # Hugging Face download timeout (seconds)
    top_k: int = 5
    max_context_length: int = 3000

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"


settings = Settings()
