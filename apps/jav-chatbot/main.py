"""Entrypoint — run FastAPI server."""
import uvicorn

from src.api import app
from src.config import settings

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info"
    )
