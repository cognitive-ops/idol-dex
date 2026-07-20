"""Entrypoint — run FastAPI server."""
import logging

from src.config import settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

import uvicorn

from src.api import app

if __name__ == "__main__":
    logger.debug("starting uvicorn host=%s port=%s", settings.api_host, settings.api_port)
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower()
    )
