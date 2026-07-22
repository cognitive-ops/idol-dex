"""Agentic RAG — Claude breaks the question into focused sub-queries, calls the
search tool once per sub-query (deciding whether more searches are needed from
each result), then synthesizes a final answer from everything it gathered."""
from __future__ import annotations

import logging

import anthropic
from anthropic import beta_tool

from .config import settings
from .rag import Document, rag_store

logger = logging.getLogger(__name__)

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

SYSTEM_PROMPT = (
    "You are a movie/TV research assistant backed by an IMDb dataset search tool. "
    "Break the user's question into the distinct pieces of information it actually "
    "needs, then call search_imdb once per piece — e.g. one call per title or person "
    "being compared, or one call per part of a multi-part question. Issue calls one "
    "at a time and read each result before deciding whether another search is "
    "needed; don't guess at facts the tool can look up. Once you have enough "
    "results, answer the original question, weighing everything you've gathered "
    "and noting clearly if some part of the question couldn't be answered from "
    "the data."
)


def _build_source(doc: Document, distance: float) -> dict:
    src = {
        "title": doc.title,
        "url": doc.metadata.get("url", ""),
        "similarity": 1 / (1 + distance),
    }
    if doc.metadata.get("type") == "person":
        src.update({
            "name": doc.metadata.get("name", ""),
            "birth_year": doc.metadata.get("birth_year", ""),
            "death_year": doc.metadata.get("death_year", ""),
            "professions": doc.metadata.get("professions", ""),
            "known_for": doc.metadata.get("known_for", []),
        })
    else:
        src.update({
            "actors": doc.metadata.get("actors", ""),
            "year": doc.metadata.get("year", ""),
            "genres": doc.metadata.get("genres", ""),
            "rating": doc.metadata.get("rating", ""),
        })
    return src


def answer(query: str) -> dict:
    """Run the decompose -> search -> synthesize loop for one user question."""
    sources: dict[str, dict] = {}
    sub_queries: list[str] = []

    @beta_tool
    def search_imdb(sub_query: str, doc_type: str = "") -> str:
        """Search the IMDb dataset for titles or people relevant to one focused
        sub-question. Call this once per distinct piece of information you need.

        Args:
            sub_query: A single, focused search query — not the whole original question.
            doc_type: Optional filter, "title" or "person". Leave empty to search both.
        """
        sub_queries.append(sub_query)
        filters = {"type": doc_type} if doc_type else None
        results = rag_store.search(sub_query, k=settings.top_k, filters=filters)
        if not results:
            return "No matching results in the IMDb dataset."

        lines = []
        for doc, distance in results:
            sources.setdefault(doc.doc_id, _build_source(doc, distance))
            lines.append(f"Source: {doc.title}\n{doc.content}")
        return "\n\n---\n\n".join(lines)[:settings.max_context_length]

    logger.debug("agent.answer: query=%r", query)
    runner = client.beta.messages.tool_runner(
        model=settings.model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=[search_imdb],
        messages=[{"role": "user", "content": query}],
        max_iterations=6,
    )

    final_text = ""
    for message in runner:
        text_blocks = [block.text for block in message.content if block.type == "text"]
        if text_blocks:
            final_text = "\n".join(text_blocks)

    logger.debug(
        "agent.answer: sub_queries=%d sources=%d answer_len=%d",
        len(sub_queries), len(sources), len(final_text),
    )
    return {
        "answer": final_text or "I couldn't find an answer in the IMDb dataset.",
        "sources": list(sources.values()),
        "sub_queries": sub_queries,
    }
