"""FastAPI app — chat endpoint with RAG + Claude."""
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import anthropic

from . import agent
from .config import settings
from .rag import rag_store
from .imdb_data import ingest_imdb_data, ingest_people_only

logger = logging.getLogger(__name__)

app = FastAPI(title="IMDb RAG Chatbot", version="0.1.0")
client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


class AgentChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    sub_queries: list[str]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest():
    """One-time: load IMDb dataset (top titles by votes + cast) and index into FAISS."""
    logger.debug("POST /ingest")
    count = await ingest_imdb_data()
    logger.info("/ingest done: indexed=%d", count)
    return {"indexed": count, "index_path": settings.faiss_index_path}


@app.post("/ingest/people")
async def ingest_people():
    """Ingest actor/actress profiles only (name, birth/death year, professions, known-for)."""
    logger.debug("POST /ingest/people")
    count = await ingest_people_only()
    logger.info("/ingest/people done: indexed=%d", count)
    return {"indexed": count, "index_path": settings.faiss_index_path}


@app.post("/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    """Chat endpoint with RAG."""
    logger.debug("POST /chat query=%r", req.query)
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query required")

    # Retrieve relevant documents
    search_results = rag_store.search(req.query, k=settings.top_k)
    logger.debug("/chat: %d search results", len(search_results))
    if not search_results:
        return ChatResponse(
            answer="No relevant information found in the database.",
            sources=[]
        )

    # Format context
    context_parts = []
    sources = []
    for doc, distance in search_results:
        context_parts.append(f"Source: {doc.title}\n{doc.content}")
        src = {
            "title": doc.title,
            "url": doc.metadata.get("url", ""),
            "similarity": 1 / (1 + distance)  # convert L2 distance to similarity
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
        sources.append(src)

    context = "\n\n---\n\n".join(context_parts[:settings.top_k])
    context = context[:settings.max_context_length]

    # Query Claude with context
    system_prompt = (
        "You are a helpful movie/TV information assistant backed by IMDb's non-commercial "
        "dataset. Answer user questions based on the provided context — this includes both "
        "titles (name, year, genres, rating, cast) and people (actor/actress name, birth/death "
        "year, professions, known-for titles). Be factual and helpful. "
        "If information is not in the context, say so clearly."
    )

    user_message = f"""Context from IMDb dataset:

{context}

---

User question: {req.query}

Please answer based on the context above. If the question cannot be answered from the context, say so."""

    logger.debug("/chat: calling Claude model=%s context_len=%d", settings.model, len(context))
    response = client.messages.create(
        model=settings.model,
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    answer = response.content[0].text
    logger.debug("/chat: answer_len=%d sources=%d", len(answer), len(sources))

    return ChatResponse(answer=answer, sources=sources)


@app.post("/chat/agent")
async def chat_agent(req: ChatRequest) -> AgentChatResponse:
    """Agentic RAG: Claude decomposes the question into sub-queries, searches
    the IMDb dataset once per sub-query, and synthesizes a final answer from
    everything it finds. Slower and pricier than /chat — use for multi-part
    or comparison questions (e.g. "compare X and Y's careers")."""
    logger.debug("POST /chat/agent query=%r", req.query)
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query required")

    result = agent.answer(req.query)
    logger.debug(
        "/chat/agent: sub_queries=%d sources=%d",
        len(result["sub_queries"]), len(result["sources"]),
    )
    return AgentChatResponse(**result)


@app.get("/search")
async def search(q: str, type: str | None = None, actors: str | None = None):
    """Direct metadata search (no Claude). Optional `type`/`actors` narrow via
    the Qdrant payload index before the vector search runs."""
    logger.debug("GET /search q=%r type=%r actors=%r", q, type, actors)
    filters = {k: v for k, v in {"type": type, "actors": actors}.items() if v}
    results = rag_store.search(q, k=settings.top_k, filters=filters or None)
    return {
        "query": q,
        "results": [
            {
                "title": doc.title,
                "actors": doc.metadata.get("actors", ""),
                "year": doc.metadata.get("year", ""),
                "url": doc.metadata.get("url", ""),
                "similarity": 1 / (1 + distance)
            }
            for doc, distance in results
        ]
    }
