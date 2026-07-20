"""FastAPI app — chat endpoint with RAG + Claude."""
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import anthropic

from .config import settings
from .rag import rag_store
from .scraper import ingest_idols_only, ingest_jav_data

logger = logging.getLogger(__name__)

app = FastAPI(title="JAV RAG Chatbot", version="0.1.0")
client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest():
    """One-time: scrape JAV data (movies + idols) and index into FAISS."""
    logger.debug("POST /ingest")
    count = await ingest_jav_data()
    logger.info("/ingest done: indexed=%d", count)
    return {"indexed": count, "index_path": settings.faiss_index_path}


@app.post("/ingest/idols")
async def ingest_idols():
    """Ingest idol profiles only (name, age, debut, cup size, movie codes)."""
    logger.debug("POST /ingest/idols")
    count = await ingest_idols_only()
    logger.info("/ingest/idols done: indexed=%d", count)
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
        if doc.metadata.get("type") == "idol":
            src.update({
                "name": doc.metadata.get("name", ""),
                "age": doc.metadata.get("age", ""),
                "debut": doc.metadata.get("debut", ""),
                "cup": doc.metadata.get("cup", ""),
                "movie_codes": doc.metadata.get("movie_codes", []),
            })
        else:
            src.update({
                "actors": doc.metadata.get("actors", ""),
                "date": doc.metadata.get("release_date", ""),
            })
        sources.append(src)

    context = "\n\n---\n\n".join(context_parts[:settings.top_k])
    context = context[:settings.max_context_length]

    # Query Claude with context
    system_prompt = (
        "You are a helpful JAV (Japanese Adult Video) information assistant. "
        "Answer user questions based on the provided context — this includes both movie "
        "titles (title, actors, release date, plot) and idol profiles (name, age, debut, "
        "cup size, movie codes). Be factual, helpful, and respectful. "
        "If information is not in the context, say so clearly."
    )

    user_message = f"""Context from JAV database:

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
                "date": doc.metadata.get("release_date", ""),
                "url": doc.metadata.get("url", ""),
                "similarity": 1 / (1 + distance)
            }
            for doc, distance in results
        ]
    }
