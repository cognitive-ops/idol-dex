"""FastAPI app — chat endpoint with RAG + Claude."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import anthropic

from .config import settings
from .rag import rag_store
from .scraper import ingest_jav_data

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
    """One-time: scrape JAV data and index into FAISS."""
    count = await ingest_jav_data()
    return {"indexed": count, "index_path": settings.faiss_index_path}


@app.post("/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    """Chat endpoint with RAG."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query required")

    # Retrieve relevant documents
    search_results = rag_store.search(req.query, k=settings.top_k)
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
        sources.append({
            "title": doc.title,
            "url": doc.metadata.get("url", ""),
            "actors": doc.metadata.get("actors", ""),
            "date": doc.metadata.get("release_date", ""),
            "similarity": 1 / (1 + distance)  # convert L2 distance to similarity
        })

    context = "\n\n---\n\n".join(context_parts[:settings.top_k])
    context = context[:settings.max_context_length]

    # Query Claude with context
    system_prompt = (
        "You are a helpful JAV (Japanese Adult Video) information assistant. "
        "Answer user questions based on the provided context (database of JAV titles, "
        "actors, release dates, plots). Be factual, helpful, and respectful. "
        "If information is not in the context, say so clearly."
    )

    user_message = f"""Context from JAV database:

{context}

---

User question: {req.query}

Please answer based on the context above. If the question cannot be answered from the context, say so."""

    response = client.messages.create(
        model=settings.model,
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    answer = response.content[0].text

    return ChatResponse(answer=answer, sources=sources)


@app.get("/search")
async def search(q: str):
    """Direct metadata search (no Claude)."""
    results = rag_store.search(q, k=settings.top_k)
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
