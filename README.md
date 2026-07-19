# Idol Dex — JAV RAG Chatbot

RAG-powered chatbot for Japanese Adult Video (JAV) information. Scrapes metadata from r18.com, javlibrary.com, dmm.co.jp. Users ask questions; Claude answers using retrieved context from FAISS vector index.

## Architecture

```
FastAPI server
    ↓
User query
    ↓
FAISS retrieval (semantic search via sentence-transformers)
    ↓
Retrieved documents (title, actors, date, plot)
    ↓
Claude (with RAG context)
    ↓
Answer + sources
```

## Setup

```bash
cd D:\Work\pet\idol-dex
pip install -r requirements.txt
cp .env.example .env
# Fill ANTHROPIC_API_KEY in .env
python main.py
```

## Usage

### 1. Ingest data (one-time)

```bash
python -m uvicorn src.api:app --reload
# POST /ingest
```

Scrapes r18.com, javlibrary, dmm.co.jp and indexes into FAISS.

### 2. Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the most popular JAV actresses?"}'
```

Response:
```json
{
  "answer": "Based on the database...",
  "sources": [
    {
      "title": "SDMU-605",
      "url": "https://www.r18.com/...",
      "actors": "Tsubomi, Aiko Natsukawa",
      "date": "2023-01-15",
      "similarity": 0.92
    }
  ]
}
```

### 3. Direct search (no Claude)

```bash
curl "http://localhost:8000/search?q=milf"
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/ingest` | Scrape sources + index (one-time) |
| POST | `/chat` | Query with RAG + Claude |
| GET | `/search` | Direct metadata search |

## Components

- **embedder.py** — sentence-transformers (local, no API calls)
- **rag.py** — FAISS index + metadata storage
- **scraper.py** — r18.com, javlibrary, dmm crawlers
- **api.py** — FastAPI app + Claude integration
- **config.py** — env settings

## Limitations

- **r18.com scraper** is basic (CSS selectors may need updates if site changes)
- **javlibrary, dmm** scrapers not fully implemented (need respectful rate-limiting, proxies for DMM outside Japan)
- **No multi-modal** (images, trailers, video clips)
- **Local FAISS only** (no distributed/cloud option yet)
- **Single replica** (no concurrent requests handling)

## Roadmap

- [ ] Implement full javlibrary + dmm scrapers with politeness (rate-limits, robots.txt respect)
- [ ] Add Postgres for distributed state + concurrent requests
- [ ] Cache query results + embeddings
- [ ] OpenSearch/Qdrant backend option (vs local FAISS)
- [ ] Multi-modal (images from listings, trailer metadata)
- [ ] Approval-gate sensitive content (safety check before returning results)
- [ ] Webhook ingestion (push new titles from source sites)
- [ ] Analytics (query logging, popular searches)

## Safety

- Answers flagged as "adult content" (metadata filtered if needed)
- Rate-limiting to prevent abuse
- Source attribution always included
- Claude instructed to be factual, not promotional

## License

MIT
