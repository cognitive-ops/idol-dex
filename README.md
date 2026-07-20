# Idol Dex — IMDb RAG Chatbot

RAG-powered chatbot for movies, TV shows, actors and actresses. Loads IMDb's free [non-commercial datasets](https://datasets.imdbws.com/) (titles, ratings, cast, people — no plot summaries). Users ask questions; Claude answers using retrieved context from FAISS/Qdrant.

> The official IMDb API (developer.imdb.com) is GraphQL, paid, and gated behind AWS Data Exchange — no public key, no plain REST. This project uses the free non-commercial dataset dumps instead.

## Architecture

```
FastAPI server
    ↓
User query
    ↓
FAISS/Qdrant retrieval (semantic search via sentence-transformers)
    ↓
Retrieved documents (title/year/genres/rating/cast, or person/birth-year/known-for)
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

### Docker Compose (Qdrant backend)

```bash
docker compose up -d --build
```

Runs `qdrant` (6333), `api` (8000), `streamlit` (8501). Windows without a native Docker CLI: run the same command inside WSL (e.g. `wsl -d <distro> -- bash -lc "cd /mnt/d/Work/pet/idol-dex && docker compose up -d --build"`) — ports still publish to `localhost` on the Windows side.

`VECTOR_BACKEND=qdrant` is set for `api`/`streamlit` in `docker-compose.yml`, overriding the FAISS default. Downloaded IMDb dataset TSVs persist in the `imdb_data` named volume (`/app/data/imdb`), so re-ingesting after a container rebuild doesn't re-download. Qdrant's own storage persists in the `qdrant_data` volume.

```bash
curl -X POST http://localhost:8000/ingest
docker compose ps                 # check container health
docker compose logs -f api        # tail ingest progress
```

## Usage

### GUI (Streamlit)

```bash
# Terminal 1: Start FastAPI backend
python main.py

# Terminal 2: Start Streamlit frontend
streamlit run streamlit_app.py
```

Opens `http://localhost:8501` in browser. Chat, view sources, ingest data from GUI.

### CLI (API only)

#### 1. Ingest data (one-time)

```bash
python -m uvicorn src.api:app --reload
# POST /ingest
```

Downloads IMDb's dataset TSVs (cached under `IMDB_DATA_DIR`), joins top titles by vote count with their cast and person profiles, and indexes into FAISS/Qdrant. First run is slow (multi-hundred-MB dataset files); subsequent runs reuse the cached files.

#### 2. Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are Tom Hanks most popular movies?"}'
```

Response:
```json
{
  "answer": "Based on the database...",
  "sources": [
    {
      "title": "Forrest Gump",
      "url": "https://www.imdb.com/title/tt0109830/",
      "actors": "Tom Hanks, Robin Wright, Gary Sinise",
      "year": "1994",
      "genres": "Drama, Romance",
      "rating": 8.8,
      "similarity": 0.92
    }
  ]
}
```

#### 3. Direct search (no Claude)

```bash
curl "http://localhost:8000/search?q=heist"
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/ingest` | Load top titles + cast, index (one-time) |
| POST | `/ingest/people` | Load actor/actress profiles only |
| POST | `/chat` | Query with RAG + Claude |
| GET | `/search` | Direct metadata search |

## Components

- **embedder.py** — sentence-transformers (local, no API calls)
- **rag.py** — FAISS/Qdrant index + metadata storage
- **imdb_data.py** — downloads + parses IMDb non-commercial dataset TSVs
- **api.py** — FastAPI app + Claude integration
- **config.py** — env settings

## Limitations

- **No plot summaries** — the non-commercial datasets only ship structured facts (title, year, genres, rating, cast, birth/death year), not synopses
- **Bounded ingest** — only the top `IMDB_MAX_TITLES` titles by vote count are indexed (dataset has 10M+ titles total); raise the limit for broader coverage at the cost of slower ingest
- **No multi-modal** (images, trailers, video clips)
- **Local FAISS only** by default (Qdrant available via docker-compose)
- **Single replica** (no concurrent requests handling)
- **Docker on Windows** — this repo has no native Docker CLI on the host; run `docker compose` from inside a WSL distro with Docker installed

## Roadmap

- [ ] Resolve full known-for lists (not just titles already in the ingested top-N set)
- [ ] Add Postgres for distributed state + concurrent requests
- [ ] Cache query results + embeddings
- [ ] Multi-modal (poster images)
- [ ] Webhook ingestion (pick up IMDb's periodic dataset refreshes automatically)
- [ ] Analytics (query logging, popular searches)

## License

MIT
