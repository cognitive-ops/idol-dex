# RAG Architecture: Data Flow from IMDb Datasets to Vector Store

## Overview

The IMDb Chatbot uses a Retrieval-Augmented Generation (RAG) pipeline to load IMDb's free [non-commercial dataset dumps](https://datasets.imdbws.com/), join them into documents, embed them, index them in a vector store, and retrieve relevant documents when users query.

> IMDb's official API (developer.imdb.com) is GraphQL, paid, and distributed exclusively via AWS Data Exchange — no public API key, no plain REST endpoint. The non-commercial datasets are the free alternative: gzipped TSV dumps of titles, ratings, cast, and people, refreshed daily, with no plot summaries.

Vector store backend is pluggable (`src/rag.py`), picked via `settings.vector_backend`:

| Backend | When used | Storage |
|---------|-----------|---------|
| `qdrant` | docker-compose stack (`docker-compose.yml` sets `VECTOR_BACKEND=qdrant`) | Qdrant server, collection `imdb_docs`, vectors + payload live in Qdrant (no local files) |
| `faiss` | local dev default (`Settings.vector_backend` default) | `data/faiss_index/index.faiss` + `data/metadata.json` on disk |

Both backends implement the same interface (`add_documents`, `search`) and are selected once at import time via `_build_store()`, exposed as the module-level singleton `rag_store`.

```
┌─────────────────────────────────────────────────────────────────┐
│                    IMDb RAG Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Dataset Loader  Embedder      Vector Store       Retrieval      │
│  ─────────────   ────────      ────────────       ─────────      │
│                                                                 │
│  datasets.imdbws  sentence-tx   Qdrant / FAISS   Claude + Context │
│      ↓             ↓             ↓                   ↓          │
│   TSV.gz     embedding()    add_documents()    search(query)   │
│   parse+join 384-dim vec    cosine / L2        get context     │
│                                                Claude answers   │
└─────────────────────────────────────────────────────────────────┘
```

## Stage 1: Dataset Loader (TSV dumps → Document Objects)

### Dataset Sources (`src/imdb_data.py`)

```
https://datasets.imdbws.com/title.basics.tsv.gz      (tconst, titleType, primaryTitle, isAdult, startYear, genres, ...)
https://datasets.imdbws.com/title.ratings.tsv.gz      (tconst, averageRating, numVotes)
https://datasets.imdbws.com/title.principals.tsv.gz   (tconst, ordering, nconst, category, ...)
https://datasets.imdbws.com/name.basics.tsv.gz        (nconst, primaryName, birthYear, deathYear, primaryProfession, knownForTitles)
```

Files are downloaded once and cached under `IMDB_DATA_DIR` (`IMDbDataSource._download()`); re-ingesting reuses the cached copy instead of re-downloading.

### Join Pipeline (`IMDbDataSource` + `_build_docs()`)

1. **`top_titles()`** — loads `title.ratings.tsv.gz` into a `{tconst: (rating, votes)}` dict filtered by `IMDB_MIN_VOTES`, then streams `title.basics.tsv.gz` and keeps the top `IMDB_MAX_TITLES` by vote count (min-heap, `titleType` restricted to `IMDB_TITLE_TYPES`, adult titles excluded).
2. **`cast_for_titles()`** — streams `title.principals.tsv.gz` once, collecting up to `IMDB_CAST_PER_TITLE` actor/actress `nconst`s per selected `tconst`.
3. **`people()`** — streams `name.basics.tsv.gz` once, resolving names/birth-year/death-year/professions/known-for for every `nconst` collected above.
4. Titles and people are cross-linked: each title document lists its cast by resolved name; each person document lists "known for" titles that overlap the ingested title set.

### TSV → Document Parsing

**Input: title.basics.tsv.gz row + title.ratings.tsv.gz row (joined on tconst)**
```
tconst=tt0109830  titleType=movie  primaryTitle=Forrest Gump  startYear=1994  genres=Drama,Romance
tconst=tt0109830  averageRating=8.8  numVotes=2200000
```

**Processing** (`src/imdb_data.py:_title_document()`)
```python
doc = Document(
    doc_id="title:tt0109830",
    title="Forrest Gump",
    metadata={
        "type": "title",
        "tconst": "tt0109830",
        "title_type": "movie",
        "year": "1994",
        "genres": "Drama,Romance",
        "rating": 8.8,
        "votes": 2200000,
        "actors": "Tom Hanks, Robin Wright, Gary Sinise",
        "url": "https://www.imdb.com/title/tt0109830/",
    },
    content="Title: Forrest Gump (1994)\n"
            "Type: movie\n"
            "Genres: Drama,Romance\n"
            "Rating: 8.8/10 (2200000 votes)\n"
            "Cast: Tom Hanks, Robin Wright, Gary Sinise"
)
```

**Input: name.basics.tsv.gz row**
```
nconst=nm0000158  primaryName=Tom Hanks  birthYear=1956  primaryProfession=actor,producer,soundtrack
```

**Processing** (`src/imdb_data.py:_person_document()`)
```python
doc = Document(
    doc_id="person:nm0000158",
    title="Tom Hanks",
    metadata={
        "type": "person",
        "nconst": "nm0000158",
        "name": "Tom Hanks",
        "birth_year": "1956",
        "death_year": "",
        "professions": "actor,producer,soundtrack",
        "known_for": ["Forrest Gump", "Cast Away"],
        "url": "https://www.imdb.com/name/nm0000158/",
    },
    content="Name: Tom Hanks\n"
            "Born: 1956\n"
            "Professions: actor,producer,soundtrack\n"
            "Known for: Forrest Gump, Cast Away"
)
```

**Output: Document List**
```python
title_docs, person_docs = _build_docs(limit=300)
documents = title_docs + person_docs
```

## Stage 2: Embedder (Text → 384-dim Vectors)

### Model
- **Name:** `sentence-transformers/all-MiniLM-L6-v2`
- **Output:** 384-dimensional embeddings
- **Purpose:** Convert text to semantic vectors (similar meanings = close vectors)

### Embedding Process (`src/embedder.py`)

**Input: Document Content (text)**
```
"Title: Forrest Gump (1994)
Type: movie
Genres: Drama,Romance
Rating: 8.8/10 (2200000 votes)
Cast: Tom Hanks, Robin Wright, Gary Sinise"
```

**Processing**
```python
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embedding = embedder.encode(content)
# Returns numpy array: [0.23, -0.15, 0.89, ..., 0.12]  # 384 dims
```

**Output: Vector**
```
[0.23, -0.15, 0.89, 0.45, -0.32, ..., 0.12, 0.67, -0.23]
 ↑                                                      ↑
 dim 0                                              dim 383
```

### Batch Embedding
```python
embeddings = embedder.embed_batch([doc1.content, doc2.content, doc3.content, ...])
# Returns numpy array (N, 384) where N = number of documents
```

## Stage 3: Vector Store (Storage & Indexing)

Two interchangeable implementations in `src/rag.py`, chosen by `settings.vector_backend`.

### Qdrant backend (`src/rag.py:QdrantStore`) — used by docker-compose

- **Backend:** Qdrant server (`qdrant/qdrant:v1.11.3`, port 6333)
- **Distance Metric:** Cosine
- **Collection:** `imdb_docs` (name via `settings.qdrant_collection`), created on first run with `VectorParams(size=384, distance=Distance.COSINE)`
- **Storage:** vectors + full document payload (`doc_id`, `title`, `metadata`, `content`) stored together as Qdrant points — no separate metadata file
- **Point ID:** deterministic `uuid.uuid5(NAMESPACE_URL, doc_id)`, so re-indexing the same `doc_id` upserts in place

**Indexing Process (`QdrantStore.add_documents()`)**
```python
points = [
    PointStruct(id=_point_id(doc.doc_id), vector=emb, payload=doc.to_dict())
    for doc, emb in zip(docs, embeddings)
]
client.upsert(collection_name="imdb_docs", points=points)
```

**Search (`QdrantStore.search()`)**
```python
hits = client.search(collection_name="imdb_docs", query_vector=query_emb, limit=k)
# hit.payload -> reconstruct Document
# hit.score is cosine similarity (higher = more relevant)
distance = 1.0 - hit.score  # normalized to distance-like value, lower = more relevant
```

### FAISS backend (`src/rag.py:RAGStore`) — local dev default

- **Backend:** FAISS (Facebook AI Similarity Search), in-process
- **Distance Metric:** L2 (Euclidean distance)
- **Index Type:** `IndexFlatL2` (exact search, no approximation)
- **Storage:** index vectors in a `.faiss` binary file, document metadata in a separate `.json` file, joined by list position

**Indexing Process (`RAGStore.add_documents()`)**
```python
index = faiss.IndexFlatL2(384)  # L2 distance, 384 dims
index.add(embeddings_np)        # Add all document vectors
```

**Index State (in-memory)**
```
FAISS Index {
  vector[0] = [0.23, -0.15, 0.89, ...]  ↔ Document ID: title:tt0109830
  vector[1] = [0.12, 0.45, -0.32, ...]  ↔ Document ID: person:nm0000158
  vector[2] = [0.67, -0.23, 0.11, ...]  ↔ Document ID: title:tt0468569
  ...
}
```

**Document Mapping (JSON)**
```json
{
  "title:tt0109830": {
    "doc_id": "title:tt0109830",
    "title": "Forrest Gump",
    "metadata": {
      "type": "title",
      "year": "1994",
      "genres": "Drama,Romance",
      "rating": 8.8,
      "actors": "Tom Hanks, Robin Wright, Gary Sinise",
      "url": "https://www.imdb.com/title/tt0109830/"
    },
    "content": "Title: Forrest Gump...\nGenres: ...\nRating: ...\nCast: ..."
  },
  "person:nm0000158": { ... },
  ...
}
```

## Stage 4: Persistence

### Qdrant — server-side, no local files

Persistence handled by the Qdrant server itself, backed by a docker volume (`qdrant_data:/qdrant/storage` in `docker-compose.yml`). App container has no on-disk index/metadata; restarting the API container just reconnects to the existing collection (`_ensure_collection()` is a no-op if the collection already exists).

### FAISS — disk files (`src/rag.py:RAGStore._save()` / `_load()`)

**FAISS Index File**
```
data/faiss_index/index.faiss  (binary format)
```

**Metadata JSON File**
```
data/metadata.json  (JSON format)
```

**Load on Restart**
```python
# Load index
index = faiss.read_index("data/faiss_index/index.faiss")

# Load metadata
with open("data/metadata.json") as f:
    documents = {doc["doc_id"]: Document(**doc) for doc in json.load(f)}
```

### Dataset cache (`src/imdb_data.py:IMDbDataSource._download()`)

```
data/imdb/title.basics.tsv.gz
data/imdb/title.ratings.tsv.gz
data/imdb/title.principals.tsv.gz
data/imdb/name.basics.tsv.gz
```

Downloaded once, reused on every subsequent `/ingest` call unless deleted manually.

## Stage 5: Retrieval (Query → Top-K Documents)

### User Query Processing (`src/api.py:/chat`)

Caller always goes through the shared `rag_store.search(query, k)` — backend swap is transparent to `api.py`.

**Input: User Question**
```
"What are Tom Hanks' most popular movies?"
```

**Embedding the Query**
```python
query_emb = embedder.embed_text(user_query)
# Returns: [0.25, -0.12, 0.91, ...]  (384 dims)
```

**Qdrant search internals**
```python
hits = client.search(collection_name="imdb_docs", query_vector=query_emb, limit=k)
# hit.score = cosine similarity, higher = more relevant
# distance = 1.0 - hit.score, for a consistent "lower = more relevant" contract
```

**FAISS search internals**
```python
distances, indices = index.search(
    query_emb.reshape(1, -1),  # Shape: (1, 384)
    k=5                         # Top 5 results
)

# Returns:
# distances = [[0.15, 0.23, 0.45, 0.67, 0.89]]  (L2 distances)
# indices = [[0, 2, 1, 5, 3]]  (positions in index)
```

**Similarity Calculation (FAISS L2 distance → 0-1 score)**
```python
# L2 distance → similarity score (0-1)
similarity = 1 / (1 + distance)

# Example:
# distance=0.15  → similarity=0.87  (87% match)
# distance=0.45  → similarity=0.69  (69% match)
```

**Result shape (both backends return the same `list[tuple[Document, float]]`)**
```python
# results = [
#   (Document(person:nm0000158, name="Tom Hanks", ...), 0.87),
#   (Document(title:tt0109830, title="Forrest Gump", ...), 0.78),
#   (Document(title:tt0088763, title="Big", ...), 0.69),
#   ...
# ]
```

### Context for Claude

**Retrieved Context (top 3 results)**
```
Source: Tom Hanks
Name: Tom Hanks
Born: 1956
Professions: actor,producer,soundtrack
Known for: Forrest Gump, Cast Away

---

Source: Forrest Gump
Title: Forrest Gump (1994)
Type: movie
Genres: Drama,Romance
Rating: 8.8/10 (2200000 votes)
Cast: Tom Hanks, Robin Wright, Gary Sinise

---

Source: Cast Away
Title: Cast Away (2000)
Type: movie
Genres: Adventure,Drama,Romance
Rating: 7.8/10 (500000 votes)
Cast: Tom Hanks, Helen Hunt
```

## Stage 6: Claude Answers with RAG Context

### System Prompt
```
You are a helpful movie/TV information assistant backed by IMDb's
non-commercial dataset. Answer user questions based on the provided
context (titles with year/genres/rating/cast, people with birth/death
year/professions/known-for). Be factual and helpful.
```

### User Message
```
Context from IMDb dataset:

[Retrieved context from Stage 5]

---

User question: What are Tom Hanks' most popular movies?

Please answer based on the context above.
```

### Claude Response
```
Based on the database, Tom Hanks' most popular movies (by IMDb vote count) include:

1. **Forrest Gump** (1994) — Drama/Romance, rated 8.8/10 with 2.2M votes.
   Also stars Robin Wright and Gary Sinise.

2. **Cast Away** (2000) — Adventure/Drama/Romance, rated 7.8/10 with 500K votes.
   Also stars Helen Hunt.

These are the titles in the indexed dataset featuring Tom Hanks as top-billed cast.
```

## Data Flow Summary

```
┌──────────────────────────┐
│ datasets.imdbws.com      │
└────────┬──────────────────┘
         │ TSV.gz download (cached)
         ↓
┌──────────────────────────┐
│ IMDbDataSource            │
│ (imdb_data.py)            │
│ Join: titles + ratings +  │
│ cast + people              │
└────────┬─────────────────┘
         │ Document objects
         ↓
┌──────────────────────────┐
│ Embedder                 │
│ sentence-transformers    │
│ 384-dim vectors          │
└────────┬─────────────────┘
         │ Embeddings
         ↓
┌──────────────────────────┐
│ rag_store (rag.py)       │
│ QdrantStore (docker) or  │
│ RAGStore/FAISS (local)   │
│ picked via vector_backend│
└────────┬─────────────────┘
         │ On restart: reconnect (Qdrant) / reload from disk (FAISS)
         ↓
┌──────────────────────────┐
│ API (api.py:/chat)       │
│ Embed query              │
│ rag_store.search()       │
│ Retrieve top-k docs      │
└────────┬─────────────────┘
         │ Context + metadata
         ↓
┌──────────────────────────┐
│ Claude                   │
│ Answer with context      │
└──────────────────────────┘
```

## Key Components

| File | Role |
|------|------|
| `src/imdb_data.py` | Download + stream-parse IMDb dataset TSVs, join titles/ratings/cast/people, create Documents |
| `src/embedder.py` | Convert Document text → 384-dim vectors |
| `src/rag.py` | `rag_store` factory, `QdrantStore` + `RAGStore` (FAISS) implementations, search |
| `src/config.py` | `vector_backend` switch + per-backend settings (`qdrant_url`, `qdrant_collection`, `faiss_index_path`, `metadata_path`, `imdb_*`) |
| `src/api.py` | HTTP endpoints, query embedding, retrieval |
| `docker-compose.yml` | `qdrant` service (port 6333, volume `qdrant_data`) + app services wired with `VECTOR_BACKEND=qdrant` |
| Qdrant collection `imdb_docs` | Vectors + payload (title, actors, url, content) — server-side, no local files |
| `data/faiss_index/index.faiss` | Binary FAISS index (disk, local-dev fallback only) |
| `data/metadata.json` | Document metadata (local-dev fallback only) |
| `data/imdb/*.tsv.gz` | Cached IMDb dataset downloads |

## Performance Notes

- **Qdrant indexing/search:** network round-trip to the Qdrant server per call (local docker network, typically low-single-digit ms overhead beyond the vector op itself); cosine similarity, HNSW-backed once collection grows past Qdrant's flat-search threshold
- **FAISS indexing:** ~100ms for 1000 documents
- **FAISS search:** ~5ms per query (L2 distance, exact match, in-process)
- **Dataset ingest:** dominated by streaming `name.basics.tsv.gz` and `title.principals.tsv.gz` (tens of millions of rows each) — one full pass per file, first run also pays the TSV.gz download; cached afterward
- **Memory:** ~4MB per 1000 documents for FAISS (index + metadata in-process); Qdrant holds vectors/payload server-side instead of in the app process
- **Scalability:** Qdrant scales via its own server (sharding, disk-backed storage, filtering on payload) — preferred path past local dev. FAISS `IndexFlatL2` is exact but in-memory only; for >1M vectors without Qdrant, consider `IndexIVFFlat` or GPU

## Future Improvements

- [x] Incremental indexing without full rebuild — Qdrant `upsert` by deterministic point ID (`uuid5(doc_id)`) already does this
- [x] Payload filtering (actor, year, etc.) — Qdrant payload supports filtered search; not yet exposed in `QdrantStore.search()` or `api.py`
- [ ] Resolve full known-for lists via a second `title.basics` lookup instead of only titles already in the ingested top-N set
- [ ] Use `IndexIVFFlat` for FAISS approximate search (faster, lower memory) — FAISS path only, relevant for local dev with large corpora
- [ ] Add GPU support (FAISS GPU backend) — FAISS path only
- [ ] Add reranking (coarse vector search, fine-rank with Claude)
- [ ] Retire FAISS backend once Qdrant is the only deploy target, or keep for offline/no-docker dev
