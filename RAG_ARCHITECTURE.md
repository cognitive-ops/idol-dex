# RAG Architecture: Data Flow from JAVDatabase to FAISS Index

## Overview

The JAV Chatbot uses a Retrieval-Augmented Generation (RAG) pipeline to fetch data from javdatabase.com, embed it, index it with FAISS, and retrieve relevant documents when users query.

```
┌─────────────────────────────────────────────────────────────────┐
│                    JAV RAG Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Scraper        Embedder         FAISS Index      Retrieval     │
│  ────────       ────────         ───────────      ─────────     │
│                                                                 │
│  javdatabase   sentence-tx   vector DB        Claude + Context │
│      ↓             ↓             ↓                   ↓          │
│   HTML       embedding()    add_documents()    search(query)   │
│   parse      384-dim vec    L2 distance        get context     │
│                                                Claude answers   │
└─────────────────────────────────────────────────────────────────┘
```

## Stage 1: Scraper (javdatabase.com → Document Objects)

### URL Sources
```
https://www.javdatabase.com/latest/
https://www.javdatabase.com/movies/
https://www.javdatabase.com/
```

### HTML → Document Parsing

**Input: HTML page**
```html
<div class="movie-item">
  <h2><a href="/movie/SDMU-605">SDMU-605 - Temptation Of A Married Woman</a></h2>
  <div class="actors">
    <a>Tsubomi</a>
    <a>Aiko Natsukawa</a>
  </div>
  <div class="release-date">2023-01-15</div>
  <p class="description">A story about temptation and desire</p>
</div>
```

**Processing** (`src/scraper.py:scrape_javdatabase()`)
```python
for item in soup.select(".movie-item"):
    title = "SDMU-605 - Temptation Of A Married Woman"
    actors = "Tsubomi, Aiko Natsukawa"
    date_str = "2023-01-15"
    plot = "A story about temptation and desire"
    
    # Create Document object
    doc = Document(
        doc_id="javdb:SDMU-605",
        title=title,
        metadata={
            "actors": actors,
            "release_date": date_str,
            "source": "javdatabase.com",
            "url": "https://www.javdatabase.com/movie/SDMU-605"
        },
        content="Title: SDMU-605 - Temptation Of A Married Woman\n"
                "Actors: Tsubomi, Aiko Natsukawa\n"
                "Date: 2023-01-15\n"
                "Plot: A story about temptation and desire"
    )
```

**Output: Document List**
```python
documents = [
    Document(doc_id="javdb:SDMU-605", title="...", metadata={...}, content="..."),
    Document(doc_id="javdb:DASD-802", title="...", metadata={...}, content="..."),
    Document(doc_id="javdb:SSIS-123", title="...", metadata={...}, content="..."),
    ...
]
```

## Stage 2: Embedder (Text → 384-dim Vectors)

### Model
- **Name:** `sentence-transformers/all-MiniLM-L6-v2`
- **Output:** 384-dimensional embeddings
- **Purpose:** Convert text to semantic vectors (similar meanings = close vectors)

### Embedding Process (`src/embedder.py`)

**Input: Document Content (text)**
```
"Title: SDMU-605 - Temptation Of A Married Woman
Actors: Tsubomi, Aiko Natsukawa
Date: 2023-01-15
Plot: A story about temptation and desire"
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

## Stage 3: FAISS Index (Vector Storage & Indexing)

### Index Type
- **Backend:** FAISS (Facebook AI Similarity Search)
- **Distance Metric:** L2 (Euclidean distance)
- **Index Type:** `IndexFlatL2` (exact search, no approximation)

### Indexing Process (`src/rag.py:RAGStore.add_documents()`)

**Input: Embeddings (N×384 matrix)**
```
Document 1: [0.23, -0.15, 0.89, ...]  → javdb:SDMU-605
Document 2: [0.12, 0.45, -0.32, ...]  → javdb:DASD-802
Document 3: [0.67, -0.23, 0.11, ...]  → javdb:SSIS-123
...
```

**FAISS Index Structure**
```python
index = faiss.IndexFlatL2(384)  # L2 distance, 384 dims
index.add(embeddings_np)        # Add all document vectors
```

**Index State (in-memory)**
```
FAISS Index {
  vector[0] = [0.23, -0.15, 0.89, ...]  ↔ Document ID: javdb:SDMU-605
  vector[1] = [0.12, 0.45, -0.32, ...]  ↔ Document ID: javdb:DASD-802
  vector[2] = [0.67, -0.23, 0.11, ...]  ↔ Document ID: javdb:SSIS-123
  ...
}
```

**Document Mapping (JSON)**
```json
{
  "javdb:SDMU-605": {
    "doc_id": "javdb:SDMU-605",
    "title": "SDMU-605 - Temptation Of A Married Woman",
    "metadata": {
      "actors": "Tsubomi, Aiko Natsukawa",
      "release_date": "2023-01-15",
      "source": "javdatabase.com",
      "url": "https://www.javdatabase.com/movie/SDMU-605"
    },
    "content": "Title: SDMU-605...\nActors: ...\nDate: ...\nPlot: ..."
  },
  "javdb:DASD-802": { ... },
  ...
}
```

## Stage 4: Persistence (Disk Storage)

### Save to Disk (`src/rag.py:RAGStore._save()`)

**FAISS Index File**
```
data/faiss_index/index.faiss  (binary format)
```

**Metadata JSON File**
```
data/metadata.json  (JSON format)
```

### Load on Restart (`src/rag.py:RAGStore._load()`)

```python
# Load index
index = faiss.read_index("data/faiss_index/index.faiss")

# Load metadata
with open("data/metadata.json") as f:
    documents = {doc["doc_id"]: Document(**doc) for doc in json.load(f)}
```

## Stage 5: Retrieval (Query → Top-K Documents)

### User Query Processing (`src/api.py:/chat`)

**Input: User Question**
```
"What are popular JAV actresses?"
```

**Embedding the Query**
```python
query_emb = embedder.embed_text(user_query)
# Returns: [0.25, -0.12, 0.91, ...]  (384 dims)
```

**FAISS Search**
```python
distances, indices = index.search(
    query_emb.reshape(1, -1),  # Shape: (1, 384)
    k=5                         # Top 5 results
)

# Returns:
# distances = [[0.15, 0.23, 0.45, 0.67, 0.89]]  (L2 distances)
# indices = [[0, 2, 1, 5, 3]]  (positions in index)
```

**Similarity Calculation**
```python
# L2 distance → similarity score (0-1)
similarity = 1 / (1 + distance)

# Example:
# distance=0.15  → similarity=0.87  (87% match)
# distance=0.45  → similarity=0.69  (69% match)
```

**Retrieve Documents**
```python
results = []
for idx, distance in zip(indices[0], distances[0]):
    doc = documents_list[idx]  # Get document by position
    similarity = 1 / (1 + distance)
    results.append((doc, similarity))

# results = [
#   (Document(javdb:SDMU-605, actors="Tsubomi, Aiko Natsukawa", ...), 0.87),
#   (Document(javdb:SSIS-123, actors="Yuki Nagano", ...), 0.78),
#   (Document(javdb:DASD-802, actors="Mio Kimijima", ...), 0.69),
#   ...
# ]
```

### Context for Claude

**Retrieved Context (top 3 results)**
```
Source: SDMU-605 - Temptation Of A Married Woman
Title: SDMU-605 - Temptation Of A Married Woman
Actors: Tsubomi, Aiko Natsukawa
Date: 2023-01-15
Plot: A story about temptation and desire

---

Source: SSIS-123 - Perfect Companion
Title: SSIS-123 - Perfect Companion
Actors: Yuki Nagano
Date: 2023-03-05
Plot: A perfect companion for everyday life

---

Source: DASD-802 - Innocent Angel
Title: DASD-802 - Innocent Angel
Actors: Mio Kimijima
Date: 2023-02-10
Plot: An innocent journey turns into passion
```

## Stage 6: Claude Answers with RAG Context

### System Prompt
```
You are a helpful JAV information assistant. Answer user questions 
based on the provided context (database of JAV titles, actors, 
release dates, plots). Be factual, helpful, and respectful.
```

### User Message
```
Context from JAV database:

[Retrieved context from Stage 5]

---

User question: What are popular JAV actresses?

Please answer based on the context above.
```

### Claude Response
```
Based on the database, popular JAV actresses include:

1. **Tsubomi** - Known for "SDMU-605: Temptation Of A Married Woman" (2023-01-15)
   This title features her in a compelling story about temptation and desire.

2. **Yuki Nagano** - Starred in "SSIS-123: Perfect Companion" (2023-03-05)
   Described as a perfect companion for everyday life.

3. **Mio Kimijima** - Appeared in "DASD-802: Innocent Angel" (2023-02-10)
   An innocent journey that turns into passion.

These actresses have multiple releases and appear in recent titles from 2023.
```

## Data Flow Summary

```
┌─────────────────┐
│ javdatabase.com │
└────────┬────────┘
         │ HTML scraping
         ↓
┌──────────────────────────┐
│ Scraper (scraper.py)     │
│ Extract: title, actors,  │
│ date, plot               │
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
│ RAGStore (rag.py)        │
│ FAISS index + metadata   │
│ Disk: .faiss + .json     │
└────────┬─────────────────┘
         │ On restart
         ↓
┌──────────────────────────┐
│ API (api.py:/chat)       │
│ Embed query              │
│ FAISS search             │
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
| `src/scraper.py` | Fetch HTML from javdatabase, parse, create Documents |
| `src/embedder.py` | Convert Document text → 384-dim vectors |
| `src/rag.py` | FAISS index, store/load embeddings, search |
| `src/api.py` | HTTP endpoints, query embedding, retrieval |
| `data/faiss_index/index.faiss` | Binary FAISS index (disk) |
| `data/metadata.json` | Document metadata (title, actors, url) |

## Performance Notes

- **Indexing:** ~100ms for 1000 documents
- **Search:** ~5ms per query (L2 distance, exact match)
- **Memory:** ~4MB per 1000 documents (FAISS + metadata)
- **Scalability:** FAISS can handle millions of vectors; for >1M consider GPU or approximate search

## Future Improvements

- [ ] Use `IndexIVFFlat` for approximate search (faster, lower memory)
- [ ] Add GPU support (FAISS GPU backend)
- [ ] Use Postgres for metadata (enable filtering by actor, date, etc.)
- [ ] Implement incremental indexing (add new docs without full rebuild)
- [ ] Add reranking (coarse search with FAISS, fine-rank with Claude)
