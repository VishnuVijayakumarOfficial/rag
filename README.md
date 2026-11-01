# RAG (Retrieval-Augmented Generation) Backend

A FastAPI backend for document ingestion, text chunking, vector embedding, and retrieval-augmented generation using PostgreSQL (`pgvector`) and Groq LLMs.

---

## Features

- **Document Parsing**: Extract text from `.pdf`, `.docx`, and `.txt` files.
- **Smart Chunking**: Split documents into overlapping chunks for semantic retrieval.
- **Vector Search**:
  - PostgreSQL with `pgvector` extension and HNSW indexing (with automatic SQL cosine similarity fallback).
  - Local file-based fallback vector store (`Numpy` + JSON).
- **Native RAG Pipeline**: Standard similarity retrieval and answer synthesis (`/api/query`).
- **Advanced RAG Pipeline**:
  - **Multi-Query Expansion**: Generates diverse search angles from user queries to maximize retrieval recall.
  - **HyDE (Hypothetical Document Embeddings)**: Hallucinates ideal answering passages to match semantic vector space.
  - **Multi-Pass Retrieval & Deduplication**: Merges multi-query and HyDE candidate pools without duplicate chunks.
  - **Re-Ranking & Relevance Filtering**: Re-scores candidate chunks with vector similarity and filters low-confidence context.
  - **Bracketed Source Citations**: Outputs inline references (`[Source 1]`) and confidence metrics (`/api/advanced/query`).
- **Graph RAG Pipeline**:
  - **Entity & Relation Extraction**: Extracts structured knowledge graph entities and directional relationship edges from text.
  - **Subgraph Traversal**: Performs multi-hop BFS neighbor expansion around query seed entities.
  - **Graph + Chunk Fusion**: Synthesizes answers combining structured relational triplets with textual document chunks (`/api/graph/query`).
  - **Graph Inspection**: Endpoint to view and inspect all nodes, edges, and connections in the graph (`/api/graph/data`).
- **FastAPI Endpoints**: REST API with interactive Swagger UI documentation.

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/VishnuVijayakumarOfficial/rag.git
cd rag
```

### 2. Set up virtual environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Copy `.env.example` to `.env` and fill in your settings:
```bash
cp .env.example .env
```

---

## Running the Server

```bash
python main.py
```
Or with Uvicorn:
```bash
uvicorn main:app --reload --port 8000
```

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## API Endpoints

- **`POST /api/upload`**: Upload documents (`.pdf`, `.docx`, `.txt`) to index into the vector database.
- **`POST /api/query`**: Native RAG query with standard top-k similarity retrieval.
- **`POST /api/advanced/query`**: Advanced RAG query featuring Multi-Query expansion, HyDE, candidate deduplication, and semantic re-ranking.
- **`POST /api/advanced/hybrid-search`**: Hybrid search combining Vector Search + Keyword Search with semantic re-ranking.
- **`POST /api/graph/query`**: Graph RAG query with entity linking, subgraph traversal, and relational knowledge fusion.
- **`GET /api/graph/data`**: Inspect all extracted knowledge graph nodes, relations, and connectivity.
- **`POST /api/graph/build`**: Rebuild knowledge graph relationships from all indexed document chunks.
