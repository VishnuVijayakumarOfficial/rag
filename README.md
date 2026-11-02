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
- **Modular RAG Pipeline**:
  - **Routing Module**: Evaluates query intent to dynamically activate optimal retrieval strategies.
  - **Pre-Retrieval Rewriting**: Semantic query densification and noise removal.
  - **Multi-Source Plug-and-Play Retrievers**: Interleaved Dense Vector, Sparse Keyword, and Knowledge Graph retrievers.
  - **Context Compression**: Filters out non-essential sentences to fit high-information density passages into the LLM context.
  - **Self-Correction & Faithfulness Evaluator**: Verifies that the synthesized answer is grounded in retrieved facts (`/api/modular/query`).
- **Corrective RAG (CRAG) Pipeline**:
  - **Retrieval Evaluator**: Grades document relevance against query, assigning a confidence score.
  - **Tri-State Action Decision**: Classifies retrieval outcome as `CORRECT`, `INCORRECT`, or `AMBIGUOUS`.
  - **Knowledge Refinement**: Decomposes documents into fine-grained strips, filtering irrelevant sentences.
  - **Corrective Fallback Search**: Triggers automated knowledge generation when documents are evaluated as irrelevant or ambiguous (`/api/corrective/query`).
- **Self-RAG Pipeline**:
  - **Reflection Tokens**: Introduces self-reflective critique tokens (`[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]`).
  - **Adaptive Retrieval Trigger**: Assesses whether external retrieval is genuinely necessary for the query.
  - **Passage Filtering**: Evaluates each retrieved chunk for relevance before LLM conditioning.
  - **Faithfulness Critique**: Verifies that generated assertions are grounded in retrieved context (`/api/self-rag/query`).
- **Adaptive RAG Pipeline**:
  - **Query Complexity Classifier**: Automatically categorizes queries into Low, Medium, or High complexity.
  - **Direct Answer Strategy (Tier 1)**: Directly answers simple greetings and arithmetic without unnecessary document retrieval.
  - **Single-Step Retrieval Strategy (Tier 2)**: Employs standard vector search for straightforward informational queries.
  - **Multi-Step Decomposition Strategy (Tier 3)**: Breaks complex multi-faceted queries into sub-questions, executing sequential retrieval and holistic synthesis (`/api/adaptive/query`).
- **Agentic RAG Pipeline**:
  - **Autonomous ReAct Reasoning Loop**: Iteratively reasons, selects tools, observes tool results, and refines thoughts (`Thought` -> `Action` -> `Observation`).
  - **Multi-Tool Suite**: Document vector search, exact keyword lookup, arithmetic calculation, and final answer synthesis.
  - **Self-Directed Termination**: Agent autonomously determines when sufficient information is gathered (`/api/agentic/query`).
- **Multimodal RAG Pipeline**:
  - **Visual Asset Store**: Stores and indexes diagrams, charts, figures, and photographs with rich descriptive captions.
  - **Dual Textual & Visual Retrieval**: Simultaneously retrieves text chunks from vector store and relevant visual figures from the asset store.
  - **Multimodal Context Fusion**: Synthesizes responses that explicitly cross-reference figures (`[Visual Asset X]`) and textual evidence (`/api/multimodal/query`).
  - **Asset Catalog & Registration**: Endpoints to list (`/api/multimodal/assets`) and register (`/api/multimodal/register-asset`) visual assets.
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
- **`POST /api/modular/query`**: Modular RAG query with dynamic routing, pre-retrieval rewriting, multi-source retrieval, compression, and evaluation.
- **`GET /api/modular/modules`**: List all configurable plug-and-play modules available in the Modular RAG framework.
- **`POST /api/corrective/query`**: Corrective RAG (CRAG) query with confidence grading, knowledge refinement, and fallback generation.
- **`POST /api/self-rag/query`**: Self-RAG query with reflection tokens (`[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]`) and critique scoring.
- **`POST /api/adaptive/query`**: Adaptive RAG query with dynamic query complexity classification (No Retrieval -> Single-Step -> Multi-Step decomposition).
- **`POST /api/agentic/query`**: Agentic RAG query with autonomous ReAct reasoning loop across multiple tools.
- **`POST /api/multimodal/query`**: Multimodal RAG query fusing textual passages and visual asset descriptions (diagrams, figures, charts).
- **`GET /api/multimodal/assets`**: Catalog of all registered multimodal visual assets and diagrams.
- **`POST /api/multimodal/register-asset`**: Register a new visual asset (diagram, chart, image) with descriptive caption.

---

## Example API Queries

### 1. Self-RAG Query
```bash
curl -X POST "http://localhost:8000/api/self-rag/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the core capabilities of the system?", "top_k": 4}'
```

### 2. Adaptive RAG Query
```bash
curl -X POST "http://localhost:8000/api/adaptive/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "Compare the performance differences and summarize the main trade-offs", "top_k": 4}'
```

### 3. Graph RAG Query
```bash
curl -X POST "http://localhost:8000/api/graph/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "How is the pipeline connected to the database?", "depth": 2, "top_k": 4}'
```

### 4. Agentic RAG Query
```bash
curl -X POST "http://localhost:8000/api/agentic/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "Search for system requirements, calculate total memory needed, and provide a summary", "max_iterations": 5}'
```

### 5. Multimodal RAG Query
```bash
curl -X POST "http://localhost:8000/api/multimodal/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "Explain the architecture diagram and workflow", "top_k_text": 4, "top_k_visual": 3}'
```
