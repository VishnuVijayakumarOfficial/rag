# RAG (Retrieval-Augmented Generation) Backend

A FastAPI backend for document ingestion, text chunking, vector embedding, and retrieval-augmented generation using PostgreSQL (`pgvector`) and Groq LLMs.

---

## Features

- **Document Parsing**: Extract text from `.pdf`, `.docx`, and `.txt` files.
- **Smart Chunking**: Split documents into overlapping chunks for semantic retrieval.
- **Vector Search**:
  - PostgreSQL with `pgvector` extension (with automatic SQL cosine similarity fallback).
  - Local file-based fallback vector store (`Numpy` + JSON).
- **LLM Synthesis**: High-speed answer generation using Groq API (`openai/gpt-oss-20b` or custom models).
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
- **`POST /api/query`**: Submit a query to retrieve relevant chunks and generate synthesized answers.
