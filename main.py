import os
import shutil
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

from rag_engine import DocumentProcessor, UnifiedVectorStore, RAGPipeline, DATA_DIR

vector_store = UnifiedVectorStore()
rag_pipeline = RAGPipeline(vector_store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if vector_store.pg_store:
        vector_store.pg_store.close()
    rag_pipeline.close()


app = FastAPI(
    title="RAG Backend API",
    description="Minimal RAG system API with document ingestion and retrieval-augmented generation.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    mode: Optional[str] = "RAG: Document Search"


def _save_and_ingest(file_obj, file_path: str, filename: str):
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file_obj, buffer)
    pages_content = DocumentProcessor.extract_text_from_file(file_path, filename)
    chunks = DocumentProcessor.chunk_document(filename, pages_content)
    vector_store.delete_document(filename)
    vector_store.add_documents(chunks)
    return len(chunks), len(pages_content)


@app.post("/api/upload", summary="Upload and index documents for RAG")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload documents (.pdf, .docx, .txt), extract text, split into chunks, and index into vector store."""
    uploaded_summary = []

    for file in files:
        if not file.filename:
            continue

        file_path = os.path.join(DATA_DIR, file.filename)

        try:
            chunks_count, pages_count = await run_in_threadpool(
                _save_and_ingest, file.file, file_path, file.filename
            )

            uploaded_summary.append({
                "filename": file.filename,
                "status": "success",
                "chunks_created": chunks_count,
                "pages_parsed": pages_count
            })
        except Exception as e:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            uploaded_summary.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })

    status_info = vector_store.get_status()
    return {
        "message": f"Processed {len(files)} file(s)",
        "details": uploaded_summary,
        "total_chunks": status_info["total_chunks"],
        "backend": status_info["backend"]
    }


@app.post("/api/query", summary="Query the RAG pipeline")
async def process_query(req: QueryRequest):
    """Retrieve relevant chunks from vector store and generate an answer using the LLM pipeline."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    return await run_in_threadpool(
        rag_pipeline.generate_response,
        req.query,
        req.mode or "RAG: Document Search"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
