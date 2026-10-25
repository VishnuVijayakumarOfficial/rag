import os
import shutil
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel

from rag_engine import DocumentProcessor, UnifiedVectorStore, RAGPipeline, DATA_DIR

app = FastAPI(
    title="RAG Backend API",
    description="Minimal RAG system API with document ingestion and retrieval-augmented generation.",
    version="1.0.0"
)

# Fix Swagger UI file upload rendering (forces binary format instead of array<string>)
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    for component in schema.get("components", {}).get("schemas", {}).values():
        for prop in component.get("properties", {}).values():
            if prop.get("contentMediaType") == "application/octet-stream":
                prop.pop("contentMediaType", None)
                prop["format"] = "binary"
            items = prop.get("items", {})
            if items.get("contentMediaType") == "application/octet-stream":
                items.pop("contentMediaType", None)
                items["format"] = "binary"
    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi

# CORS middleware for frontend / Swagger communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector store and RAG pipeline
vector_store = UnifiedVectorStore()
rag_pipeline = RAGPipeline(vector_store)


class QueryRequest(BaseModel):
    query: str
    mode: Optional[str] = "RAG: Document Search"


@app.post("/api/upload", summary="Upload and index documents for RAG")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload documents (.pdf, .docx, .txt), extract text, split into chunks, and index into vector store."""
    uploaded_summary = []

    for file in files:
        if not file.filename:
            continue

        file_path = os.path.join(DATA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            pages_content = DocumentProcessor.extract_text_from_file(file_path, file.filename)
            chunks = DocumentProcessor.chunk_document(file.filename, pages_content)
            
            # Delete any previous chunk records for this document
            vector_store.delete_document(file.filename)
            # Insert into vector store
            vector_store.add_documents(chunks)

            uploaded_summary.append({
                "filename": file.filename,
                "status": "success",
                "chunks_created": len(chunks),
                "pages_parsed": len(pages_content)
            })
        except Exception as e:
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
def process_query(req: QueryRequest):
    """Retrieve relevant chunks from vector store and generate an answer using the RAG pipeline."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    result = rag_pipeline.generate_response(req.query, mode=req.mode or "RAG: Document Search")
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
