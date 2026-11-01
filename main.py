import os
import shutil
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

from rag_engine import DocumentProcessor, UnifiedVectorStore, RAGPipeline, DATA_DIR
from advanced_rag_engine import AdvancedRAGPipeline
from graph_rag_engine import GraphRAGPipeline

vector_store = UnifiedVectorStore()
rag_pipeline = RAGPipeline(vector_store)
advanced_rag_pipeline = AdvancedRAGPipeline(vector_store)
graph_rag_pipeline = GraphRAGPipeline(vector_store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if vector_store.pg_store:
        vector_store.pg_store.close()
    rag_pipeline.close()
    advanced_rag_pipeline.close()
    graph_rag_pipeline.close()


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


class AdvancedQueryRequest(BaseModel):
    query: str
    use_hyde: Optional[bool] = True
    use_multi_query: Optional[bool] = True
    use_keyword_search: Optional[bool] = True
    top_k: Optional[int] = 4


class GraphQueryRequest(BaseModel):
    query: str
    depth: Optional[int] = 2
    top_k: Optional[int] = 4


def _save_and_ingest(file_obj, file_path: str, filename: str):
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file_obj, buffer)
    pages_content = DocumentProcessor.extract_text_from_file(file_path, filename)
    chunks = DocumentProcessor.chunk_document(filename, pages_content)
    vector_store.delete_document(filename)
    vector_store.add_documents(chunks)
    for c in chunks:
        graph_rag_pipeline.extract_triplets_from_text(c["text"], c["id"])
    graph_rag_pipeline.graph_store.save_graph()
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


# this is the advanced rag based code
@app.post("/api/advanced/query", summary="Query the Advanced RAG pipeline")
async def process_advanced_query(req: AdvancedQueryRequest):
    """Query using Advanced RAG: Multi-query expansion, HyDE, candidate deduplication, and semantic re-ranking."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    return await run_in_threadpool(
        advanced_rag_pipeline.generate_advanced_response,
        req.query,
        req.use_hyde if req.use_hyde is not None else True,
        req.use_multi_query if req.use_multi_query is not None else True,
        req.use_keyword_search if req.use_keyword_search is not None else True,
        req.top_k or 4
    )


# this is the vector search + keyword search code
@app.post("/api/advanced/hybrid-search", summary="Perform hybrid Vector Search + Keyword Search")
async def process_hybrid_search(req: AdvancedQueryRequest):
    """Retrieve chunks using hybrid Vector Search + Keyword Search and semantic re-ranking."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    return await run_in_threadpool(
        advanced_rag_pipeline.hybrid_search,
        req.query,
        req.top_k or 4
    )


# this is the graph rag based code
@app.post("/api/graph/query", summary="Query the Graph RAG pipeline")
async def process_graph_query(req: GraphQueryRequest):
    """Query using Graph RAG: Entity extraction, Subgraph traversal, and relational knowledge fusion."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    return await run_in_threadpool(
        graph_rag_pipeline.generate_graph_response,
        req.query,
        req.depth or 2,
        req.top_k or 4
    )


@app.get("/api/graph/data", summary="Get knowledge graph nodes and edges")
def get_graph_data():
    """Retrieve full knowledge graph representation (nodes, edges, statistics)."""
    return {
        "total_nodes": len(graph_rag_pipeline.graph_store.nodes),
        "total_edges": len(graph_rag_pipeline.graph_store.edges),
        "nodes": list(graph_rag_pipeline.graph_store.nodes.values()),
        "edges": graph_rag_pipeline.graph_store.edges
    }


@app.post("/api/graph/build", summary="Rebuild knowledge graph from stored document chunks")
async def build_knowledge_graph():
    """Rebuild or refresh the knowledge graph from all currently indexed document chunks."""
    return await run_in_threadpool(graph_rag_pipeline.auto_build_graph)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
