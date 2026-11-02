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
from modular_rag_engine import ModularRAGPipeline
from corrective_rag_engine import CorrectiveRAGPipeline
from self_rag_engine import SelfRAGPipeline
from adaptive_rag_engine import AdaptiveRAGPipeline
from agentic_rag_engine import AgenticRAGPipeline
from multimodal_rag_engine import MultimodalRAGPipeline

vector_store = UnifiedVectorStore()
rag_pipeline = RAGPipeline(vector_store)
advanced_rag_pipeline = AdvancedRAGPipeline(vector_store)
graph_rag_pipeline = GraphRAGPipeline(vector_store)
modular_rag_pipeline = ModularRAGPipeline(vector_store, graph_store=graph_rag_pipeline.graph_store)
corrective_rag_pipeline = CorrectiveRAGPipeline(vector_store)
self_rag_pipeline = SelfRAGPipeline(vector_store)
adaptive_rag_pipeline = AdaptiveRAGPipeline(vector_store)
agentic_rag_pipeline = AgenticRAGPipeline(vector_store)
multimodal_rag_pipeline = MultimodalRAGPipeline(vector_store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if vector_store.pg_store:
        vector_store.pg_store.close()
    rag_pipeline.close()
    advanced_rag_pipeline.close()
    graph_rag_pipeline.close()
    modular_rag_pipeline.close()
    corrective_rag_pipeline.close()
    self_rag_pipeline.close()
    adaptive_rag_pipeline.close()
    agentic_rag_pipeline.close()
    multimodal_rag_pipeline.close()


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


class ModularQueryRequest(BaseModel):
    query: str
    enabled_modules: Optional[List[str]] = None
    top_k: Optional[int] = 4


class CorrectiveQueryRequest(BaseModel):
    query: str
    correct_threshold: Optional[float] = 0.65
    incorrect_threshold: Optional[float] = 0.35
    top_k: Optional[int] = 4


class SelfRAGQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 4


class AdaptiveQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 4


class AgenticQueryRequest(BaseModel):
    query: str
    max_iterations: Optional[int] = 5


class MultimodalQueryRequest(BaseModel):
    query: str
    image_base64: Optional[str] = None
    top_k_text: Optional[int] = 4
    top_k_visual: Optional[int] = 3


class AssetRegistrationRequest(BaseModel):
    asset_id: str
    filename: str
    caption: str
    asset_type: Optional[str] = "image"
    page: Optional[int] = 1


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


# this is the modular rag based code
@app.post("/api/modular/query", summary="Query the Modular RAG pipeline")
async def process_modular_query(req: ModularQueryRequest):
    """Query using Modular RAG: Dynamic routing, pre-retrieval rewriting, multi-source retrieval, compression, and faithfulness evaluation."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    return await run_in_threadpool(
        modular_rag_pipeline.execute_modular_pipeline,
        req.query,
        req.enabled_modules,
        req.top_k or 4
    )


@app.get("/api/modular/modules", summary="Get available Modular RAG components")
def get_modular_components():
    """List all configurable plug-and-play modules available in the Modular RAG architecture."""
    return {
        "modules": [
            {"id": "router", "category": "Routing", "description": "Intent classification to select optimal retrieval strategies"},
            {"id": "rewriter", "category": "Pre-Retrieval", "description": "Query transformation and semantic densification"},
            {"id": "vector_retriever", "category": "Retrieval", "description": "Dense vector semantic similarity retrieval"},
            {"id": "keyword_retriever", "category": "Retrieval", "description": "Sparse keyword exact matching retrieval"},
            {"id": "graph_retriever", "category": "Retrieval", "description": "Knowledge graph multi-hop relational retrieval"},
            {"id": "compressor", "category": "Post-Retrieval", "description": "Context sentence filtering and passage compression"},
            {"id": "evaluator", "category": "Self-Correction", "description": "Faithfulness and hallucination verification"}
        ]
    }


# this is the corrective rag based code
@app.post("/api/corrective/query", summary="Query the Corrective RAG (CRAG) pipeline")
async def process_corrective_query(req: CorrectiveQueryRequest):
    """Query using Corrective RAG: Confidence grading, knowledge strip refinement, and corrective fallback."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    return await run_in_threadpool(
        corrective_rag_pipeline.generate_corrective_response,
        req.query,
        req.correct_threshold or 0.65,
        req.incorrect_threshold or 0.35,
        req.top_k or 4
    )


# this is the self rag based code
@app.post("/api/self-rag/query", summary="Query the Self-RAG pipeline")
async def process_self_rag_query(req: SelfRAGQueryRequest):
    """Query using Self-RAG: Reflection tokens [Retrieve], [IsRel], [IsSup], and [IsUse] with critique grading."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    return await run_in_threadpool(
        self_rag_pipeline.generate_self_rag_response,
        req.query,
        req.top_k or 4
    )


# this is the adaptive rag based code
@app.post("/api/adaptive/query", summary="Query the Adaptive RAG pipeline")
async def process_adaptive_query(req: AdaptiveQueryRequest):
    """Query using Adaptive RAG: Dynamic query complexity routing (No retrieval -> Single-step -> Multi-step decomposition)."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    return await run_in_threadpool(
        adaptive_rag_pipeline.generate_adaptive_response,
        req.query,
        req.top_k or 4
    )


# this is the agentic rag based code
@app.post("/api/agentic/query", summary="Query the Agentic RAG pipeline")
async def process_agentic_query(req: AgenticQueryRequest):
    """Query using Agentic RAG: ReAct autonomous loop with document search, keyword lookup, and calculation tools."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    return await run_in_threadpool(
        agentic_rag_pipeline.generate_agentic_response,
        req.query,
        req.max_iterations or 5
    )


# this is the multimodal rag based code
@app.post("/api/multimodal/query", summary="Query the Multimodal RAG pipeline")
async def process_multimodal_query(req: MultimodalQueryRequest):
    """Query using Multimodal RAG: Dual textual and visual asset retrieval and multimodal synthesis."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    return await run_in_threadpool(
        multimodal_rag_pipeline.generate_multimodal_response,
        req.query,
        req.image_base64,
        req.top_k_text or 4,
        req.top_k_visual or 3
    )


@app.get("/api/multimodal/assets", summary="Get all registered visual assets")
def get_multimodal_assets():
    """Retrieve catalog of all registered visual figures, diagrams, and images."""
    return {
        "total_assets": len(multimodal_rag_pipeline.asset_store.assets),
        "assets": multimodal_rag_pipeline.asset_store.assets
    }


@app.post("/api/multimodal/register-asset", summary="Register a visual asset for Multimodal RAG")
def register_multimodal_asset(req: AssetRegistrationRequest):
    """Register a new visual asset (figure, diagram, photo, table) with caption for multimodal retrieval."""
    multimodal_rag_pipeline.asset_store.add_asset(
        asset_id=req.asset_id,
        filename=req.filename,
        caption=req.caption,
        asset_type=req.asset_type or "image",
        page=req.page or 1
    )
    return {
        "status": "success",
        "message": f"Asset {req.asset_id} registered successfully",
        "asset_id": req.asset_id
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
