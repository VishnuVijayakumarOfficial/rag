import os
import json
from typing import List, Dict, Any, Optional
import numpy as np
import httpx
from pypdf import PdfReader
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer

from config import (
    VECTOR_STORE_BACKEND,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    EMBEDDING_MODEL_NAME,
    GROQ_API_KEY,
    GROQ_MODEL
)
from postgres_store import PostgresVectorStore

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
STORE_FILE = os.path.join(DATA_DIR, "vector_store.json")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
os.makedirs(DATA_DIR, exist_ok=True)

UNICODE_TRANSLATION = str.maketrans({"\u202f": " ", "\u00a0": " "})


# this is the nlp based code
class DocumentProcessor:
    @staticmethod
    def extract_text_from_file(file_path: str, filename: str) -> List[Dict[str, Any]]:
        ext = os.path.splitext(filename)[1].lower()
        pages_content = []

        if ext == ".pdf":
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages_content.append({"page": page_num, "text": text.strip()})

        elif ext == ".docx":
            doc = DocxDocument(file_path)
            full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            if full_text.strip():
                pages_content.append({"page": 1, "text": full_text})

        elif ext in [".txt", ".md", ".json", ".csv"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                full_text = f.read()
            if full_text.strip():
                pages_content.append({"page": 1, "text": full_text})

        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return pages_content

    # this is the tokenization based code
    @staticmethod
    def chunk_document(filename: str, pages_content: List[Dict[str, Any]], chunk_size: int = 500, overlap: int = 100) -> List[Dict[str, Any]]:
        chunks = []
        doc_type = os.path.splitext(filename)[1].lower().replace(".", "")
        chunk_id = 0
        step = max(chunk_size - overlap, 1)

        for item in pages_content:
            page_num = item["page"]
            text = item["text"]

            # this is where split the input into tokens
            words = text.split()
            total_words = len(words)
            if not total_words:
                continue

            if total_words <= chunk_size:
                chunk_text = " ".join(words)
                if len(chunk_text) > 20:
                    chunk_id += 1
                    chunks.append({
                        "id": f"{filename}_p{page_num}_c{chunk_id}",
                        "filename": filename,
                        "page": page_num,
                        "text": chunk_text,
                        "doc_type": doc_type
                    })
                continue

            for i in range(0, total_words, step):
                chunk_words = words[i:i + chunk_size]
                chunk_text = " ".join(chunk_words)

                if len(chunk_text) > 20:
                    chunk_id += 1
                    chunks.append({
                        "id": f"{filename}_p{page_num}_c{chunk_id}",
                        "filename": filename,
                        "page": page_num,
                        "text": chunk_text,
                        "doc_type": doc_type
                    })

        return chunks


class FileVectorStore:
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.load_index()

    def add_documents(self, new_chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        if not new_chunks:
            return

        self.chunks.extend(new_chunks)
        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        self.save_index()

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Dict[str, Any]]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        similarities = np.dot(self.embeddings, query_embedding)
        
        if len(self.chunks) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0.15:
                results.append({**self.chunks[idx], "score": round(score, 4)})

        return results

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        return self.chunks

    def clear(self):
        self.chunks = []
        self.embeddings = None
        if os.path.exists(STORE_FILE):
            os.remove(STORE_FILE)
        if os.path.exists(EMBEDDINGS_FILE):
            os.remove(EMBEDDINGS_FILE)

    def delete_document(self, filename: str):
        if not self.chunks:
            return
        indices_to_keep = [i for i, c in enumerate(self.chunks) if c["filename"] != filename]
        self.chunks = [self.chunks[i] for i in indices_to_keep]
        if len(self.chunks) == 0:
            self.clear()
        else:
            self.embeddings = self.embeddings[indices_to_keep] if self.embeddings is not None else None
            self.save_index()

    def save_index(self):
        with open(STORE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)
        if self.embeddings is not None:
            np.save(EMBEDDINGS_FILE, self.embeddings)

    def load_index(self):
        if os.path.exists(STORE_FILE) and os.path.exists(EMBEDDINGS_FILE):
            try:
                with open(STORE_FILE, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                self.embeddings = np.load(EMBEDDINGS_FILE)
            except Exception as e:
                print(f"Failed to load file vector store: {e}")
                self.clear()


# this is the embedding based code
class UnifiedVectorStore:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        print(f"[RAG Engine] Loading Embedding Model ({model_name})...")
        # this is the transformer layer based code
        self.embedding_model = SentenceTransformer(model_name)
        
        self.active_backend = "file"
        self.pg_store: Optional[PostgresVectorStore] = None
        self.file_store = FileVectorStore()

        if VECTOR_STORE_BACKEND == "postgres":
            print(f"[RAG Engine] Initializing PostgreSQL Vector Store on {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}...")
            dim = self.embedding_model.get_sentence_embedding_dimension() or 384
            self.pg_store = PostgresVectorStore(embedding_dimension=dim)
            if self.pg_store.is_connected:
                self.active_backend = "postgres"
                print(f"[RAG Engine] Active Vector Store: PostgreSQL ({POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB})")
            else:
                print(f"[RAG Engine] Could not connect to PostgreSQL. Falling back to local File Vector Store.")

    def add_documents(self, new_chunks: List[Dict[str, Any]]):
        if not new_chunks:
            return

        texts = [c["text"] for c in new_chunks]
        # this is where convert each token as numerical vector
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=min(len(texts), 64),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        if self.active_backend == "postgres" and self.pg_store and self.pg_store.is_connected:
            self.pg_store.add_documents(new_chunks, embeddings)
        else:
            self.file_store.add_documents(new_chunks, embeddings)

    # this is where semantic search is working
    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        if self.active_backend == "postgres" and self.pg_store and self.pg_store.is_connected:
            return self.pg_store.search(query_embedding, top_k=top_k)
        else:
            return self.file_store.search(query_embedding, top_k=top_k)

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        if self.active_backend == "postgres" and self.pg_store and self.pg_store.is_connected:
            return self.pg_store.get_all_chunks()
        else:
            return self.file_store.get_all_chunks()

    def delete_document(self, filename: str):
        if self.active_backend == "postgres" and self.pg_store and self.pg_store.is_connected:
            self.pg_store.delete_document(filename)
        else:
            self.file_store.delete_document(filename)

    def clear(self):
        if self.active_backend == "postgres" and self.pg_store and self.pg_store.is_connected:
            self.pg_store.clear()
        else:
            self.file_store.clear()

    def get_status(self) -> Dict[str, Any]:
        chunks = self.get_all_chunks()
        return {
            "backend": self.active_backend,
            "configured_backend": VECTOR_STORE_BACKEND,
            "postgres_connected": bool(self.pg_store and self.pg_store.is_connected),
            "postgres_host": POSTGRES_HOST,
            "postgres_port": POSTGRES_PORT,
            "postgres_db": POSTGRES_DB,
            "total_chunks": len(chunks),
            "unique_documents": len(set(c["filename"] for c in chunks))
        }


# this is the llm based code
class RAGPipeline:
    def __init__(self, vector_store: UnifiedVectorStore):
        self.vector_store = vector_store
        self.http_client = httpx.Client(
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )

    def generate_response(self, query: str, mode: str = "RAG: Document Search") -> Dict[str, Any]:
        sources = []
        if mode == "RAG: Document Search":
            retrieved_chunks = self.vector_store.search(query, top_k=4)

            if not retrieved_chunks:
                context_str = "No indexed documents found or no relevant context returned."
            else:
                context_parts = []
                file_pages: Dict[str, Dict[str, Any]] = {}

                for c in retrieved_chunks:
                    fname = c["filename"]
                    page = c["page"]
                    text = c["text"]
                    ftype = c.get("doc_type", "pdf")

                    context_parts.append(f"--- Chunk from {fname} (Page {page}) ---\n{text}")

                    if fname not in file_pages:
                        file_pages[fname] = {"type": ftype, "pages": set(), "snippets": []}
                    file_pages[fname]["pages"].add(page)
                    file_pages[fname]["snippets"].append(text)

                context_str = "\n\n".join(context_parts)

                for fname, data in file_pages.items():
                    sorted_pages = sorted(data["pages"])
                    page_str = f"Page {sorted_pages[0]}" if len(sorted_pages) == 1 else f"Page {sorted_pages[0]} – {sorted_pages[-1]}"
                    sources.append({
                        "filename": fname,
                        "pages": page_str,
                        "type": data["type"],
                        "snippets": data["snippets"][:2]
                    })

            prompt = f"""You are an Intelligent Knowledge Assistant. Answer the user's question accurately using ONLY the provided Document Context below.

Document Context:
{context_str}

User Question: {query}

Instructions:
- Provide a clear, natural, well-formatted answer in standard paragraphs and bullet points.
- Do NOT output markdown tables, pipe characters ('|'), or table separator lines ('|---|---|').
- If the context contains relevant information, answer thoroughly and clearly.
- If the context does not contain enough information to answer the question, state politely that the uploaded documents do not mention this.
"""
            answer = self._call_llm(prompt, context_str, query)
        else:
            prompt = f"""You are an AI assistant. Answer the user's request clearly and naturally using standard paragraphs or bullet points without markdown tables:

User Request: {query}
"""
            answer = self._call_llm(prompt, "", query)

        return {
            "answer": answer,
            "sources": sources,
            "mode": mode
        }

    def _call_llm(self, prompt: str, context_str: str, query: str) -> str:
        if GROQ_API_KEY:
            try:
                # this is the token probability select next token code
                response = self.http_client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={
                        "model": GROQ_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                )
                data = response.json()
                if "choices" in data:
                    content = data["choices"][0]["message"]["content"]
                    return content.translate(UNICODE_TRANSLATION).strip()
                else:
                    print(f"Groq API error: {data}")
            except Exception as e:
                print(f"Groq API error (falling back to extractive synthesizer): {e}")

        if context_str and "No indexed documents found" not in context_str:
            return (
                f"**Based on the retrieved document context:**\n\n"
                f"{context_str}\n\n"
                f"*(Note: Powered by Python RAG similarity engine & PostgreSQL vector DB. Ensure a valid `GROQ_API_KEY` is set in `.env`.)*"
            )
        else:
            return (
                f"Please upload documents (PDF, DOCX, TXT) via the /api/upload endpoint so I can search and answer questions for: **'{query}'**."
            )

    def close(self):
        self.http_client.close()
