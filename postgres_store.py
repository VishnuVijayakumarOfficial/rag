from typing import List, Dict, Any, Optional
import numpy as np
import psycopg2
from psycopg2.extras import execute_values


from config import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD
)


class PostgresVectorStore:
    """PostgreSQL Vector Store supporting pgvector extension and SQL cosine similarity."""

    def __init__(self, embedding_dimension: int = 384):
        self.dim = embedding_dimension
        self.is_connected = False
        self.has_pgvector = False
        self._init_db()

    def _get_server_connection(self):
        """Connects to root postgres database to ensure target DB exists."""
        return psycopg2.connect(
            dbname="postgres",
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            connect_timeout=3
        )

    def _get_db_connection(self):
        """Connects to the application RAG database."""
        return psycopg2.connect(
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            connect_timeout=3
        )

    def _init_db(self):
        try:
            # 1. Ensure target database exists
            server_conn = self._get_server_connection()
            server_conn.autocommit = True
            with server_conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (POSTGRES_DB,))
                if not cur.fetchone():
                    print(f"[Postgres] Creating database '{POSTGRES_DB}'...")
                    cur.execute(f'CREATE DATABASE "{POSTGRES_DB}";')
            server_conn.close()

            # 2. Connect to RAG DB and initialize tables / pgvector
            conn = self._get_db_connection()
            conn.autocommit = True
            with conn.cursor() as cur:
                # Try creating pgvector extension
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    self.has_pgvector = True
                    print("[Postgres] pgvector extension initialized.")
                except Exception as ext_err:
                    print(f"[Postgres] pgvector extension not installed; falling back to standard SQL vector array: {ext_err}")
                    self.has_pgvector = False

                # Create document_chunks table
                if self.has_pgvector:
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS document_chunks (
                            id VARCHAR(255) PRIMARY KEY,
                            filename VARCHAR(255) NOT NULL,
                            page INT NOT NULL,
                            doc_type VARCHAR(50) NOT NULL,
                            chunk_text TEXT NOT NULL,
                            embedding vector({self.dim}),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                        CREATE INDEX IF NOT EXISTS idx_chunks_filename ON document_chunks(filename);
                    """)
                else:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS document_chunks (
                            id VARCHAR(255) PRIMARY KEY,
                            filename VARCHAR(255) NOT NULL,
                            page INT NOT NULL,
                            doc_type VARCHAR(50) NOT NULL,
                            chunk_text TEXT NOT NULL,
                            embedding float8[] NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                        CREATE INDEX IF NOT EXISTS idx_chunks_filename ON document_chunks(filename);
                    """)

            conn.close()
            self.is_connected = True
            print(f"[Postgres] Connected to PostgreSQL database '{POSTGRES_DB}' successfully!")

        except Exception as e:
            self.is_connected = False
            print(f"[Postgres] Failed to initialize PostgreSQL store: {e}")

    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        if not self.is_connected or not chunks:
            return

        conn = self._get_db_connection()
        conn.autocommit = True
        try:
            records = []
            for chunk, emb in zip(chunks, embeddings):
                emb_val = emb.tolist() if isinstance(emb, np.ndarray) else list(emb)
                if self.has_pgvector:
                    # pgvector expects string representation like '[0.1, 0.2, ...]'
                    emb_repr = str(emb_val)
                else:
                    emb_repr = emb_val

                records.append((
                    chunk["id"],
                    chunk["filename"],
                    chunk["page"],
                    chunk.get("doc_type", "pdf"),
                    chunk["text"],
                    emb_repr
                ))

            with conn.cursor() as cur:
                query = """
                    INSERT INTO document_chunks (id, filename, page, doc_type, chunk_text, embedding)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        chunk_text = EXCLUDED.chunk_text,
                        embedding = EXCLUDED.embedding;
                """
                execute_values(cur, query, records)

            print(f"[Postgres] Inserted/Updated {len(records)} chunks in PostgreSQL.")
        finally:
            conn.close()

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Dict[str, Any]]:
        if not self.is_connected:
            return []

        conn = self._get_db_connection()
        query_emb_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else list(query_embedding)
        results = []

        try:
            with conn.cursor() as cur:
                if self.has_pgvector:
                    emb_str = str(query_emb_list)
                    cur.execute("""
                        SELECT id, filename, page, doc_type, chunk_text,
                               1 - (embedding <=> %s::vector) AS similarity
                        FROM document_chunks
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                    """, (emb_str, emb_str, top_k))
                    rows = cur.fetchall()
                    for r in rows:
                        score = float(r[5])
                        if score > 0.15:
                            results.append({
                                "id": r[0],
                                "filename": r[1],
                                "page": r[2],
                                "doc_type": r[3],
                                "text": r[4],
                                "score": round(score, 4)
                            })
                else:
                    # Pure SQL / Numpy fallback for cosine similarity
                    cur.execute("SELECT id, filename, page, doc_type, chunk_text, embedding FROM document_chunks;")
                    rows = cur.fetchall()
                    if rows:
                        db_embeddings = np.array([r[5] for r in rows])
                        query_vec = np.array(query_emb_list)
                        
                        # Compute cosine similarity
                        sims = np.dot(db_embeddings, query_vec) / (
                            np.linalg.norm(db_embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-10
                        )
                        top_indices = np.argsort(sims)[::-1][:top_k]

                        for idx in top_indices:
                            score = float(sims[idx])
                            if score > 0.15:
                                r = rows[idx]
                                results.append({
                                    "id": r[0],
                                    "filename": r[1],
                                    "page": r[2],
                                    "doc_type": r[3],
                                    "text": r[4],
                                    "score": round(score, 4)
                                })
        finally:
            conn.close()

        return results

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        if not self.is_connected:
            return []

        conn = self._get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id, filename, page, doc_type, chunk_text FROM document_chunks;")
                rows = cur.fetchall()
                return [{
                    "id": r[0],
                    "filename": r[1],
                    "page": r[2],
                    "doc_type": r[3],
                    "text": r[4]
                } for r in rows]
        finally:
            conn.close()

    def delete_document(self, filename: str):
        if not self.is_connected:
            return

        conn = self._get_db_connection()
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_chunks WHERE filename = %s;", (filename,))
        finally:
            conn.close()

    def clear(self):
        if not self.is_connected:
            return

        conn = self._get_db_connection()
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE document_chunks;")
        finally:
            conn.close()
