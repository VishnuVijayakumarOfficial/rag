from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import numpy as np
import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values

from config import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    POSTGRES_MIN_POOL,
    POSTGRES_MAX_POOL
)


class PostgresVectorStore:
    def __init__(self, embedding_dimension: int = 384):
        self.dim = embedding_dimension
        self.is_connected = False
        self.has_pgvector = False
        self.pool: Optional[pool.ThreadedConnectionPool] = None
        self._init_db()

    @contextmanager
    def _get_connection(self):
        if not self.pool:
            raise ConnectionError("Postgres connection pool is not initialized.")
        conn = self.pool.getconn()
        try:
            conn.autocommit = True
            yield conn
        finally:
            self.pool.putconn(conn)

    def _init_db(self):
        try:
            server_conn = psycopg2.connect(
                dbname="postgres",
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                connect_timeout=3
            )
            server_conn.autocommit = True
            with server_conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (POSTGRES_DB,))
                if not cur.fetchone():
                    print(f"[Postgres] Creating database '{POSTGRES_DB}'...")
                    cur.execute(f'CREATE DATABASE "{POSTGRES_DB}";')
            server_conn.close()

            self.pool = pool.ThreadedConnectionPool(
                minconn=POSTGRES_MIN_POOL,
                maxconn=POSTGRES_MAX_POOL,
                dbname=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                connect_timeout=3
            )

            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    try:
                        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                        self.has_pgvector = True
                        print("[Postgres] pgvector extension initialized.")
                    except Exception as ext_err:
                        print(f"[Postgres] pgvector extension not installed; falling back to standard SQL vector array: {ext_err}")
                        self.has_pgvector = False

                    emb_type = f"vector({self.dim})" if self.has_pgvector else "float8[]"
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS document_chunks (
                            id VARCHAR(255) PRIMARY KEY,
                            filename VARCHAR(255) NOT NULL,
                            page INT NOT NULL,
                            doc_type VARCHAR(50) NOT NULL,
                            chunk_text TEXT NOT NULL,
                            embedding {emb_type},
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                        CREATE INDEX IF NOT EXISTS idx_chunks_filename ON document_chunks(filename);
                    """)

                    if self.has_pgvector:
                        try:
                            cur.execute("""
                                CREATE INDEX IF NOT EXISTS idx_chunks_hnsw 
                                ON document_chunks USING hnsw (embedding vector_cosine_ops);
                            """)
                        except Exception as idx_err:
                            print(f"[Postgres] Note: HNSW index creation skipped: {idx_err}")

            self.is_connected = True
            print(f"[Postgres] Connected to PostgreSQL pool ('{POSTGRES_DB}') successfully!")

        except Exception as e:
            self.is_connected = False
            print(f"[Postgres] Failed to initialize PostgreSQL store: {e}")

    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        if not self.is_connected or not chunks:
            return

        emb_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        records = [
            (
                chunk["id"],
                chunk["filename"],
                chunk["page"],
                chunk.get("doc_type", "pdf"),
                chunk["text"],
                str(emb) if self.has_pgvector else emb
            )
            for chunk, emb in zip(chunks, emb_list)
        ]

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    INSERT INTO document_chunks (id, filename, page, doc_type, chunk_text, embedding)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        chunk_text = EXCLUDED.chunk_text,
                        embedding = EXCLUDED.embedding;
                """
                execute_values(cur, query, records, page_size=200)

        print(f"[Postgres] Inserted/Updated {len(records)} chunks in PostgreSQL.")

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Dict[str, Any]]:
        if not self.is_connected:
            return []

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if self.has_pgvector:
                    query_emb_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
                    emb_str = str(query_emb_list)
                    cur.execute("""
                        SELECT id, filename, page, doc_type, chunk_text,
                               1 - (embedding <=> %s::vector) AS similarity
                        FROM document_chunks
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                    """, (emb_str, emb_str, top_k))
                    rows = cur.fetchall()
                    return [
                        {
                            "id": cid,
                            "filename": fn,
                            "page": pg,
                            "doc_type": dt,
                            "text": txt,
                            "score": round(float(sim), 4)
                        }
                        for cid, fn, pg, dt, txt, sim in rows
                        if float(sim) > 0.15
                    ]
                else:
                    cur.execute("SELECT id, filename, page, doc_type, chunk_text, embedding FROM document_chunks;")
                    rows = cur.fetchall()
                    if not rows:
                        return []

                    db_embeddings = np.array([r[5] for r in rows])
                    query_vec = np.asarray(query_embedding)
                    
                    sims = np.dot(db_embeddings, query_vec) / (
                        np.linalg.norm(db_embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-10
                    )
                    if len(rows) <= top_k:
                        top_indices = np.argsort(sims)[::-1]
                    else:
                        top_indices = np.argpartition(sims, -top_k)[-top_k:]
                        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

                    results = []
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
                    return results

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        if not self.is_connected:
            return []

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, filename, page, doc_type, chunk_text FROM document_chunks;")
                rows = cur.fetchall()
                return [
                    {
                        "id": cid,
                        "filename": fn,
                        "page": pg,
                        "doc_type": dt,
                        "text": txt
                    }
                    for cid, fn, pg, dt, txt in rows
                ]

    def delete_document(self, filename: str):
        if not self.is_connected:
            return

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_chunks WHERE filename = %s;", (filename,))

    def clear(self):
        if not self.is_connected:
            return

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE document_chunks;")

    def close(self):
        if self.pool:
            self.pool.closeall()
            self.is_connected = False
