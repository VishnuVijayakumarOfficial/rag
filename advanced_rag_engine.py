from typing import List, Dict, Any, Optional
import numpy as np
import httpx

from config import GROQ_API_KEY, GROQ_MODEL
from rag_engine import UnifiedVectorStore, UNICODE_TRANSLATION


# this is the advanced rag based code
class AdvancedRAGPipeline:
    def __init__(self, vector_store: UnifiedVectorStore):
        self.vector_store = vector_store
        # this is the transformer layer based code
        self.embedding_model = vector_store.embedding_model
        self.http_client = httpx.Client(
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )

    # this is the llm based code
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not GROQ_API_KEY:
            return ""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            # this is the token probability select next token code
            response = self.http_client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": GROQ_MODEL,
                    "messages": messages,
                    "temperature": 0.2,
                }
            )
            data = response.json()
            if "choices" in data:
                content = data["choices"][0]["message"]["content"]
                return content.translate(UNICODE_TRANSLATION).strip()
            else:
                print(f"[Advanced RAG] LLM error: {data}")
                return ""
        except Exception as e:
            print(f"[Advanced RAG] LLM request failed: {e}")
            return ""

    def _generate_multi_queries(self, query: str) -> List[str]:
        prompt = (
            f"You are an AI search assistant. Generate 2 alternative search queries that rephrase "
            f"or explore different perspectives of the user question below. "
            f"Return ONLY the 2 queries, each on a new line, without numbers, bullets, or explanations.\n\n"
            f"Original Question: {query}"
        )
        response = self._call_llm(prompt)
        if not response:
            return []

        queries = [line.strip().lstrip("1234567890.- ") for line in response.splitlines() if line.strip()]
        return queries[:2]

    def _generate_hyde_passage(self, query: str) -> str:
        prompt = (
            f"Write a concise, factual passage that hypothetical documents would contain to answer "
            f"this question accurately. Do not include introductory text, just write the passage directly:\n\n"
            f"Question: {query}"
        )
        return self._call_llm(prompt)

    def _keyword_search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        words = [w.lower() for w in query.split() if len(w) > 2]
        if not words:
            return []

        all_chunks = self.vector_store.get_all_chunks()
        scored = []
        for c in all_chunks:
            text_lower = c["text"].lower()
            matches = sum(1 for w in words if w in text_lower)
            if matches > 0:
                scored.append({
                    **c,
                    "keyword_score": round(matches / len(words), 4)
                })

        scored.sort(key=lambda x: x["keyword_score"], reverse=True)
        return scored[:top_k]

    # this is the vector search + keyword search code
    def hybrid_search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        vector_results = self.vector_store.search(query, top_k=top_k * 2)
        keyword_results = self._keyword_search(query, top_k=top_k * 2)

        candidate_map: Dict[str, Dict[str, Any]] = {}
        for c in vector_results:
            candidate_map[c["id"]] = c
        for c in keyword_results:
            if c["id"] not in candidate_map:
                candidate_map[c["id"]] = c

        return self._rerank_chunks(query, list(candidate_map.values()), top_k=top_k)

    def _rerank_chunks(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        query_emb = self.embedding_model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        texts = [c["text"] for c in candidates]
        # this is where convert each token as numerical vector
        chunk_embs = self.embedding_model.encode(
            texts,
            batch_size=min(len(texts), 64),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        similarities = np.dot(chunk_embs, query_emb)

        scored_candidates = []
        for chunk, score in zip(candidates, similarities):
            score_val = float(score)
            if score_val > 0.15:
                scored_candidates.append({
                    **chunk,
                    "relevance_score": round(score_val, 4)
                })

        scored_candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_candidates[:top_k]

    def generate_advanced_response(
        self,
        query: str,
        use_hyde: bool = True,
        use_multi_query: bool = True,
        use_keyword_search: bool = True,
        top_k: int = 4
    ) -> Dict[str, Any]:
        retrieval_queries = [query]
        expanded_queries = []
        hyde_passage = ""

        if use_multi_query and GROQ_API_KEY:
            expanded_queries = self._generate_multi_queries(query)
            retrieval_queries.extend(expanded_queries)

        if use_hyde and GROQ_API_KEY:
            hyde_passage = self._generate_hyde_passage(query)
            if hyde_passage:
                retrieval_queries.append(hyde_passage)

        # this is the vector search + keyword search code
        candidate_map: Dict[str, Dict[str, Any]] = {}
        for q in retrieval_queries:
            # this is where semantic search is working
            chunks = self.vector_store.search(q, top_k=top_k * 2)
            for c in chunks:
                if c["id"] not in candidate_map:
                    candidate_map[c["id"]] = c

        keyword_matches_count = 0
        if use_keyword_search:
            keyword_chunks = self._keyword_search(query, top_k=top_k * 2)
            keyword_matches_count = len(keyword_chunks)
            for c in keyword_chunks:
                if c["id"] not in candidate_map:
                    candidate_map[c["id"]] = c

        all_candidates = list(candidate_map.values())
        reranked_chunks = self._rerank_chunks(query, all_candidates, top_k=top_k)

        if not reranked_chunks:
            return {
                "answer": f"No relevant information found in the indexed documents for: **'{query}'**.",
                "sources": [],
                "advanced_metadata": {
                    "expanded_queries": expanded_queries,
                    "hyde_applied": bool(hyde_passage),
                    "keyword_search_applied": use_keyword_search,
                    "keyword_matches_found": keyword_matches_count,
                    "candidates_evaluated": len(all_candidates),
                    "reranked_selected": 0,
                    "average_confidence": 0.0
                }
            }

        context_blocks = []
        sources = []
        for idx, chunk in enumerate(reranked_chunks, start=1):
            source_tag = f"[Source {idx}: {chunk['filename']} (Page {chunk['page']})]"
            context_blocks.append(f"{source_tag}\n{chunk['text']}")
            sources.append({
                "source_id": idx,
                "filename": chunk["filename"],
                "page": chunk["page"],
                "doc_type": chunk.get("doc_type", "pdf"),
                "relevance_score": chunk.get("relevance_score", 0.0),
                "snippet": chunk["text"][:200] + ("..." if len(chunk["text"]) > 200 else "")
            })

        context_str = "\n\n".join(context_blocks)
        avg_confidence = round(
            sum(s["relevance_score"] for s in sources) / len(sources), 4
        ) if sources else 0.0

        system_prompt = (
            "You are an Advanced RAG Intelligence Agent. "
            "Synthesize comprehensive, logically structured answers based strictly on the provided document sources. "
            "Cite sources in your response using bracketed tags like [Source 1], [Source 2] where appropriate. "
            "Do not output markdown tables or fabricate unsupported facts."
        )

        user_prompt = f"""Document Context:
{context_str}

User Question: {query}

Instructions:
- Provide an in-depth, clear, well-structured answer citing the relevant [Source X] references.
- Explain reasoning clearly using standard paragraphs and bullet points.
- If the sources do not provide sufficient evidence, acknowledge the limitation directly.
"""

        answer = self._call_llm(user_prompt, system_prompt=system_prompt)
        if not answer:
            answer = (
                f"**Based on the top re-ranked document context ({sources[0]['filename']} - Page {sources[0]['page']}):**\n\n"
                f"{reranked_chunks[0]['text']}\n\n"
                f"*(Note: Advanced RAG retrieval complete. Ensure a valid `GROQ_API_KEY` is configured in `.env` for LLM generation.)*"
            )

        return {
            "answer": answer,
            "sources": sources,
            "advanced_metadata": {
                "expanded_queries": expanded_queries,
                "hyde_applied": bool(hyde_passage),
                "keyword_search_applied": use_keyword_search,
                "keyword_matches_found": keyword_matches_count,
                "candidates_evaluated": len(all_candidates),
                "reranked_selected": len(reranked_chunks),
                "average_confidence": avg_confidence
            }
        }

    def close(self):
        self.http_client.close()
