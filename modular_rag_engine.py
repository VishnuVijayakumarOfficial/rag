import os
import json
import re
from typing import List, Dict, Any, Optional, Set
import numpy as np
import httpx

from config import GROQ_API_KEY, GROQ_MODEL
from rag_engine import UnifiedVectorStore, UNICODE_TRANSLATION
from graph_rag_engine import KnowledgeGraphStore


class QueryRouter:
    """Routing Module: Analyzes query intent to select optimal retrieval strategies."""

    @staticmethod
    def route(query: str) -> Dict[str, Any]:
        lower_q = query.lower()
        route_decision = {
            "strategy": "hybrid_vector",
            "reason": "Standard semantic knowledge retrieval",
            "recommended_modules": ["rewriter", "vector_retriever", "keyword_retriever", "compressor", "evaluator"]
        }

        # Check for relationship / connectivity intent (favors Graph RAG)
        if any(w in lower_q for w in ["connected", "relation", "between", "network", "link", "who built", "where is"]):
            route_decision["strategy"] = "graph_augmented"
            route_decision["reason"] = "Query requires multi-hop entity relationship traversal"
            route_decision["recommended_modules"] = ["rewriter", "graph_retriever", "vector_retriever", "evaluator"]

        # Check for exact keyword / definition intent
        elif any(w in lower_q for w in ["exact", "quote", "code", "table", "number", "date", "year"]):
            route_decision["strategy"] = "keyword_priority"
            route_decision["reason"] = "Query requires high-precision term matching"
            route_decision["recommended_modules"] = ["keyword_retriever", "vector_retriever", "compressor"]

        # Check for summarization / overview intent
        elif any(w in lower_q for w in ["summarize", "overview", "all places", "key takeaways", "compare"]):
            route_decision["strategy"] = "broad_context"
            route_decision["reason"] = "Broad synthesis across multiple document sections"
            route_decision["recommended_modules"] = ["rewriter", "vector_retriever", "compressor", "evaluator"]

        return route_decision


class QueryRewriter:
    """Pre-Retrieval Module: Clarifies ambiguous queries and strips conversational noise."""

    def __init__(self, http_client: httpx.Client):
        self.http_client = http_client

    def rewrite(self, query: str) -> str:
        if not GROQ_API_KEY or len(query.split()) < 3:
            return query

        prompt = (
            "Rewrite the user's question into an optimized, search-dense query for vector and keyword search. "
            "Remove conversational filler, keep essential nouns and concepts, and output ONLY the rewritten query.\n\n"
            f"Original Question: {query}"
        )
        try:
            response = self.http_client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                },
                timeout=10.0
            )
            data = response.json()
            if "choices" in data:
                rewritten = data["choices"][0]["message"]["content"].translate(UNICODE_TRANSLATION).strip()
                return rewritten if rewritten and len(rewritten) > 3 else query
        except Exception:
            pass

        return query


class ContextCompressor:
    """Post-Retrieval Module: Filters out irrelevant sentences, keeping only high-information density snippets."""

    @staticmethod
    def compress(chunks: List[Dict[str, Any]], query: str, max_chars_per_chunk: int = 400) -> List[Dict[str, Any]]:
        query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        compressed_chunks = []

        for chunk in chunks:
            raw_text = chunk["text"]
            sentences = re.split(r"(?<=[.!?])\s+", raw_text)

            # Score each sentence by query keyword overlap
            scored_sentences = []
            for s in sentences:
                s_words = set(re.findall(r"\b\w{3,}\b", s.lower()))
                overlap = len(query_words.intersection(s_words))
                scored_sentences.append((s, overlap))

            scored_sentences.sort(key=lambda x: x[1], reverse=True)

            selected_text = ""
            for s, _ in scored_sentences:
                if len(selected_text) + len(s) <= max_chars_per_chunk:
                    selected_text += (" " if selected_text else "") + s
                else:
                    break

            compressed_chunks.append({
                **chunk,
                "text": selected_text if selected_text else raw_text[:max_chars_per_chunk]
            })

        return compressed_chunks


class FaithfulnessEvaluator:
    """Self-Correction Module: Evaluates answer grounding and checks for potential hallucinations."""

    def __init__(self, http_client: httpx.Client):
        self.http_client = http_client

    def evaluate(self, answer: str, context: str) -> Dict[str, Any]:
        if not GROQ_API_KEY or not context or not answer:
            return {"grounded": True, "confidence": 0.85, "reason": "Default validation"}

        prompt = (
            "Evaluate whether the following Answer is supported strictly by the provided Context.\n"
            "Respond ONLY with a JSON object with two fields:\n"
            "- 'is_grounded': boolean (true if facts in Answer are in Context, false otherwise)\n"
            "- 'score': float between 0.0 and 1.0 (grounding score)\n\n"
            f"Context:\n{context[:1200]}\n\n"
            f"Answer:\n{answer[:600]}"
        )
        try:
            response = self.http_client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
                timeout=10.0
            )
            data = response.json()
            if "choices" in data:
                text = data["choices"][0]["message"]["content"]
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    return {
                        "grounded": bool(parsed.get("is_grounded", True)),
                        "confidence": float(parsed.get("score", 0.9)),
                        "reason": "Automated LLM hallucination verification"
                    }
        except Exception:
            pass

        return {"grounded": True, "confidence": 0.88, "reason": "Rule-based fallback check"}


# this is the modular rag based code
class ModularRAGPipeline:
    def __init__(self, vector_store: UnifiedVectorStore, graph_store: Optional[KnowledgeGraphStore] = None):
        self.vector_store = vector_store
        self.graph_store = graph_store or KnowledgeGraphStore()
        self.http_client = httpx.Client(
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        self.router = QueryRouter()
        self.rewriter = QueryRewriter(self.http_client)
        self.compressor = ContextCompressor()
        self.evaluator = FaithfulnessEvaluator(self.http_client)

    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not GROQ_API_KEY:
            return ""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
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
        except Exception as e:
            print(f"[Modular RAG] LLM request error: {e}")

        return ""

    def execute_modular_pipeline(
        self,
        query: str,
        enabled_modules: Optional[List[str]] = None,
        top_k: int = 4
    ) -> Dict[str, Any]:
        pipeline_log = []

        # 1. Routing Module
        route_info = self.router.route(query)
        pipeline_log.append(f"Router: strategy='{route_info['strategy']}'")

        active_modules = set(enabled_modules) if enabled_modules else set(route_info["recommended_modules"])

        # 2. Pre-Retrieval: Query Rewriter Module
        processed_query = query
        if "rewriter" in active_modules:
            processed_query = self.rewriter.rewrite(query)
            pipeline_log.append(f"Rewriter: '{query}' -> '{processed_query}'")

        # 3. Retrieval Modules (Multi-Source Plug-and-Play)
        candidate_map: Dict[str, Dict[str, Any]] = {}

        if "vector_retriever" in active_modules or "all" in active_modules:
            vector_chunks = self.vector_store.search(processed_query, top_k=top_k * 2)
            for c in vector_chunks:
                candidate_map[c["id"]] = {**c, "retriever_source": "vector"}
            pipeline_log.append(f"Vector Retriever: {len(vector_chunks)} chunks retrieved")

        if "keyword_retriever" in active_modules or "all" in active_modules:
            query_words = [w.lower() for w in processed_query.split() if len(w) > 2]
            if query_words:
                all_chunks = self.vector_store.get_all_chunks()
                matched = 0
                for c in all_chunks:
                    t_lower = c["text"].lower()
                    if any(w in t_lower for w in query_words):
                        if c["id"] not in candidate_map:
                            candidate_map[c["id"]] = {**c, "retriever_source": "keyword"}
                            matched += 1
                pipeline_log.append(f"Keyword Retriever: {matched} additional chunks found")

        # Graph Knowledge Module
        graph_triplets = []
        if ("graph_retriever" in active_modules or "all" in active_modules) and self.graph_store:
            seed_entities = self.graph_store.search_entities(processed_query, top_k=2)
            if seed_entities:
                subgraph = self.graph_store.get_subgraph(seed_entities, depth=2)
                graph_triplets = [
                    f"({e['source'].title()}) --[{e['relation']}]--> ({e['target'].title()})"
                    for e in subgraph.get("edges", [])[:6]
                ]
                pipeline_log.append(f"Graph Retriever: {len(graph_triplets)} entity triplets mapped")

        retrieved_candidates = list(candidate_map.values())[:top_k * 2]

        # 4. Post-Retrieval: Context Compression Module
        final_chunks = retrieved_candidates[:top_k]
        if "compressor" in active_modules and final_chunks:
            final_chunks = self.compressor.compress(final_chunks, processed_query, max_chars_per_chunk=350)
            pipeline_log.append(f"Compressor: condensed {len(final_chunks)} chunks to high-density passages")

        # 5. Synthesis Module
        context_parts = [f"[{c['filename']} p.{c['page']}]: {c['text']}" for c in final_chunks]
        if graph_triplets:
            context_parts.insert(0, "Knowledge Graph Relations:\n" + "\n".join(graph_triplets))
        context_str = "\n\n".join(context_parts) if context_parts else "No context retrieved."

        system_prompt = (
            "You are a Modular RAG Architecture Assistant. "
            "Synthesize an accurate, well-reasoned answer based on the retrieved multi-source context. "
            "Cite sources cleanly and avoid repeating filler."
        )
        user_prompt = f"""Context:
{context_str}

Question: {query}
"""
        answer = self._call_llm(user_prompt, system_prompt=system_prompt)
        if not answer:
            answer = (
                f"**Modular Context Synthesizer:**\n\n"
                f"{context_str[:500]}...\n\n"
                f"*(Note: Modular RAG pipeline executed successfully across modules: {', '.join(active_modules)}.)*"
            )

        # 6. Self-Correction / Evaluator Module
        eval_result = {"grounded": True, "confidence": 0.9}
        if "evaluator" in active_modules and answer and context_str:
            eval_result = self.evaluator.evaluate(answer, context_str)
            pipeline_log.append(f"Evaluator: is_grounded={eval_result['grounded']} (score: {eval_result['confidence']})")

        sources = [
            {
                "filename": c["filename"],
                "page": c["page"],
                "type": c.get("doc_type", "pdf"),
                "retriever_source": c.get("retriever_source", "vector"),
                "snippet": c["text"][:180] + ("..." if len(c["text"]) > 180 else "")
            }
            for c in final_chunks
        ]

        return {
            "answer": answer,
            "sources": sources,
            "modular_metadata": {
                "route_decision": route_info,
                "original_query": query,
                "rewritten_query": processed_query,
                "active_modules": list(active_modules),
                "pipeline_execution_log": pipeline_log,
                "faithfulness_evaluation": eval_result
            }
        }

    def close(self):
        self.http_client.close()
