import os
import json
import re
from typing import List, Dict, Any, Optional
import httpx

from config import GROQ_API_KEY, GROQ_MODEL
from rag_engine import UnifiedVectorStore, UNICODE_TRANSLATION


# this is the adaptive rag based code
class AdaptiveRAGPipeline:
    """Adaptive RAG Pipeline:
    Dynamically routes queries based on estimated complexity:
    - Low Complexity: Direct Generation (No Retrieval)
    - Medium Complexity: Single-Step Retrieval
    - High Complexity: Multi-Step Sequential Query Decomposition & Context Accumulation
    """

    def __init__(self, vector_store: UnifiedVectorStore):
        self.vector_store = vector_store
        self.http_client = httpx.Client(
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )

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
            print(f"[Adaptive RAG] LLM error: {e}")

        return ""

    def classify_complexity(self, query: str) -> Dict[str, Any]:
        """Classifies query complexity into 'no_retrieval', 'single_step', or 'multi_step'."""
        lower_q = query.lower()

        # Heuristic rules
        if any(w in lower_q for w in ["hi", "hello", "who are you", "what can you do", "calculate", "2+2"]):
            return {"tier": "no_retrieval", "level": "Low", "reason": "Conversational or direct reasoning query"}

        if any(w in lower_q for w in ["compare", "difference between", "how do x and y relate", "chronological", "evolution", "all steps"]):
            return {"tier": "multi_step", "level": "High", "reason": "Multi-hop query requiring sequential decomposition"}

        if GROQ_API_KEY:
            prompt = (
                "Classify the complexity of the following user query for a document RAG system into ONE tier:\n"
                "- 'no_retrieval': Common greetings, math, or pure logical reasoning.\n"
                "- 'single_step': Simple factoid question answerable from a single document passage.\n"
                "- 'multi_step': Complex, multi-part, or comparative question requiring multiple search steps.\n"
                "Respond ONLY with a JSON object: {\"tier\": \"no_retrieval\" | \"single_step\" | \"multi_step\", \"reason\": \"string\"}\n\n"
                f"Question: {query}"
            )
            response = self._call_llm(prompt)
            try:
                match = re.search(r"\{.*\}", response, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    tier = data.get("tier", "single_step")
                    return {
                        "tier": tier,
                        "level": "High" if tier == "multi_step" else ("Medium" if tier == "single_step" else "Low"),
                        "reason": data.get("reason", "LLM complexity classification")
                    }
            except Exception:
                pass

        return {"tier": "single_step", "level": "Medium", "reason": "Default single-step document retrieval"}

    def decompose_query(self, query: str) -> List[str]:
        """Decomposes a high-complexity query into 2 to 3 sequential sub-queries."""
        if not GROQ_API_KEY:
            return [query]

        prompt = (
            "Break down this complex question into 2 or 3 simpler, sequential sub-questions to search "
            "step-by-step. Return ONLY the sub-questions, one per line, without numbers or bullets.\n\n"
            f"Question: {query}"
        )
        response = self._call_llm(prompt)
        sub_queries = [line.strip().lstrip("1234567890.- ") for line in response.splitlines() if line.strip()]
        return sub_queries[:3] if sub_queries else [query]

    def generate_adaptive_response(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        execution_trace = []
        sources = []

        # 1. Complexity Assessment
        complexity = self.classify_complexity(query)
        tier = complexity["tier"]
        execution_trace.append(f"Query Classification: {tier.upper()} (Complexity: {complexity['level']}) - {complexity['reason']}")

        # Tier 1: No Retrieval
        if tier == "no_retrieval":
            system_prompt = "You are an intelligent AI assistant. Provide a direct, helpful, and natural answer."
            answer = self._call_llm(query, system_prompt=system_prompt)
            if not answer:
                answer = "Hello! I am ready to answer your questions and search your uploaded documents."
            execution_trace.append("Execution: Directly generated answer from parametric memory without document search")
            return {
                "answer": answer,
                "sources": [],
                "adaptive_metadata": {
                    "complexity": complexity,
                    "execution_strategy": tier,
                    "sub_queries_executed": [],
                    "execution_trace": execution_trace
                }
            }

        # Tier 2: Single-Step Retrieval
        elif tier == "single_step":
            chunks = self.vector_store.search(query, top_k=top_k)
            sources = chunks
            execution_trace.append(f"Execution: Single-step vector retrieval fetched {len(chunks)} candidate chunks")

            context_str = "\n\n".join([f"[{c['filename']} p.{c['page']}]: {c['text']}" for c in chunks])
            system_prompt = "You are an Adaptive RAG Knowledge Assistant. Answer the question accurately using ONLY the provided Document Context."
            user_prompt = f"Document Context:\n{context_str}\n\nQuestion: {query}"

            answer = self._call_llm(user_prompt, system_prompt=system_prompt)
            if not answer:
                answer = f"**Single-Step Retrieved Context:**\n\n{context_str[:400]}..."

            return {
                "answer": answer,
                "sources": [
                    {"filename": c["filename"], "page": c["page"], "snippet": c["text"][:180] + "..."}
                    for c in sources
                ],
                "adaptive_metadata": {
                    "complexity": complexity,
                    "execution_strategy": tier,
                    "sub_queries_executed": [query],
                    "execution_trace": execution_trace
                }
            }

        # Tier 3: Multi-Step Iterative Retrieval
        else:
            sub_queries = self.decompose_query(query)
            execution_trace.append(f"Decomposition: Generated {len(sub_queries)} sub-queries: {sub_queries}")

            accumulated_chunks: Dict[str, Dict[str, Any]] = {}
            for step_idx, sub_q in enumerate(sub_queries, start=1):
                step_chunks = self.vector_store.search(sub_q, top_k=2)
                for c in step_chunks:
                    if c["id"] not in accumulated_chunks:
                        accumulated_chunks[c["id"]] = c
                execution_trace.append(f"Step {step_idx}: Sub-query '{sub_q}' -> {len(step_chunks)} chunks retrieved")

            all_retrieved = list(accumulated_chunks.values())
            sources = all_retrieved[:top_k]

            context_str = "\n\n".join([f"[{c['filename']} p.{c['page']}]: {c['text']}" for c in sources])
            system_prompt = (
                "You are an Adaptive RAG Multi-Hop Synthesis Agent. "
                "Synthesize a unified, comprehensive answer by connecting findings across the sequential search steps."
            )
            user_prompt = f"Multi-Step Context:\n{context_str}\n\nComplex Question: {query}"

            answer = self._call_llm(user_prompt, system_prompt=system_prompt)
            if not answer:
                answer = f"**Multi-Step Accumulated Context:**\n\n{context_str[:400]}..."

            return {
                "answer": answer,
                "sources": [
                    {"filename": c["filename"], "page": c["page"], "snippet": c["text"][:180] + "..."}
                    for c in sources
                ],
                "adaptive_metadata": {
                    "complexity": complexity,
                    "execution_strategy": tier,
                    "sub_queries_executed": sub_queries,
                    "execution_trace": execution_trace
                }
            }

    def close(self):
        self.http_client.close()
