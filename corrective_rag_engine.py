import os
import json
import re
from typing import List, Dict, Any, Optional
import numpy as np
import httpx

from config import GROQ_API_KEY, GROQ_MODEL
from rag_engine import UnifiedVectorStore, UNICODE_TRANSLATION


class RetrievalEvaluator:
    """Evaluates the relevance of retrieved document chunks and assigns a corrective action."""

    def __init__(self, http_client: httpx.Client):
        self.http_client = http_client

    def grade_chunk_relevance(self, query: str, chunk_text: str) -> float:
        if not GROQ_API_KEY:
            # Fast heuristic keyword overlap score
            q_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
            c_words = set(re.findall(r"\b\w{3,}\b", chunk_text.lower()))
            if not q_words:
                return 0.5
            return min(len(q_words.intersection(c_words)) / len(q_words), 1.0)

        prompt = (
            "You are a Retrieval Quality Evaluator. Assess if the following document chunk contains information "
            "relevant and useful to answer the user's question.\n"
            "Return ONLY a JSON object with one key 'score' between 0.0 (completely irrelevant) and 1.0 (highly relevant and accurate).\n\n"
            f"Question: {query}\n\n"
            f"Document Chunk:\n{chunk_text[:800]}"
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
                    return float(parsed.get("score", 0.5))
        except Exception:
            pass

        return 0.5

    def determine_action(
        self,
        chunk_scores: List[float],
        correct_threshold: float = 0.65,
        incorrect_threshold: float = 0.35
    ) -> Dict[str, Any]:
        if not chunk_scores:
            return {"action": "INCORRECT", "confidence": 0.0, "reason": "No document chunks retrieved"}

        avg_score = float(np.mean(chunk_scores))
        max_score = float(np.max(chunk_scores))

        if avg_score >= correct_threshold or max_score >= 0.8:
            action = "CORRECT"
            reason = "Retrieved documents are relevant and sufficient to answer the question"
        elif max_score <= incorrect_threshold:
            action = "INCORRECT"
            reason = "Retrieved documents contain irrelevant or contradictory information; triggering fallback"
        else:
            action = "AMBIGUOUS"
            reason = "Retrieved documents are partially relevant; combining refined context with fallback expansion"

        return {
            "action": action,
            "average_score": round(avg_score, 4),
            "max_score": round(max_score, 4),
            "reason": reason
        }


class KnowledgeRefiner:
    """Decomposes chunks into fine-grained knowledge strips, filtering out irrelevant noise."""

    @staticmethod
    def refine_chunks(query: str, chunks: List[Dict[str, Any]], min_overlap: int = 1) -> List[Dict[str, Any]]:
        query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        refined_chunks = []

        for c in chunks:
            raw_text = c["text"]
            sentences = re.split(r"(?<=[.!?])\s+", raw_text)
            relevant_sentences = []

            for s in sentences:
                s_words = set(re.findall(r"\b\w{3,}\b", s.lower()))
                if len(query_words.intersection(s_words)) >= min_overlap:
                    relevant_sentences.append(s.strip())

            refined_text = " ".join(relevant_sentences) if relevant_sentences else raw_text[:300]
            refined_chunks.append({
                **c,
                "text": refined_text,
                "stripped_filler": len(raw_text) - len(refined_text)
            })

        return refined_chunks


# this is the corrective rag based code
class CorrectiveRAGPipeline:
    def __init__(self, vector_store: UnifiedVectorStore):
        self.vector_store = vector_store
        self.http_client = httpx.Client(
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        self.evaluator = RetrievalEvaluator(self.http_client)
        self.refiner = KnowledgeRefiner()

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
            print(f"[Corrective RAG] LLM request error: {e}")

        return ""

    def _generate_fallback_knowledge(self, query: str) -> str:
        prompt = (
            f"You are a factual knowledge generator. Provide a concise, highly accurate factual overview "
            f"to answer this question, as retrieved documents were found irrelevant or ambiguous:\n\n"
            f"Question: {query}"
        )
        return self._call_llm(prompt)

    def generate_corrective_response(
        self,
        query: str,
        correct_threshold: float = 0.65,
        incorrect_threshold: float = 0.35,
        top_k: int = 4
    ) -> Dict[str, Any]:
        correction_log = []

        # 1. Initial Retrieval
        raw_chunks = self.vector_store.search(query, top_k=top_k)
        correction_log.append(f"Initial Retrieval: {len(raw_chunks)} candidate chunks fetched")

        # 2. Retrieval Evaluation & Confidence Grading
        graded_chunks = []
        chunk_scores = []
        for c in raw_chunks:
            score = self.evaluator.grade_chunk_relevance(query, c["text"])
            chunk_scores.append(score)
            graded_chunks.append({**c, "eval_score": round(score, 4)})

        decision = self.evaluator.determine_action(
            chunk_scores,
            correct_threshold=correct_threshold,
            incorrect_threshold=incorrect_threshold
        )
        action = decision["action"]
        correction_log.append(f"Evaluator Decision: {action} (Avg score: {decision['average_score']}, Max score: {decision['max_score']}) - {decision['reason']}")

        # 3. Action-based Knowledge Processing
        sources = []
        fallback_knowledge = ""

        # Case A: CORRECT - Document chunks are trustworthy
        if action == "CORRECT":
            refined_chunks = self.refiner.refine_chunks(query, [c for c in graded_chunks if c["eval_score"] >= 0.4])
            context_blocks = [f"--- [{c['filename']} (Page {c['page']})] ---\n{c['text']}" for c in refined_chunks]
            context_str = "\n\n".join(context_blocks)
            sources = refined_chunks
            correction_log.append("Action executed: Refined and stripped non-essential filler from verified documents")

        # Case B: INCORRECT - Document chunks are irrelevant/misleading
        elif action == "INCORRECT":
            correction_log.append("Action executed: Discarded irrelevant document context; triggered corrective knowledge generation")
            fallback_knowledge = self._generate_fallback_knowledge(query)
            context_str = f"Corrective Knowledge Base Synthesizer:\n{fallback_knowledge}"

        # Case C: AMBIGUOUS - Document chunks are partially relevant
        else:
            correction_log.append("Action executed: Partially relevant context; combining refined chunks with corrective knowledge")
            refined_chunks = self.refiner.refine_chunks(query, [c for c in graded_chunks if c["eval_score"] >= 0.3])
            doc_context = "\n\n".join([f"--- [{c['filename']} (Page {c['page']})] ---\n{c['text']}" for c in refined_chunks])
            fallback_knowledge = self._generate_fallback_knowledge(query)
            context_str = f"Document Evidence:\n{doc_context}\n\nSupplementary Knowledge:\n{fallback_knowledge}"
            sources = refined_chunks

        # 4. Corrective Grounded Generation
        system_prompt = (
            "You are a Corrective RAG (CRAG) Intelligence Assistant. "
            "Synthesize an accurate, reliable answer strictly grounded in the provided context. "
            "If document chunks were evaluated as incorrect or ambiguous, make sure the corrected knowledge is clearly explained. "
            "Do not output markdown tables or fabricate unsupported assertions."
        )

        user_prompt = f"""Context ({action} Verified):
{context_str}

Question: {query}

Instructions:
- Provide a clear, natural, well-formatted answer in standard paragraphs and bullet points.
- Explain the findings clearly and accurately.
"""

        answer = self._call_llm(user_prompt, system_prompt=system_prompt)
        if not answer:
            answer = (
                f"**Corrective Context Output ({action}):**\n\n"
                f"{context_str[:500]}...\n\n"
                f"*(Note: Corrective RAG evaluated retrieval as {action}. Configure `GROQ_API_KEY` in `.env` for LLM generation.)*"
            )

        formatted_sources = [
            {
                "filename": c["filename"],
                "page": c["page"],
                "type": c.get("doc_type", "pdf"),
                "eval_score": c.get("eval_score", 0.0),
                "snippet": c["text"][:200] + ("..." if len(c["text"]) > 200 else "")
            }
            for c in sources
        ]

        return {
            "answer": answer,
            "sources": formatted_sources,
            "corrective_metadata": {
                "decision": decision,
                "action": action,
                "correction_log": correction_log,
                "chunks_evaluated": len(graded_chunks),
                "chunks_used": len(sources),
                "fallback_triggered": bool(fallback_knowledge)
            }
        }

    def close(self):
        self.http_client.close()
