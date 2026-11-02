import os
import json
import base64
import re
from typing import List, Dict, Any, Optional
import httpx

from config import GROQ_API_KEY, GROQ_MODEL
from rag_engine import UnifiedVectorStore, UNICODE_TRANSLATION, DATA_DIR

ASSETS_DIR = os.path.join(DATA_DIR, "multimodal_assets")
ASSETS_INDEX_FILE = os.path.join(DATA_DIR, "multimodal_index.json")
os.makedirs(ASSETS_DIR, exist_ok=True)


class MultimodalAssetStore:
    """Stores and indexes visual assets (images, figures, charts, tables) extracted from documents."""

    def __init__(self):
        self.assets: List[Dict[str, Any]] = []
        self.load_index()

    def add_asset(
        self,
        asset_id: str,
        filename: str,
        caption: str,
        asset_type: str = "image",
        page: int = 1,
        image_path: Optional[str] = None
    ):
        for a in self.assets:
            if a["id"] == asset_id:
                a.update({"caption": caption, "filename": filename, "page": page, "path": image_path or ""})
                self.save_index()
                return

        self.assets.append({
            "id": asset_id,
            "filename": filename,
            "caption": caption,
            "type": asset_type,
            "page": page,
            "path": image_path or ""
        })
        self.save_index()

    def search_assets(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        if not query_words:
            return self.assets[:top_k]

        scored = []
        for a in self.assets:
            cap_words = set(re.findall(r"\b\w{3,}\b", a["caption"].lower()))
            overlap = len(query_words.intersection(cap_words))
            score = overlap + (2 if any(w in a["filename"].lower() for w in query_words) else 0)
            if score > 0:
                scored.append({**a, "visual_score": round(score / len(query_words), 4)})

        scored.sort(key=lambda x: x.get("visual_score", 0), reverse=True)
        return scored[:top_k]

    def save_index(self):
        with open(ASSETS_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(self.assets, f, indent=2)

    def load_index(self):
        if os.path.exists(ASSETS_INDEX_FILE):
            try:
                with open(ASSETS_INDEX_FILE, "r", encoding="utf-8") as f:
                    self.assets = json.load(f)
            except Exception as e:
                print(f"[Multimodal] Failed to load assets index: {e}")


# this is the multimodal rag based code
class MultimodalRAGPipeline:
    def __init__(self, vector_store: UnifiedVectorStore):
        self.vector_store = vector_store
        self.asset_store = MultimodalAssetStore()
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
            print(f"[Multimodal RAG] LLM error: {e}")

        return ""

    def generate_multimodal_response(
        self,
        query: str,
        image_input_base64: Optional[str] = None,
        top_k_text: int = 4,
        top_k_visual: int = 3
    ) -> Dict[str, Any]:
        multimodal_trace = []

        # 1. Text Context Retrieval
        text_chunks = self.vector_store.search(query, top_k=top_k_text)
        multimodal_trace.append(f"Textual Retrieval: {len(text_chunks)} text passages retrieved")

        # 2. Visual Asset Retrieval
        matched_assets = self.asset_store.search_assets(query, top_k=top_k_visual)
        multimodal_trace.append(f"Visual Retrieval: {len(matched_assets)} related figures/images retrieved")

        # 3. Multimodal Context Fusion
        text_context = "\n\n".join([f"[{c['filename']} p.{c['page']}]: {c['text']}" for c in text_chunks])
        visual_context = "\n".join([
            f"- [Visual Asset {a['id']} ({a['type']}) from {a['filename']} p.{a['page']}]: {a['caption']}"
            for a in matched_assets
        ]) if matched_assets else "No direct visual diagrams registered."

        system_prompt = (
            "You are a Multimodal RAG Assistant. "
            "Synthesize your answer using both textual document evidence and the descriptions of visual assets (charts, diagrams, photos). "
            "Cite visual figures (e.g., [Visual Asset X]) where relevant to support your explanations."
        )

        user_prompt = f"""Visual Asset Descriptions:
{visual_context}

Textual Document Context:
{text_context}

User Question: {query}
"""
        answer = self._call_llm(user_prompt, system_prompt=system_prompt)
        if not answer:
            answer = (
                f"**Multimodal Context Output:**\n\n"
                f"{text_context[:400]}...\n\n"
                f"*(Note: Visual assets considered: {', '.join([a['id'] for a in matched_assets]) if matched_assets else 'None'}.)*"
            )

        sources = [
            {"filename": c["filename"], "page": c["page"], "type": "text", "snippet": c["text"][:180] + "..."}
            for c in text_chunks
        ]

        return {
            "answer": answer,
            "text_sources": sources,
            "visual_assets": matched_assets,
            "multimodal_metadata": {
                "trace": multimodal_trace,
                "has_image_query": bool(image_input_base64),
                "text_chunks_count": len(text_chunks),
                "visual_assets_count": len(matched_assets)
            }
        }

    def close(self):
        self.http_client.close()
