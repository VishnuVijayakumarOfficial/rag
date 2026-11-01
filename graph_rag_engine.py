import os
import json
import re
from typing import List, Dict, Any, Optional, Set
import httpx

from config import GROQ_API_KEY, GROQ_MODEL
from rag_engine import UnifiedVectorStore, UNICODE_TRANSLATION, DATA_DIR

GRAPH_FILE = os.path.join(DATA_DIR, "knowledge_graph.json")


class KnowledgeGraphStore:
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self.adjacency: Dict[str, Set[str]] = {}
        self.load_graph()

    def add_node(self, entity_id: str, name: str, entity_type: str = "Concept", chunk_id: Optional[str] = None):
        clean_id = entity_id.strip().lower()
        if not clean_id:
            return

        if clean_id not in self.nodes:
            self.nodes[clean_id] = {
                "id": clean_id,
                "name": name.strip(),
                "type": entity_type.strip(),
                "chunks": [chunk_id] if chunk_id else []
            }
            self.adjacency[clean_id] = set()
        else:
            if chunk_id and chunk_id not in self.nodes[clean_id]["chunks"]:
                self.nodes[clean_id]["chunks"].append(chunk_id)

    def add_edge(self, source_id: str, target_id: str, relation: str, chunk_id: Optional[str] = None):
        src = source_id.strip().lower()
        tgt = target_id.strip().lower()
        rel = relation.strip().upper()

        if not src or not tgt or src == tgt:
            return

        self.add_node(src, source_id)
        self.add_node(tgt, target_id)

        self.adjacency[src].add(tgt)
        self.adjacency[tgt].add(src)

        for edge in self.edges:
            if edge["source"] == src and edge["target"] == tgt and edge["relation"] == rel:
                if chunk_id and chunk_id not in edge["chunks"]:
                    edge["chunks"].append(chunk_id)
                edge["weight"] += 1
                return

        self.edges.append({
            "source": src,
            "target": tgt,
            "relation": rel,
            "weight": 1,
            "chunks": [chunk_id] if chunk_id else []
        })

    def search_entities(self, query: str, top_k: int = 5) -> List[str]:
        query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        if not query_words:
            return list(self.nodes.keys())[:top_k]

        scored_nodes = []
        for node_id, node_data in self.nodes.items():
            node_name_words = set(re.findall(r"\b\w{3,}\b", node_data["name"].lower()))
            overlap = len(query_words.intersection(node_name_words))
            if overlap > 0 or node_id in query.lower() or any(w in node_id for w in query_words):
                score = overlap + (2 if node_id in query.lower() else 0)
                scored_nodes.append((node_id, score))

        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in scored_nodes[:top_k]]

    def get_subgraph(self, seed_entity_ids: List[str], depth: int = 2) -> Dict[str, Any]:
        visited_nodes: Set[str] = set()
        subgraph_edges: List[Dict[str, Any]] = []
        current_layer = set(seed_entity_ids)

        for _ in range(depth):
            next_layer = set()
            for node_id in current_layer:
                if node_id in self.nodes and node_id not in visited_nodes:
                    visited_nodes.add(node_id)
                    neighbors = self.adjacency.get(node_id, set())
                    next_layer.update(neighbors - visited_nodes)

                    for edge in self.edges:
                        if (edge["source"] == node_id or edge["target"] == node_id) and edge not in subgraph_edges:
                            subgraph_edges.append(edge)
            current_layer = next_layer
            if not current_layer:
                break

        subgraph_nodes = [self.nodes[n] for n in visited_nodes if n in self.nodes]
        return {
            "nodes": subgraph_nodes,
            "edges": subgraph_edges
        }

    def save_graph(self):
        data = {
            "nodes": self.nodes,
            "edges": self.edges
        }
        with open(GRAPH_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_graph(self):
        if os.path.exists(GRAPH_FILE):
            try:
                with open(GRAPH_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.nodes = data.get("nodes", {})
                    self.edges = data.get("edges", [])
                    self.adjacency = {}
                    for nid in self.nodes:
                        self.adjacency[nid] = set()
                    for e in self.edges:
                        src, tgt = e["source"], e["target"]
                        self.adjacency.setdefault(src, set()).add(tgt)
                        self.adjacency.setdefault(tgt, set()).add(src)
            except Exception as e:
                print(f"[Graph Store] Failed to load graph: {e}")

    def clear(self):
        self.nodes = {}
        self.edges = []
        self.adjacency = {}
        if os.path.exists(GRAPH_FILE):
            os.remove(GRAPH_FILE)


# this is the graph rag based code
class GraphRAGPipeline:
    def __init__(self, vector_store: UnifiedVectorStore):
        self.vector_store = vector_store
        self.graph_store = KnowledgeGraphStore()
        self.http_client = httpx.Client(
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        if not self.graph_store.nodes:
            self.auto_build_graph()

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
            else:
                print(f"[Graph RAG] LLM error: {data}")
                return ""
        except Exception as e:
            print(f"[Graph RAG] LLM request failed: {e}")
            return ""

    def extract_triplets_from_text(self, text: str, chunk_id: str) -> List[Dict[str, str]]:
        triplets = []
        if GROQ_API_KEY:
            prompt = (
                "Extract key knowledge graph relationships from the following text.\n"
                "Return ONLY a JSON array of objects with keys: 'source', 'target', 'relation', 'source_type', 'target_type'.\n"
                "Keep entity names clean and short. Do not wrap in markdown or write explanations.\n"
                "Example format: [{\"source\": \"Taj Mahal\", \"relation\": \"LOCATED_IN\", \"target\": \"Agra\", \"source_type\": \"Monument\", \"target_type\": \"City\"}]\n\n"
                f"Text:\n{text[:1200]}"
            )
            response = self._call_llm(prompt)
            if response:
                try:
                    match = re.search(r"\[.*\]", response, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(0))
                        for item in parsed:
                            if isinstance(item, dict) and "source" in item and "target" in item and "relation" in item:
                                triplets.append(item)
                except Exception:
                    pass

        # Fast rule-based NLP fallback for relationship extraction
        if not triplets:
            capitalized_phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
            unique_entities = list(dict.fromkeys(capitalized_phrases))[:6]
            for i in range(len(unique_entities) - 1):
                triplets.append({
                    "source": unique_entities[i],
                    "relation": "ASSOCIATED_WITH",
                    "target": unique_entities[i + 1],
                    "source_type": "Entity",
                    "target_type": "Entity"
                })

        for trip in triplets:
            self.graph_store.add_node(trip["source"], trip["source"], trip.get("source_type", "Concept"), chunk_id)
            self.graph_store.add_node(trip["target"], trip["target"], trip.get("target_type", "Concept"), chunk_id)
            self.graph_store.add_edge(trip["source"], trip["target"], trip["relation"], chunk_id)

        return triplets

    def auto_build_graph(self) -> Dict[str, Any]:
        chunks = self.vector_store.get_all_chunks()
        if not chunks:
            return {"status": "no_chunks", "nodes": 0, "edges": 0}

        print(f"[Graph RAG] Building Knowledge Graph from {len(chunks)} chunks...")
        for c in chunks:
            self.extract_triplets_from_text(c["text"], c["id"])

        self.graph_store.save_graph()
        return {
            "status": "success",
            "total_nodes": len(self.graph_store.nodes),
            "total_edges": len(self.graph_store.edges)
        }

    def generate_graph_response(self, query: str, depth: int = 2, top_k: int = 4) -> Dict[str, Any]:
        # 1. Seed Entity Identification
        seed_entities = self.graph_store.search_entities(query, top_k=3)

        # 2. Subgraph Traversal & Community Exploration
        subgraph = self.graph_store.get_subgraph(seed_entities, depth=depth)

        # 3. Linked Context Chunk Collection
        relevant_chunk_ids: Set[str] = set()
        for node in subgraph["nodes"]:
            relevant_chunk_ids.update(node.get("chunks", []))
        for edge in subgraph["edges"]:
            relevant_chunk_ids.update(edge.get("chunks", []))

        all_chunks = self.vector_store.get_all_chunks()
        chunk_map = {c["id"]: c for c in all_chunks}
        graph_linked_chunks = [chunk_map[cid] for cid in relevant_chunk_ids if cid in chunk_map]

        # Complement with vector similarity if graph coverage is sparse
        if len(graph_linked_chunks) < top_k:
            vector_chunks = self.vector_store.search(query, top_k=top_k)
            for vc in vector_chunks:
                if vc["id"] not in relevant_chunk_ids and len(graph_linked_chunks) < top_k:
                    graph_linked_chunks.append(vc)

        # 4. Structured Triplet Context & Document Text
        triplet_lines = [
            f"- ({e['source'].title()}) --[{e['relation']}]--> ({e['target'].title()})"
            for e in subgraph["edges"][:15]
        ]
        graph_context = "\n".join(triplet_lines) if triplet_lines else "No direct entity relations mapped."

        doc_context = "\n\n".join([
            f"--- Document Context from {c['filename']} (Page {c['page']}) ---\n{c['text']}"
            for c in graph_linked_chunks[:top_k]
        ])

        # 5. Graph-Augmented Synthesis Prompt
        system_prompt = (
            "You are a Graph RAG Knowledge Agent. "
            "Synthesize your answer by fusing the structured Knowledge Graph entity relationships "
            "with the provided textual document context. "
            "Highlight connections between entities, explain multi-hop relations, and provide a clear, coherent answer. "
            "Do not output markdown tables or fabricate unsupported connections."
        )

        user_prompt = f"""Structured Knowledge Graph Relationships:
{graph_context}

Document Context:
{doc_context}

User Question: {query}

Instructions:
- Explain the key entities and how they relate based on the Knowledge Graph and text.
- Provide a clear, natural, well-formatted answer using standard paragraphs and bullet points.
- If the graph and context do not have enough information, politely state the limitations.
"""

        answer = self._call_llm(user_prompt, system_prompt=system_prompt)
        if not answer:
            answer = (
                f"**Knowledge Graph Entities Explored:** {', '.join([n['name'] for n in subgraph['nodes'][:5]])}\n\n"
                f"**Key Relationships Identified:**\n{graph_context}\n\n"
                f"*(Note: Graph RAG traversal complete. Ensure a valid `GROQ_API_KEY` is configured in `.env` for LLM synthesis.)*"
            )

        sources = []
        for c in graph_linked_chunks[:top_k]:
            sources.append({
                "filename": c["filename"],
                "page": c["page"],
                "doc_type": c.get("doc_type", "pdf"),
                "snippet": c["text"][:200] + ("..." if len(c["text"]) > 200 else "")
            })

        return {
            "answer": answer,
            "sources": sources,
            "graph_metadata": {
                "seed_entities": seed_entities,
                "nodes_traversed": len(subgraph["nodes"]),
                "edges_traversed": len(subgraph["edges"]),
                "subgraph_triplets": triplet_lines[:10],
                "depth": depth
            }
        }

    def close(self):
        self.http_client.close()
