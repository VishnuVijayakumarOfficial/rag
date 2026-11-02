import os
import json
import re
from typing import List, Dict, Any, Optional
import httpx

from config import GROQ_API_KEY, GROQ_MODEL
from rag_engine import UnifiedVectorStore, UNICODE_TRANSLATION


# this is the agentic rag based code
class AgenticRAGPipeline:
    """Agentic RAG Pipeline:
    Autonomous reasoning agent operating in a ReAct (Reason -> Act -> Observe) loop.
    Equipped with tools for multi-step document search, keyword lookup, calculation, and answer synthesis.
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
                    "temperature": 0.1,
                }
            )
            data = response.json()
            if "choices" in data:
                content = data["choices"][0]["message"]["content"]
                return content.translate(UNICODE_TRANSLATION).strip()
        except Exception as e:
            print(f"[Agentic RAG] LLM error: {e}")

        return ""

    def _tool_document_search(self, query: str) -> str:
        chunks = self.vector_store.search(query, top_k=3)
        if not chunks:
            return "No matching document passages found."
        return "\n\n".join([f"[{c['filename']} p.{c['page']}]: {c['text']}" for c in chunks])

    def _tool_keyword_lookup(self, term: str) -> str:
        all_chunks = self.vector_store.get_all_chunks()
        matches = [c for c in all_chunks if term.lower() in c["text"].lower()][:3]
        if not matches:
            return f"No document passages containing the keyword '{term}' found."
        return "\n\n".join([f"[{c['filename']} p.{c['page']}]: {c['text']}" for c in matches])

    def _tool_calculator(self, expression: str) -> str:
        clean_expr = re.sub(r"[^0-9+\-*/(). ]", "", expression)
        try:
            result = eval(clean_expr, {"__builtins__": None}, {})
            return str(result)
        except Exception as e:
            return f"Calculation error: {e}"

    def run_agentic_loop(self, query: str, max_iterations: int = 4) -> Dict[str, Any]:
        agent_scratchpad = []
        collected_sources = []

        system_prompt = (
            "You are an Autonomous Agentic RAG assistant. You solve the user's question using a ReAct thought-action loop.\n"
            "You have access to the following tools:\n"
            "1. Action: document_search[query] - Searches documents using semantic vector search.\n"
            "2. Action: keyword_lookup[term] - Searches documents for exact keyword matches.\n"
            "3. Action: calculator[expression] - Computes mathematical expressions.\n"
            "4. Action: final_answer[answer] - Produces the final answer when you have gathered sufficient information.\n\n"
            "Use the exact format:\n"
            "Thought: <your reasoning>\n"
            "Action: <tool_name>[<argument>]\n"
        )

        current_prompt = f"Question: {query}\n"

        for step in range(1, max_iterations + 1):
            if not GROQ_API_KEY:
                # Direct tool fallback execution
                obs = self._tool_document_search(query)
                agent_scratchpad.append({
                    "step": step,
                    "thought": "Direct document search execution (fallback mode)",
                    "action": f"document_search[{query}]",
                    "observation": obs[:300]
                })
                return {
                    "answer": f"**Agentic RAG Result:**\n\n{obs[:500]}...",
                    "sources": [],
                    "agentic_metadata": {
                        "iterations": step,
                        "scratchpad": agent_scratchpad,
                        "completed": True
                    }
                }

            llm_response = self._call_llm(current_prompt, system_prompt=system_prompt)
            if not llm_response:
                break

            # Parse Thought and Action
            thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", llm_response, re.DOTALL)
            action_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", llm_response, re.DOTALL)

            thought = thought_match.group(1).strip() if thought_match else "Analyzing gathered information"
            
            if not action_match:
                # Treat entire response as final answer if format deviates
                agent_scratchpad.append({"step": step, "thought": thought, "action": "final_answer", "observation": "Completed"})
                return {
                    "answer": llm_response,
                    "sources": collected_sources,
                    "agentic_metadata": {
                        "iterations": step,
                        "scratchpad": agent_scratchpad,
                        "completed": True
                    }
                }

            tool_name = action_match.group(1).strip()
            tool_arg = action_match.group(2).strip()

            if tool_name == "final_answer":
                agent_scratchpad.append({"step": step, "thought": thought, "action": f"final_answer[{tool_arg[:100]}...]", "observation": "Goal achieved"})
                return {
                    "answer": tool_arg,
                    "sources": collected_sources,
                    "agentic_metadata": {
                        "iterations": step,
                        "scratchpad": agent_scratchpad,
                        "completed": True
                    }
                }

            # Execute Selected Tool
            if tool_name == "document_search":
                observation = self._tool_document_search(tool_arg)
                collected_sources.append({"tool": tool_name, "query": tool_arg, "preview": observation[:150]})
            elif tool_name == "keyword_lookup":
                observation = self._tool_keyword_lookup(tool_arg)
                collected_sources.append({"tool": tool_name, "query": tool_arg, "preview": observation[:150]})
            elif tool_name == "calculator":
                observation = self._tool_calculator(tool_arg)
            else:
                observation = f"Unknown tool '{tool_name}'."

            agent_scratchpad.append({
                "step": step,
                "thought": thought,
                "action": f"{tool_name}[{tool_arg}]",
                "observation": observation[:350]
            })

            current_prompt += f"\nThought: {thought}\nAction: {tool_name}[{tool_arg}]\nObservation: {observation[:600]}\n"

        # Final synthesis after max iterations
        synthesis_prompt = (
            f"Question: {query}\n\nExecution Observations:\n"
            + "\n".join([f"Step {s['step']}: {s['observation']}" for s in agent_scratchpad])
            + "\n\nProvide the final comprehensive answer based on the observations."
        )
        final_ans = self._call_llm(synthesis_prompt)
        return {
            "answer": final_ans or "Agent completed multi-step search observations.",
            "sources": collected_sources,
            "agentic_metadata": {
                "iterations": max_iterations,
                "scratchpad": agent_scratchpad,
                "completed": True
            }
        }

    def generate_agentic_response(self, query: str, max_iterations: int = 5) -> Dict[str, Any]:
        return self.run_agentic_loop(query, max_iterations=max_iterations)

    def close(self):
        self.http_client.close()
