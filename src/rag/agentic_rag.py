"""
agentic_rag.py
--------------
System 3 — Agentic RAG (ReAct agent with dynamic retrieval + self-verification).

Pipeline:
    Query
      │
      ▼
    Agent thinks: "What do I need to find?"
      │
      ▼
    Agent calls search_documents(query)  →  retrieves passages
      │
      ▼
    Agent thinks: "Do I have enough evidence?"
      ├─ No  → reformulate query → search again
      └─ Yes → calls verify_answer(candidate_answer)
                    ├─ Not supported → retrieve more
                    └─ Supported     → return final answer

Tools available to the agent:
    search_documents  — semantic search over the question's passage pool
    verify_answer     — checks whether retrieved evidence supports a candidate

LangChain component: create_react_agent + AgentExecutor

Usage:
    python src/rag/agentic_rag.py \
        --data    data/processed/combined_2000.json \
        --index   data/faiss_index/ \
        --output  results/agentic_rag.json \
        --k       3 \
        --limit   50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_community.callbacks import get_openai_callback
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from tqdm import tqdm

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.retriever.vector_store import VectorStoreIndex

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LLM_MODEL      = "gpt-4o-mini"
TOP_K          = 3
MAX_ITERATIONS = 8   # safety cap — prevents runaway agents

# ReAct prompt — must include {tools}, {tool_names}, {input}, {agent_scratchpad}
# {max_iterations} is pre-filled via .partial() before passing to create_react_agent
REACT_PROMPT = PromptTemplate.from_template("""\
You are an expert question-answering agent. Your goal is to answer the question \
by retrieving evidence from a document store and verifying your answer before \
committing to it.

You have access to the following tools:
{tools}

Use this exact format for every step:

Thought: <your reasoning about what to do next>
Action: <tool name, must be one of [{tool_names}]>
Action Input: <input to the tool>
Observation: <tool result>
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have a verified answer.
Final Answer: <your concise final answer>

Rules:
- Answer as concisely as possible (short phrase or single entity).
- Always call verify_answer before giving the Final Answer.
- If verify_answer says the evidence is insufficient, search for more passages.
- If after {max_iterations} searches you still cannot verify, give your best guess.
- Never repeat the same search query twice.

Question: {input}
{agent_scratchpad}""")


# ---------------------------------------------------------------------------
# Per-question tool factory
# ---------------------------------------------------------------------------
# LangChain @tool closures are built fresh for each question, binding:
#   - the current question_id   (passage isolation)
#   - a shared retrieved_docs list (verify_answer reads what search found)
#   - a retrieval step counter

def make_tools(
    store:          VectorStoreIndex,
    question_id:    str,
    k:              int,
    retrieved_docs: List[Document],   # mutated in place across tool calls
    step_counter:   List[int],        # [count] — mutable via list
    verify_llm:     ChatOpenAI,
) -> list:
    """Return [search_documents, verify_answer] bound to the current question."""

    seen_ids: set[str] = set()   # tracks passage_id to avoid duplicate storage

    @tool
    def search_documents(query: str) -> str:
        """
        Search the document store for passages relevant to the query.
        Returns the top passages as formatted text.
        Input: a search query string.
        """
        docs = store.search(question_id, query, k=k)
        step_counter[0] += 1

        # Accumulate unique passages into the shared list
        for doc in docs:
            pid = doc.metadata.get("passage_id", doc.page_content[:80])
            if pid not in seen_ids:
                retrieved_docs.append(doc)
                seen_ids.add(pid)

        if not docs:
            return "No relevant passages found."

        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(f"[{i}] {doc.metadata['title']}\n{doc.page_content}")
        return "\n\n".join(parts)

    @tool
    def verify_answer(candidate_answer: str) -> str:
        """
        Check whether the retrieved evidence supports the candidate answer.
        Returns 'SUPPORTED' or 'NOT SUPPORTED: <reason>'.
        Input: a candidate answer string.
        """
        if not retrieved_docs:
            return "NOT SUPPORTED: No passages have been retrieved yet."

        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[{i}] {doc.metadata['title']}\n{doc.page_content}")
        context = "\n\n".join(context_parts)

        verify_prompt = (
            f"Passages:\n{context}\n\n"
            f"Candidate answer: {candidate_answer}\n\n"
            "Does the evidence in the passages directly support this answer? "
            "Reply with exactly 'SUPPORTED' if yes, or "
            "'NOT SUPPORTED: <one-sentence reason>' if no."
        )
        response = verify_llm.invoke(verify_prompt)
        return response.content.strip()

    return [search_documents, verify_answer]


# ---------------------------------------------------------------------------
# Single-question inference
# ---------------------------------------------------------------------------

def run_one(
    record:     Dict[str, Any],
    store:      VectorStoreIndex,
    llm:        ChatOpenAI,
    verify_llm: ChatOpenAI,
    k:          int,
) -> Dict[str, Any]:
    qid      = record["id"]
    question = record["question"]

    # Fresh state per question
    retrieved_docs: List[Document] = []
    step_counter:   List[int]      = [0]

    tools = make_tools(store, qid, k, retrieved_docs, step_counter, verify_llm)

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=REACT_PROMPT.partial(max_iterations=MAX_ITERATIONS),
    )
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=MAX_ITERATIONS,
        handle_parsing_errors=True,
        verbose=False,
    )

    t0 = time.perf_counter()
    with get_openai_callback() as cb:
        try:
            output    = executor.invoke({"input": question})
            predicted = output.get("output", "").strip()
        except Exception as exc:  # noqa: BLE001
            predicted = f"ERROR: {exc}"
    latency = time.perf_counter() - t0

    retrieved = [
        {
            "title":   d.metadata["title"],
            "text":    d.page_content,
            "is_gold": d.metadata["is_gold"],
        }
        for d in retrieved_docs
    ]

    return {
        "id":                 qid,
        "dataset":            record["dataset"],
        "question":           question,
        "gold_answer":        record["gold_answer"],
        "all_answers":        record["all_answers"],
        "n_hops":             record["n_hops"],
        "predicted_answer":   predicted,
        "retrieval_steps":    step_counter[0],
        "tokens_prompt":      cb.prompt_tokens,
        "tokens_completion":  cb.completion_tokens,
        "tokens_total":       cb.total_tokens,
        "latency_s":          round(latency, 3),
        "retrieved_passages": retrieved,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    data_path:   str,
    index_dir:   str,
    output_path: str,
    k:           int,
    limit:       Optional[int],
    resume:      bool,
) -> None:

    with open(data_path, encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)
    if limit:
        records = records[:limit]
    print(f"[AgenticRAG] {len(records)} questions, k={k}, max_iterations={MAX_ITERATIONS}.")

    done_ids: set[str] = set()
    existing_results: List[Dict[str, Any]] = []
    if resume and os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            existing_results = json.load(f)
        done_ids = {r["id"] for r in existing_results}
        print(f"[AgenticRAG] Resuming — {len(done_ids)} already done.")

    records_todo = [r for r in records if r["id"] not in done_ids]
    if not records_todo:
        print("[AgenticRAG] Nothing to do.")
        return

    store      = VectorStoreIndex(index_dir)
    llm        = ChatOpenAI(model=LLM_MODEL, temperature=0)
    verify_llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    results = list(existing_results)
    new_results: List[Dict[str, Any]] = []
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    for record in tqdm(records_todo, desc="[AgenticRAG]"):
        result = run_one(record, store, llm, verify_llm, k)
        results.append(result)
        new_results.append(result)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # ── Summary (current run only) ──────────────────────────────────────────
    if new_results:
        n = len(new_results)
        avg_tokens  = sum(r["tokens_total"]    for r in new_results) / n
        avg_steps   = sum(r["retrieval_steps"] for r in new_results) / n
        avg_latency = sum(r["latency_s"]       for r in new_results) / n
        print(f"\n[AgenticRAG] Done — {n} new questions processed")
        print(f"  Avg retrieval steps   : {avg_steps:.2f}")
        print(f"  Avg tokens / question : {avg_tokens:.0f}")
        print(f"  Avg latency / question: {avg_latency:.2f}s")
    print(f"  Total saved: {len(results)} | Output → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Agentic RAG evaluation.")
    parser.add_argument("--data",   default="data/processed/combined_2000.json")
    parser.add_argument("--index",  default="data/faiss_index/")
    parser.add_argument("--output", default="results/agentic_rag.json")
    parser.add_argument("--k",      type=int, default=TOP_K)
    parser.add_argument("--limit",  type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    main(
        data_path=args.data,
        index_dir=args.index,
        output_path=args.output,
        k=args.k,
        limit=args.limit,
        resume=args.resume,
    )
