"""
iterative_rag.py
----------------
System 2 — Iterative RAG (fixed N-round retrieval).

Pipeline (N=2 rounds):
    Query
      │
      ▼
    Round 1: Retrieve top-k  →  LLM extracts intermediate facts
      │
      ▼
    Round 2: Re-query with (question + intermediate facts)  →  Retrieve top-k
      │
      ▼
    LLM generates final answer from accumulated, de-duplicated context

Key difference from Naive RAG: query reformulation between rounds lets the
second retrieval fetch passages about entities identified in round 1 (the
classic "2-hop" pattern in HotpotQA / MuSiQue).

LangChain component: LCEL chain (replaces deprecated SequentialChain)

Usage:
    python src/rag/iterative_rag.py \
        --data    data/processed/combined_2000.json \
        --index   data/faiss_index/ \
        --output  results/iterative_rag.json \
        --k       3 \
        --rounds  2 \
        --limit   50
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from tqdm import tqdm

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.retriever.vector_store import VectorStoreIndex

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LLM_MODEL = "gpt-4o-mini"
TOP_K      = 3
N_ROUNDS   = 2

# Round 1 prompt: extract the key intermediate entity/fact from initial passages
EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["passages", "question"],
    template="""\
You are a reasoning assistant. Read the passages and the question carefully.
Identify the single most important intermediate fact or entity needed to \
answer the question. Output ONLY that fact — no explanation, no punctuation \
beyond what is necessary.

Passages:
{passages}

Question: {question}
Key intermediate fact:""",
)

# Final answer prompt: synthesize all accumulated context
ANSWER_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""\
Use the following passages to answer the question.
Answer as concisely as possible — a short phrase or single entity is preferred.
If the passages do not contain enough information, answer with "I don't know".

Passages:
{context}

Question: {question}
Answer:""",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_passages(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"[{i}] {doc.metadata['title']}\n{doc.page_content}")
    return "\n\n".join(parts)


def _dedup(docs: List[Document]) -> List[Document]:
    """Remove duplicate passages by passage_id, preserving order."""
    seen: set[str] = set()
    out:  List[Document] = []
    for doc in docs:
        # passage_id is always set by VectorStoreIndex — use it as the key
        pid = doc.metadata.get("passage_id") or doc.page_content[:80]
        if pid not in seen:
            seen.add(pid)
            out.append(doc)
    return out


# ---------------------------------------------------------------------------
# Single-question inference
# ---------------------------------------------------------------------------

def run_one(
    record: Dict[str, Any],
    store:  VectorStoreIndex,
    llm:    ChatOpenAI,
    k:      int,
    rounds: int,
) -> Dict[str, Any]:
    qid      = record["id"]
    question = record["question"]

    extraction_chain = EXTRACTION_PROMPT | llm | StrOutputParser()
    answer_chain     = ANSWER_PROMPT     | llm | StrOutputParser()

    t0 = time.perf_counter()

    with get_openai_callback() as cb:
        # ── Iterative retrieval ─────────────────────────────────────────
        all_docs:      List[Document] = []
        current_query: str            = question
        retrieval_steps: int          = 0

        for round_idx in range(rounds):
            new_docs = store.search(qid, current_query, k=k)
            retrieval_steps += 1
            all_docs = _dedup(all_docs + new_docs)

            # Reformulate query between rounds (not after the last one)
            if round_idx < rounds - 1:
                passages_text = _format_passages(all_docs)
                intermediate  = extraction_chain.invoke({
                    "passages": passages_text,
                    "question": question,
                }).strip()

                # Guard: only extend query if extraction returned something useful
                if intermediate and intermediate.lower() != "i don't know":
                    current_query = f"{question} {intermediate}"
                # else: keep original question for next round

        # ── Final answer ────────────────────────────────────────────────
        context   = _format_passages(all_docs)
        predicted = answer_chain.invoke({
            "context":  context,
            "question": question,
        }).strip()

    latency = time.perf_counter() - t0

    retrieved = [
        {
            "title":   d.metadata["title"],
            "text":    d.page_content,
            "is_gold": d.metadata["is_gold"],
        }
        for d in all_docs
    ]

    return {
        "id":                 qid,
        "dataset":            record["dataset"],
        "question":           question,
        "gold_answer":        record["gold_answer"],
        "all_answers":        record["all_answers"],
        "n_hops":             record["n_hops"],
        "predicted_answer":   predicted,
        "retrieval_steps":    retrieval_steps,
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
    rounds:      int,
    limit:       Optional[int],
    resume:      bool,
) -> None:

    with open(data_path, encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)
    if limit:
        records = records[:limit]
    print(f"[IterativeRAG] {len(records)} questions, {rounds} retrieval rounds, k={k}.")

    done_ids: set[str] = set()
    existing_results: List[Dict[str, Any]] = []
    if resume and os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            existing_results = json.load(f)
        done_ids = {r["id"] for r in existing_results}
        print(f"[IterativeRAG] Resuming — {len(done_ids)} already done.")

    records_todo = [r for r in records if r["id"] not in done_ids]
    if not records_todo:
        print("[IterativeRAG] Nothing to do.")
        return

    store = VectorStoreIndex(index_dir)
    llm   = ChatOpenAI(model=LLM_MODEL, temperature=0)

    results = list(existing_results)
    new_results: List[Dict[str, Any]] = []
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    for record in tqdm(records_todo, desc="[IterativeRAG]"):
        result = run_one(record, store, llm, k, rounds)
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
        print(f"\n[IterativeRAG] Done — {n} new questions processed")
        print(f"  Avg retrieval steps   : {avg_steps:.1f}")
        print(f"  Avg tokens / question : {avg_tokens:.0f}")
        print(f"  Avg latency / question: {avg_latency:.2f}s")
    print(f"  Total saved: {len(results)} | Output → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Iterative RAG evaluation.")
    parser.add_argument("--data",   default="data/processed/combined_2000.json")
    parser.add_argument("--index",  default="data/faiss_index/")
    parser.add_argument("--output", default="results/iterative_rag.json")
    parser.add_argument("--k",      type=int, default=TOP_K)
    parser.add_argument("--rounds", type=int, default=N_ROUNDS)
    parser.add_argument("--limit",  type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    main(
        data_path=args.data,
        index_dir=args.index,
        output_path=args.output,
        k=args.k,
        rounds=args.rounds,
        limit=args.limit,
        resume=args.resume,
    )
