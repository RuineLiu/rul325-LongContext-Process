"""
iterative_rag.py  —  System 2: Iterative RAG (fixed N-round retrieval)

Pipeline (N=2):
    Query
      → Round 1: retrieve → LLM extracts intermediate fact
      → Round 2: retrieve with (question + fact)
      → LLM generates final answer from accumulated context
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.retriever.vector_store import VectorStoreIndex

LLM_MODEL = "gpt-4o-mini"
TOP_K      = 3
N_ROUNDS   = 2

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


def _format_passages(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{i}] {d.metadata['title']}\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )


def _dedup(docs: List[Document]) -> List[Document]:
    seen: set[str] = set()
    out: List[Document] = []
    for doc in docs:
        pid = doc.metadata.get("passage_id") or doc.page_content[:80]
        if pid not in seen:
            seen.add(pid)
            out.append(doc)
    return out


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
        all_docs:      List[Document] = []
        current_query: str            = question
        retrieval_steps: int          = 0

        for round_idx in range(rounds):
            new_docs = store.search(qid, current_query, k=k)
            retrieval_steps += 1
            all_docs = _dedup(all_docs + new_docs)

            if round_idx < rounds - 1:
                intermediate = extraction_chain.invoke({
                    "passages": _format_passages(all_docs),
                    "question": question,
                }).strip()
                if intermediate and intermediate.lower() != "i don't know":
                    current_query = f"{question} {intermediate}"

        predicted = answer_chain.invoke({
            "context":  _format_passages(all_docs),
            "question": question,
        }).strip()

    latency = time.perf_counter() - t0

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
        "retrieved_passages": [
            {"title": d.metadata["title"], "text": d.page_content,
             "is_gold": d.metadata["is_gold"]} for d in all_docs
        ],
    }


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
    print(f"[IterativeRAG] {len(records)} questions, {rounds} rounds, k={k}.")

    done_ids: set[str] = set()
    existing: List[Dict[str, Any]] = []
    if resume and os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        done_ids = {r["id"] for r in existing}
        print(f"[IterativeRAG] Resuming — {len(done_ids)} already done.")

    todo = [r for r in records if r["id"] not in done_ids]
    if not todo:
        print("[IterativeRAG] Nothing to do.")
        return

    store = VectorStoreIndex(index_dir)
    llm   = ChatOpenAI(model=LLM_MODEL, temperature=0)

    results = list(existing)
    new: List[Dict[str, Any]] = []
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    for record in tqdm(todo, desc="[IterativeRAG]"):
        result = run_one(record, store, llm, k, rounds)
        results.append(result)
        new.append(result)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    if new:
        n = len(new)
        print(f"\n[IterativeRAG] Done — {n} questions")
        print(f"  Avg steps   : {sum(r['retrieval_steps'] for r in new)/n:.1f}")
        print(f"  Avg tokens  : {sum(r['tokens_total'] for r in new)/n:.0f}")
        print(f"  Avg latency : {sum(r['latency_s'] for r in new)/n:.2f}s")
    print(f"  Total saved: {len(results)} → {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",   default="data/processed/combined_2000.json")
    p.add_argument("--index",  default="data/faiss_index/")
    p.add_argument("--output", default="results/iterative_rag.json")
    p.add_argument("--k",      type=int, default=TOP_K)
    p.add_argument("--rounds", type=int, default=N_ROUNDS)
    p.add_argument("--limit",  type=int, default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    main(args.data, args.index, args.output, args.k, args.rounds, args.limit, args.resume)
