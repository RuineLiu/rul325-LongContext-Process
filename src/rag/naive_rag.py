"""
naive_rag.py
------------
System 1 — Naive RAG (single-shot retrieve-then-read).

Pipeline:
    Query → Retrieve top-k passages (k=3, one call) → LLM generates answer

LangChain component: RetrievalQA

Usage:
    python src/rag/naive_rag.py \
        --data    data/processed/combined_2000.json \
        --index   data/faiss_index/ \
        --output  results/naive_rag.json \
        --k       3 \
        --limit   50          # optional: run on first N questions only
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from tqdm import tqdm

load_dotenv()

# Project-local import (run from repo root)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.retriever.vector_store import VectorStoreIndex

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LLM_MODEL = "gpt-4o-mini"
TOP_K      = 3

PROMPT_TEMPLATE = """\
Use the following passages to answer the question.
Answer as concisely as possible — a short phrase or single entity is preferred.
If the passages do not contain enough information, answer with "I don't know".

Passages:
{context}

Question: {question}
Answer:"""


# ---------------------------------------------------------------------------
# Single-question inference
# ---------------------------------------------------------------------------

def run_one(
    record: Dict[str, Any],
    store:  VectorStoreIndex,
    chain:  RetrievalQA,
    k:      int,
) -> Dict[str, Any]:
    """Run Naive RAG on a single question record and return a result dict."""
    qid      = record["id"]
    question = record["question"]

    # Swap the retriever to this question's passage pool
    chain.retriever = store.as_retriever(question_id=qid, k=k)

    t0 = time.perf_counter()
    with get_openai_callback() as cb:
        output = chain.invoke({"query": question})
    latency = time.perf_counter() - t0

    predicted = output.get("result", "").strip()

    # Collect retrieved passage metadata for later analysis
    retrieved_docs = store.search(qid, question, k=k)
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
        "retrieval_steps":    1,
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

    # ── Load data ──────────────────────────────────────────────────────────
    with open(data_path, encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)
    if limit:
        records = records[:limit]
    print(f"[NaiveRAG] {len(records)} questions to process.")

    # ── Resume support ─────────────────────────────────────────────────────
    done_ids: set[str] = set()
    existing_results: List[Dict[str, Any]] = []
    if resume and os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            existing_results = json.load(f)
        done_ids = {r["id"] for r in existing_results}
        print(f"[NaiveRAG] Resuming — {len(done_ids)} already done.")

    records_todo = [r for r in records if r["id"] not in done_ids]
    if not records_todo:
        print("[NaiveRAG] Nothing to do.")
        return

    # ── Build LangChain components ─────────────────────────────────────────
    store = VectorStoreIndex(index_dir)
    llm   = ChatOpenAI(model=LLM_MODEL, temperature=0)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE,
    )

    # Placeholder retriever — replaced per question inside run_one()
    dummy_retriever = store.as_retriever(question_id=records_todo[0]["id"], k=k)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=dummy_retriever,
        chain_type_kwargs={"prompt": prompt},
    )

    # ── Run ────────────────────────────────────────────────────────────────
    results = list(existing_results)
    new_results: List[Dict[str, Any]] = []
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    for record in tqdm(records_todo, desc="[NaiveRAG]"):
        result = run_one(record, store, chain, k)
        results.append(result)
        new_results.append(result)

        # Incremental save — safe against crashes
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # ── Summary (current run only) ─────────────────────────────────────────
    if new_results:
        n = len(new_results)
        avg_tokens  = sum(r["tokens_total"] for r in new_results) / n
        avg_latency = sum(r["latency_s"]    for r in new_results) / n
        print(f"\n[NaiveRAG] Done — {n} new questions processed")
        print(f"  Avg tokens / question : {avg_tokens:.0f}")
        print(f"  Avg latency / question: {avg_latency:.2f}s")
    print(f"  Total saved: {len(results)} | Output → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Naive RAG evaluation.")
    parser.add_argument("--data",   default="data/processed/combined_2000.json")
    parser.add_argument("--index",  default="data/faiss_index/")
    parser.add_argument("--output", default="results/naive_rag.json")
    parser.add_argument("--k",      type=int, default=TOP_K)
    parser.add_argument("--limit",  type=int, default=None,
                        help="Only process first N questions (for quick testing).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip questions already present in output file.")
    args = parser.parse_args()

    main(
        data_path=args.data,
        index_dir=args.index,
        output_path=args.output,
        k=args.k,
        limit=args.limit,
        resume=args.resume,
    )
