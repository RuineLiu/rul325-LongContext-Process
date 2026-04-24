"""
naive_rag.py  —  System 1: Naive RAG (single-shot retrieve-then-read)

Pipeline:
    Query → Retrieve top-k passages (k=3) → LLM generates answer
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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.retriever.vector_store import VectorStoreIndex

LLM_MODEL = "gpt-4o-mini"
TOP_K      = 3

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


def _format_passages(docs: list) -> str:
    return "\n\n".join(
        f"[{i}] {d.metadata['title']}\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )


def run_one(
    record: Dict[str, Any],
    store:  VectorStoreIndex,
    chain:  Any,
    k:      int,
) -> Dict[str, Any]:
    qid      = record["id"]
    question = record["question"]

    docs      = store.search(qid, question, k=k)
    context   = _format_passages(docs)

    t0 = time.perf_counter()
    with get_openai_callback() as cb:
        predicted = chain.invoke({"context": context, "question": question})
    latency = time.perf_counter() - t0

    return {
        "id":                 qid,
        "dataset":            record["dataset"],
        "question":           question,
        "gold_answer":        record["gold_answer"],
        "all_answers":        record["all_answers"],
        "n_hops":             record["n_hops"],
        "predicted_answer":   predicted.strip(),
        "retrieval_steps":    1,
        "tokens_prompt":      cb.prompt_tokens,
        "tokens_completion":  cb.completion_tokens,
        "tokens_total":       cb.total_tokens,
        "latency_s":          round(latency, 3),
        "retrieved_passages": [
            {"title": d.metadata["title"], "text": d.page_content,
             "is_gold": d.metadata["is_gold"]} for d in docs
        ],
    }


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
    print(f"[NaiveRAG] {len(records)} questions, k={k}.")

    done_ids: set[str] = set()
    existing: List[Dict[str, Any]] = []
    if resume and os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        done_ids = {r["id"] for r in existing}
        print(f"[NaiveRAG] Resuming — {len(done_ids)} already done.")

    todo = [r for r in records if r["id"] not in done_ids]
    if not todo:
        print("[NaiveRAG] Nothing to do.")
        return

    store = VectorStoreIndex(index_dir)
    llm   = ChatOpenAI(model=LLM_MODEL, temperature=0)
    chain = ANSWER_PROMPT | llm | StrOutputParser()

    results = list(existing)
    new: List[Dict[str, Any]] = []
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    for record in tqdm(todo, desc="[NaiveRAG]"):
        result = run_one(record, store, chain, k)
        results.append(result)
        new.append(result)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    if new:
        n = len(new)
        print(f"\n[NaiveRAG] Done — {n} questions")
        print(f"  Avg tokens  : {sum(r['tokens_total'] for r in new)/n:.0f}")
        print(f"  Avg latency : {sum(r['latency_s'] for r in new)/n:.2f}s")
    print(f"  Total saved: {len(results)} → {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",   default="data/processed/combined_2000.json")
    p.add_argument("--index",  default="data/faiss_index/")
    p.add_argument("--output", default="results/naive_rag.json")
    p.add_argument("--k",      type=int, default=TOP_K)
    p.add_argument("--limit",  type=int, default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    main(args.data, args.index, args.output, args.k, args.limit, args.resume)
