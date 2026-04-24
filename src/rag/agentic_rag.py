"""
agentic_rag.py  —  System 3: Agentic RAG (ReAct via LangGraph)

Pipeline:
    Query
      → Agent calls search_documents (dynamic, repeat as needed)
      → Agent calls verify_answer before committing
      → Final Answer

Uses langgraph.prebuilt.create_react_agent (tool-calling based ReAct).
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time
import random
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent as create_react_agent
from tqdm import tqdm

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.retriever.vector_store import VectorStoreIndex

LLM_MODEL      = "gpt-4o-mini"
TOP_K          = 3
MAX_ITERATIONS = 6   # max search calls; LangGraph limit set accordingly
RECURSION_LIMIT = MAX_ITERATIONS * 3 + 4   # ~22, enough for tool-call overhead

SYSTEM_PROMPT = """\
You are an expert question-answering agent. Use the tools to find evidence, \
then give a concise answer.

RULES:
- Call search_documents 1–3 times with different queries to gather evidence.
- Once you have enough evidence, call verify_answer ONCE with your best answer.
- After calling verify_answer, give your Final Answer immediately — do NOT \
  search again regardless of the verification result.
- Your Final Answer must be a short phrase or single entity ONLY.
  Good: "1755"  "John André"  "U2"
  Bad:  "The answer is 1755."  "Based on the evidence, U2 released it first."\
"""


# ---------------------------------------------------------------------------
# Answer post-processing: strip verbose wrapping if present
# ---------------------------------------------------------------------------

def _clean_answer(raw: str) -> str:
    """Strip common verbose prefixes the model sometimes adds."""
    import re
    # Remove prefixes like "The answer is ", "Based on ..., ", "According to ..., "
    raw = re.sub(
        r"^(the answer is|based on .*?,|according to .*?,|from the (passages?|evidence).*?,)\s*",
        "", raw, flags=re.IGNORECASE,
    ).strip()
    # Strip trailing periods
    raw = raw.rstrip(".")
    return raw


# ---------------------------------------------------------------------------
# Tool factory — verify_answer limited to ONE call per question
# ---------------------------------------------------------------------------

def make_tools(
    store:          VectorStoreIndex,
    question_id:    str,
    k:              int,
    retrieved_docs: List[Document],
    step_counter:   List[int],
    verify_llm:     ChatOpenAI,
) -> list:
    seen_ids:      set[str] = set()
    verify_called: List[int] = [0]   # [count] — limit verify to 1 call

    @tool
    def search_documents(query: str) -> str:
        """Search for passages relevant to the query. Call this 1-3 times."""
        docs = store.search(question_id, query, k=k)
        step_counter[0] += 1

        for doc in docs:
            pid = doc.metadata.get("passage_id", doc.page_content[:80])
            if pid not in seen_ids:
                retrieved_docs.append(doc)
                seen_ids.add(pid)

        if not docs:
            return "No relevant passages found."
        return "\n\n".join(
            f"[{i}] {d.metadata['title']}\n{d.page_content}"
            for i, d in enumerate(docs, 1)
        )

    @tool
    def verify_answer(candidate_answer: str) -> str:
        """Verify that the evidence supports your answer. Call this ONCE only,
        then give your Final Answer immediately."""
        # Hard limit: only run verification once to prevent re-search loops
        if verify_called[0] >= 1:
            return "SUPPORTED (verification already completed — give your final answer now)"

        verify_called[0] += 1

        if not retrieved_docs:
            return "NOT SUPPORTED: No passages retrieved yet."

        context = "\n\n".join(
            f"[{i}] {d.metadata['title']}\n{d.page_content}"
            for i, d in enumerate(retrieved_docs[:6], 1)   # cap context
        )
        resp = verify_llm.invoke(
            f"Passages:\n{context}\n\n"
            f"Candidate answer: {candidate_answer}\n\n"
            "Does the evidence support this answer (even partially)? "
            "Be lenient — reply 'SUPPORTED' if any passage is relevant, "
            "otherwise 'NOT SUPPORTED: <one-line reason>'."
        )
        return resp.content.strip()

    return [search_documents, verify_answer]


def _with_retry(fn, max_retries: int = 5):
    """Call fn(), retrying on RateLimitError with exponential back-off.
    Raises immediately on daily-quota errors (RPD exhausted)."""
    import openai
    for attempt in range(max_retries):
        try:
            return fn()
        except openai.RateLimitError as e:
            msg = str(e)
            # Daily quota (RPD) exhausted — no point retrying
            if "requests per day" in msg or "RPD" in msg:
                raise
            wait = (2 ** attempt) + random.random()
            print(f"\n[AgenticRAG] Rate limit hit (attempt {attempt+1}/{max_retries}), "
                  f"waiting {wait:.1f}s …", flush=True)
            time.sleep(wait)
    return fn()   # final attempt — let it raise if it fails


def run_one(
    record:     Dict[str, Any],
    store:      VectorStoreIndex,
    llm:        ChatOpenAI,
    verify_llm: ChatOpenAI,
    k:          int,
) -> Dict[str, Any]:
    qid      = record["id"]
    question = record["question"]

    retrieved_docs: List[Document] = []
    step_counter:   List[int]      = [0]

    tools = make_tools(store, qid, k, retrieved_docs, step_counter, verify_llm)
    agent = create_react_agent(llm, tools)

    t0 = time.perf_counter()
    with get_openai_callback() as cb:
        try:
            result = _with_retry(lambda: agent.invoke(
                {"messages": [SystemMessage(content=SYSTEM_PROMPT),
                               HumanMessage(content=question)]},
                {"recursion_limit": RECURSION_LIMIT},
            ))
            predicted = _clean_answer(result["messages"][-1].content.strip())
        except Exception as exc:  # noqa: BLE001
            # API quota / auth errors should crash loudly, not be silently saved
            import openai
            if isinstance(exc, (openai.RateLimitError,
                                 openai.AuthenticationError,
                                 openai.APIConnectionError)):
                raise
            # Recursion-limit or graph errors: fallback to direct LLM answer
            if retrieved_docs:
                context = "\n\n".join(
                    f"[{i}] {d.metadata['title']}\n{d.page_content}"
                    for i, d in enumerate(retrieved_docs[:5], 1)
                )
                resp = _with_retry(lambda: llm.invoke(
                    f"Passages:\n{context}\n\n"
                    f"Question: {question}\n"
                    "Answer with a short phrase or entity only:"
                ))
                predicted = _clean_answer(resp.content.strip())
            else:
                predicted = "I don't know"
    latency = time.perf_counter() - t0

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
        "retrieved_passages": [
            {"title": d.metadata["title"], "text": d.page_content,
             "is_gold": d.metadata["is_gold"]} for d in retrieved_docs
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
    print(f"[AgenticRAG] {len(records)} questions, k={k}, max_iter={MAX_ITERATIONS}.")

    done_ids: set[str] = set()
    existing: List[Dict[str, Any]] = []
    if resume and os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        done_ids = {r["id"] for r in existing}
        print(f"[AgenticRAG] Resuming — {len(done_ids)} already done.")

    todo = [r for r in records if r["id"] not in done_ids]
    if not todo:
        print("[AgenticRAG] Nothing to do.")
        return

    store      = VectorStoreIndex(index_dir)
    llm        = ChatOpenAI(model=LLM_MODEL, temperature=0)
    verify_llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    results = list(existing)
    new: List[Dict[str, Any]] = []
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    for record in tqdm(todo, desc="[AgenticRAG]"):
        result = run_one(record, store, llm, verify_llm, k)
        results.append(result)
        new.append(result)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    if new:
        n = len(new)
        print(f"\n[AgenticRAG] Done — {n} questions")
        print(f"  Avg steps   : {sum(r['retrieval_steps'] for r in new)/n:.2f}")
        print(f"  Avg tokens  : {sum(r['tokens_total'] for r in new)/n:.0f}")
        print(f"  Avg latency : {sum(r['latency_s'] for r in new)/n:.2f}s")
    print(f"  Total saved: {len(results)} → {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",   default="data/processed/combined_2000.json")
    p.add_argument("--index",  default="data/faiss_index/")
    p.add_argument("--output", default="results/agentic_rag.json")
    p.add_argument("--k",      type=int, default=TOP_K)
    p.add_argument("--limit",  type=int, default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    main(args.data, args.index, args.output, args.k, args.limit, args.resume)
