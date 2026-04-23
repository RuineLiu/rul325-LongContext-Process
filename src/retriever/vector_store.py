"""
vector_store.py
---------------
Build and query per-question FAISS vector stores for RAG evaluation.

Each question in HotpotQA / MuSiQue comes with its own passage pool
(gold + distractors).  We embed all passages once, cache them on disk,
then build a lightweight in-memory FAISS index per question at retrieval
time — avoiding 2 000 separate index files while keeping per-question
passage isolation.

Build (run once, ~$0.50 for 2 000 questions with text-embedding-3-small):
    python src/retriever/vector_store.py \
        --input  data/processed/combined_2000.json \
        --output data/faiss_index/

Use in RAG scripts:
    from src.retriever.vector_store import VectorStoreIndex
    store = VectorStoreIndex("data/faiss_index/")
    docs  = store.search(question_id="hotpot_0", query="Einstein university", k=3)
    # or get a LangChain-compatible retriever:
    retriever = store.as_retriever(question_id="hotpot_0", k=3)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

# Must be set before faiss is imported — macOS ships multiple OpenMP runtimes
# (one from numpy, one from faiss-cpu) which conflict on initialisation.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDINGS_FILE = "embeddings.npy"   # shape (N_passages, dim)
METADATA_FILE   = "metadata.json"    # list of passage-level metadata dicts
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE      = 100                # passages per OpenAI API call
RATE_LIMIT_SLEEP = 1.0               # seconds between batches


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_index(input_path: str, output_dir: str) -> None:
    """
    Read combined JSON, embed every passage, save embeddings + metadata.

    metadata.json schema (list of dicts):
        {
          "passage_idx": int,       # row index into embeddings.npy
          "passage_id":  str,       # e.g. "hotpot_0_p3"
          "question_id": str,       # e.g. "hotpot_0"
          "title":       str,
          "text":        str,
          "is_gold":     bool,
          "dataset":     str,
          "n_hops":      int,
        }
    """
    os.makedirs(output_dir, exist_ok=True)

    embeddings_path = os.path.join(output_dir, EMBEDDINGS_FILE)
    metadata_path   = os.path.join(output_dir, METADATA_FILE)

    # Resume support: skip if already built
    if os.path.exists(embeddings_path) and os.path.exists(metadata_path):
        print(f"[VectorStore] Index already exists at {output_dir}. Skipping build.")
        return

    print(f"[VectorStore] Loading data from {input_path} ...")
    with open(input_path, encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    # Flatten all passages into a single list
    all_texts: List[str]            = []
    all_meta:  List[Dict[str, Any]] = []

    for record in records:
        qid = record["id"]
        for passage in record["passages"]:
            # Prepend title to text so the embedding captures topic signal
            text = f"{passage['title']}. {passage['text']}"
            all_texts.append(text)
            all_meta.append({
                "passage_idx": len(all_meta),
                "passage_id":  passage["id"],
                "question_id": qid,
                "title":       passage["title"],
                "text":        passage["text"],
                "is_gold":     passage["is_gold"],
                "dataset":     record["dataset"],
                "n_hops":      record["n_hops"],
            })

    print(f"[VectorStore] {len(all_texts):,} passages to embed "
          f"(~{len(all_texts) / BATCH_SIZE:.0f} API calls) ...")

    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    all_embeddings: List[List[float]] = []
    for start in range(0, len(all_texts), BATCH_SIZE):
        batch = all_texts[start : start + BATCH_SIZE]
        vecs  = embedder.embed_documents(batch)
        all_embeddings.extend(vecs)
        print(f"  Embedded {min(start + BATCH_SIZE, len(all_texts)):>6} / {len(all_texts)}", end="\r")
        if start + BATCH_SIZE < len(all_texts):
            time.sleep(RATE_LIMIT_SLEEP)

    print()  # newline after \r

    arr = np.array(all_embeddings, dtype=np.float32)  # (N, dim)
    np.save(embeddings_path, arr)
    print(f"[VectorStore] Saved embeddings → {embeddings_path}  shape={arr.shape}")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False)
    print(f"[VectorStore] Saved metadata   → {metadata_path}")


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class VectorStoreIndex:
    """
    Load pre-built embeddings and answer per-question retrieval queries.

    Each call to `search` or `as_retriever` builds a tiny in-memory FAISS
    index for the requested question's passages (10–20 docs), which takes
    < 1 ms and avoids reading the full index on every call.
    """

    def __init__(self, index_dir: str) -> None:
        try:
            import faiss  # noqa: F401 — confirm FAISS is available
        except ImportError as e:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu") from e

        embeddings_path = os.path.join(index_dir, EMBEDDINGS_FILE)
        metadata_path   = os.path.join(index_dir, METADATA_FILE)

        if not os.path.exists(embeddings_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Index not found at {index_dir}. "
                "Run: python src/retriever/vector_store.py --input ... --output ..."
            )

        print(f"[VectorStore] Loading index from {index_dir} ...")
        self._embeddings: np.ndarray = np.load(embeddings_path)  # (N, dim)

        with open(metadata_path, encoding="utf-8") as f:
            meta_list: List[Dict[str, Any]] = json.load(f)

        # Group passage indices by question_id for fast lookup
        self._meta: List[Dict[str, Any]] = meta_list
        self._qid_to_indices: Dict[str, List[int]] = {}
        for entry in meta_list:
            qid = entry["question_id"]
            self._qid_to_indices.setdefault(qid, []).append(entry["passage_idx"])

        self._embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        print(f"[VectorStore] Ready — {len(self._embeddings):,} passages, "
              f"{len(self._qid_to_indices):,} questions.")

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    def search(
        self,
        question_id: str,
        query: str,
        k: int = 3,
    ) -> List[Document]:
        """
        Return the top-k passages for `question_id` most similar to `query`.

        Returns a list of LangChain Document objects with metadata:
            title, is_gold, dataset, n_hops, passage_id, question_id
        """
        import faiss

        indices = self._qid_to_indices.get(question_id)
        if not indices:
            raise KeyError(f"question_id '{question_id}' not found in index.")

        # Slice the relevant rows from the global embedding matrix
        sub_embeddings = self._embeddings[indices]  # (n_passages, dim)
        n_passages = len(indices)
        k_actual   = min(k, n_passages)

        # Build a flat (exact) FAISS index for this question's passage pool
        # Use IndexFlatIP (inner product) which equals cosine similarity on L2-normed vectors
        dim   = sub_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        norms  = np.linalg.norm(sub_embeddings, axis=1, keepdims=True)
        normed = sub_embeddings / np.maximum(norms, 1e-9)
        index.add(normed)

        # Embed the query
        q_vec = np.array(self._embedder.embed_query(query), dtype=np.float32).reshape(1, -1)
        q_norm = q_vec / np.maximum(np.linalg.norm(q_vec), 1e-9)

        _, local_ids = index.search(q_norm, k_actual)  # (1, k)

        docs: List[Document] = []
        for local_id in local_ids[0]:
            if local_id == -1:
                continue
            global_idx = indices[local_id]
            entry      = self._meta[global_idx]
            docs.append(Document(
                page_content=entry["text"],
                metadata={
                    "title":       entry["title"],
                    "is_gold":     entry["is_gold"],
                    "dataset":     entry["dataset"],
                    "n_hops":      entry["n_hops"],
                    "passage_id":  entry["passage_id"],
                    "question_id": entry["question_id"],
                },
            ))
        return docs

    # ------------------------------------------------------------------
    # LangChain-compatible retriever
    # ------------------------------------------------------------------

    def as_retriever(self, question_id: str, k: int = 3) -> "_QuestionRetriever":
        """Return a LangChain BaseRetriever bound to a specific question."""
        return _QuestionRetriever(store=self, question_id=question_id, k=k)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def question_ids(self) -> List[str]:
        return list(self._qid_to_indices.keys())

    def passages_for(self, question_id: str) -> List[Dict[str, Any]]:
        """Return raw metadata for all passages belonging to a question."""
        indices = self._qid_to_indices.get(question_id, [])
        return [self._meta[i] for i in indices]


# ---------------------------------------------------------------------------
# LangChain retriever wrapper
# ---------------------------------------------------------------------------

try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

    class _QuestionRetriever(BaseRetriever):
        """Thin LangChain retriever bound to a single question's passage pool."""

        store:       Any  # VectorStoreIndex — avoid circular type ref
        question_id: str
        k:           int

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        ) -> List[Document]:
            return self.store.search(self.question_id, query, self.k)

except ImportError:
    # LangChain not installed — VectorStoreIndex.search() still works
    class _QuestionRetriever:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FAISS embedding index for RAG evaluation."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/combined_2000.json",
        help="Path to the combined dataset JSON.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/faiss_index/",
        help="Directory to save embeddings + metadata.",
    )
    args = parser.parse_args()

    build_index(input_path=args.input, output_dir=args.output)

    # Quick sanity check
    print("\n[VectorStore] Running sanity check ...")
    store = VectorStoreIndex(args.output)
    sample_qid = store.question_ids()[0]
    results = store.search(sample_qid, query="university founded", k=3)
    print(f"  Question ID : {sample_qid}")
    print(f"  Query       : 'university founded'")
    for i, doc in enumerate(results, 1):
        gold_flag = "GOLD" if doc.metadata["is_gold"] else "    "
        print(f"  [{i}] [{gold_flag}] {doc.metadata['title']}: {doc.page_content[:80]}...")
    print("\n[VectorStore] Done.")


if __name__ == "__main__":
    main()
