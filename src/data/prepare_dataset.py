"""
prepare_dataset.py
------------------
Download and preprocess HotpotQA and MuSiQue datasets from HuggingFace.
Outputs a unified JSON format suitable for RAG pipeline evaluation.

Output format (per record):
{
    "id":          str,
    "dataset":     "hotpotqa" | "musique",
    "question":    str,
    "gold_answer": str,
    "all_answers": List[str],
    "n_hops":      int,
    "passages": [
        {"id": str, "title": str, "text": str, "is_gold": bool},
        ...
    ]
}

Usage:
    # From project root
    python src/data/prepare_dataset.py
    python src/data/prepare_dataset.py --n_hotpot 1000 --n_musique 1000 --output_dir data/processed/
"""

import argparse
import json
import os
import random
from typing import List, Dict, Any

from datasets import load_dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# HotpotQA
# ---------------------------------------------------------------------------

def load_hotpotqa(n_samples: int, seed: int) -> List[Dict[str, Any]]:
    """
    Load HotpotQA distractor-setting validation split.
    Each example has 10 paragraphs: 2 gold + 8 distractors.
    """
    print(f"[HotpotQA] Loading from HuggingFace ...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation")

    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n_samples, len(ds)))

    records = []
    for i in tqdm(indices, desc="[HotpotQA] Processing"):
        item = ds[i]

        gold_titles = set(item["supporting_facts"]["title"])

        passages = []
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            passages.append({
                "id": f"hotpot_{len(records)}_p{len(passages)}",
                "title": title,
                "text": " ".join(sentences).strip(),
                "is_gold": title in gold_titles,
            })

        records.append({
            "id": f"hotpot_{len(records)}",
            "dataset": "hotpotqa",
            "question": item["question"],
            "gold_answer": item["answer"],
            "all_answers": [item["answer"]],
            "n_hops": 2,
            "passages": passages,
        })

    return records


# ---------------------------------------------------------------------------
# MuSiQue
# ---------------------------------------------------------------------------

def load_musique(n_samples: int, seed: int) -> List[Dict[str, Any]]:
    """
    Load MuSiQue answerable validation split.
    Each example has 20 paragraphs: 2-4 gold + rest distractors.
    Only answerable examples are used.
    """
    print(f"[MuSiQue] Loading from HuggingFace ...")
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")

    # Filter to answerable only (all validation examples are answerable here)
    answerable = [item for item in ds if item["answerable"]]

    rng = random.Random(seed)
    sampled = rng.sample(answerable, min(n_samples, len(answerable)))

    records = []
    for item in tqdm(sampled, desc="[MuSiQue] Processing"):
        n_gold = sum(1 for p in item["paragraphs"] if p["is_supporting"])

        passages = []
        for p in item["paragraphs"]:
            passages.append({
                "id": f"musique_{len(records)}_p{len(passages)}",
                "title": p["title"],
                "text": p["paragraph_text"].strip(),
                "is_gold": p["is_supporting"],
            })

        answers = [item["answer"]] + list(item.get("answer_aliases", []))

        records.append({
            "id": f"musique_{len(records)}",
            "dataset": "musique",
            "question": item["question"],
            "gold_answer": item["answer"],
            "all_answers": answers,
            "n_hops": n_gold,       # number of supporting paragraphs = number of hops
            "passages": passages,
        })

    return records


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def print_stats(records: List[Dict[str, Any]], name: str):
    n = len(records)
    avg_passages = sum(len(r["passages"]) for r in records) / n
    avg_gold = sum(sum(1 for p in r["passages"] if p["is_gold"]) for r in records) / n
    avg_q_len = sum(len(r["question"].split()) for r in records) / n
    avg_a_len = sum(len(r["gold_answer"].split()) for r in records) / n
    hop_dist = {}
    for r in records:
        hop_dist[r["n_hops"]] = hop_dist.get(r["n_hops"], 0) + 1

    print(f"\n{'─'*45}")
    print(f"  {name} — {n} records")
    print(f"  Avg passages per question : {avg_passages:.1f}")
    print(f"  Avg gold passages         : {avg_gold:.1f}")
    print(f"  Avg question length (tok) : {avg_q_len:.1f}")
    print(f"  Avg answer length (tok)   : {avg_a_len:.1f}")
    print(f"  Hop distribution          : {hop_dist}")
    print(f"{'─'*45}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare HotpotQA + MuSiQue for RAG evaluation."
    )
    parser.add_argument("--n_hotpot",    type=int, default=1000)
    parser.add_argument("--n_musique",   type=int, default=1000)
    parser.add_argument("--output_dir",  type=str, default="data/processed/")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── HotpotQA ──────────────────────────────────────────────────────────
    hotpot = load_hotpotqa(n_samples=args.n_hotpot, seed=args.seed)
    print_stats(hotpot, "HotpotQA")
    hotpot_path = os.path.join(args.output_dir, f"hotpotqa_{len(hotpot)}.json")
    with open(hotpot_path, "w", encoding="utf-8") as f:
        json.dump(hotpot, f, ensure_ascii=False, indent=2)
    print(f"  Saved → {hotpot_path}")

    # ── MuSiQue ───────────────────────────────────────────────────────────
    musique = load_musique(n_samples=args.n_musique, seed=args.seed)
    print_stats(musique, "MuSiQue")
    musique_path = os.path.join(args.output_dir, f"musique_{len(musique)}.json")
    with open(musique_path, "w", encoding="utf-8") as f:
        json.dump(musique, f, ensure_ascii=False, indent=2)
    print(f"  Saved → {musique_path}")

    # ── Combined ──────────────────────────────────────────────────────────
    combined = hotpot + musique
    combined_path = os.path.join(args.output_dir, "combined_2000.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"  Saved combined → {combined_path}  ({len(combined)} records total)\n")


if __name__ == "__main__":
    main()
