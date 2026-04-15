"""
metrics.py
----------
Evaluate RAG system outputs against gold answers.

Metrics computed:
    Accuracy  — Exact Match (EM), Token-level F1, BERTScore
    Efficiency — Avg retrieval steps, avg tokens, avg latency

Analysis slices (printed + saved):
    • Overall (all 2 000 questions)
    • By dataset  : hotpotqa | musique
    • By hop count: 2 | 3 | 4  (MuSiQue only)
    • Retrieval success rate: fraction of questions where ≥1 gold passage
      appeared in the retrieved context

Usage:
    python src/evaluation/metrics.py --results_dir results/
    python src/evaluation/metrics.py \
        --files results/naive_rag.json results/iterative_rag.json results/agentic_rag.json \
        --output results/evaluation_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Text normalisation (SQuAD / HotpotQA standard)
# ---------------------------------------------------------------------------

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, remove articles."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _ARTICLES_RE.sub(" ", text)
    text = " ".join(text.split())
    return text


# ---------------------------------------------------------------------------
# Exact Match
# ---------------------------------------------------------------------------

def exact_match(predicted: str, gold_answers: List[str]) -> int:
    """1 if normalised prediction matches any gold answer, else 0."""
    pred_norm = normalize(predicted)
    return int(any(pred_norm == normalize(g) for g in gold_answers))


# ---------------------------------------------------------------------------
# Token-level F1
# ---------------------------------------------------------------------------

def _token_f1_single(predicted: str, gold: str) -> float:
    pred_tokens = normalize(predicted).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return int(pred_tokens == gold_tokens)

    pred_counts: Dict[str, int] = defaultdict(int)
    gold_counts: Dict[str, int] = defaultdict(int)
    for t in pred_tokens:
        pred_counts[t] += 1
    for t in gold_tokens:
        gold_counts[t] += 1

    overlap = sum(min(pred_counts[t], gold_counts[t]) for t in pred_counts)
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall    = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(predicted: str, gold_answers: List[str]) -> float:
    """Max token-level F1 across all gold answers."""
    return max(_token_f1_single(predicted, g) for g in gold_answers)


# ---------------------------------------------------------------------------
# BERTScore (batch)
# ---------------------------------------------------------------------------

def bertscore_batch(
    predictions: List[str],
    references:  List[str],
    lang:        str = "en",
    batch_size:  int = 64,
) -> List[float]:
    """
    Return per-example F1 BERTScores.
    Falls back to token-level F1 if bert_score is unavailable.
    """
    try:
        from bert_score import score as bs_score
        _, _, F = bs_score(
            predictions,
            references,
            lang=lang,
            batch_size=batch_size,
            verbose=False,
        )
        return F.tolist()
    except ImportError:
        print("[metrics] bert_score not installed — using token F1 as fallback.")
        return [
            _token_f1_single(p, r) for p, r in zip(predictions, references)
        ]


# ---------------------------------------------------------------------------
# Retrieval success rate
# ---------------------------------------------------------------------------

def retrieval_success(result: Dict[str, Any]) -> int:
    """1 if at least one gold passage appears in retrieved_passages."""
    return int(any(p["is_gold"] for p in result.get("retrieved_passages", [])))


# ---------------------------------------------------------------------------
# Per-record scoring
# ---------------------------------------------------------------------------

def score_record(result: Dict[str, Any]) -> Dict[str, Any]:
    pred  = result.get("predicted_answer", "")
    golds = result.get("all_answers") or [result.get("gold_answer", "")]

    return {
        "id":               result["id"],
        "dataset":          result["dataset"],
        "n_hops":           result["n_hops"],
        "em":               exact_match(pred, golds),
        "f1":               token_f1(pred, golds),
        "retrieval_steps":  result.get("retrieval_steps", 0),
        "tokens_total":     result.get("tokens_total", 0),
        "latency_s":        result.get("latency_s", 0.0),
        "retrieval_success": retrieval_success(result),
    }


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(scores: List[Dict[str, Any]], bert_f1s: List[float]) -> Dict[str, Any]:
    """Compute mean metrics for a list of scored records."""
    n = len(scores)
    if n == 0:
        return {}
    return {
        "n":                     n,
        "em":                    round(_mean([s["em"]                for s in scores]), 4),
        "f1":                    round(_mean([s["f1"]                for s in scores]), 4),
        "bertscore_f1":          round(_mean(bert_f1s),                                4),
        "retrieval_success_rate": round(_mean([s["retrieval_success"] for s in scores]), 4),
        "avg_retrieval_steps":   round(_mean([s["retrieval_steps"]   for s in scores]), 3),
        "avg_tokens":            round(_mean([s["tokens_total"]      for s in scores]), 1),
        "avg_latency_s":         round(_mean([s["latency_s"]         for s in scores]), 3),
    }


# ---------------------------------------------------------------------------
# Full evaluation for one system
# ---------------------------------------------------------------------------

def evaluate_system(
    results: List[Dict[str, Any]],
    system_name: str,
) -> Dict[str, Any]:
    print(f"[metrics] Scoring {len(results)} records for '{system_name}' ...")

    scored = [score_record(r) for r in results]

    # BERTScore — run once in batch over all predictions
    predictions = [r.get("predicted_answer", "") for r in results]
    references  = [r.get("gold_answer", "")       for r in results]
    bert_f1s    = bertscore_batch(predictions, references)

    # Attach bert_f1 to each scored record for slice-level aggregation
    for s, bf in zip(scored, bert_f1s):
        s["bertscore_f1"] = bf

    # ── Overall ────────────────────────────────────────────────────────
    overall = aggregate(scored, bert_f1s)

    # ── By dataset ─────────────────────────────────────────────────────
    by_dataset: Dict[str, Any] = {}
    for ds in ("hotpotqa", "musique"):
        subset = [(s, s["bertscore_f1"]) for s in scored if s["dataset"] == ds]
        if subset:
            ss, bfs = zip(*subset)
            by_dataset[ds] = aggregate(list(ss), list(bfs))

    # ── By hop count (MuSiQue only) ────────────────────────────────────
    by_hops: Dict[str, Any] = {}
    for h in (2, 3, 4):
        subset = [(s, s["bertscore_f1"]) for s in scored
                  if s["dataset"] == "musique" and s["n_hops"] == h]
        if subset:
            ss, bfs = zip(*subset)
            by_hops[str(h)] = aggregate(list(ss), list(bfs))

    return {
        "system":     system_name,
        "overall":    overall,
        "by_dataset": by_dataset,
        "by_hops":    by_hops,
    }


# ---------------------------------------------------------------------------
# Pretty-print comparison table
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def print_comparison(all_evals: List[Dict[str, Any]]) -> None:
    systems = [e["system"] for e in all_evals]
    col_w   = max(len(s) for s in systems) + 2

    def header(title: str) -> None:
        print(f"\n{'─' * 70}")
        print(f"  {title}")
        print(f"{'─' * 70}")

    def row(label: str, values: List[str]) -> None:
        print(f"  {label:<28}" + "".join(f"{v:>{col_w}}" for v in values))

    # Column headers
    header("OVERALL")
    row("", systems)
    row("EM",                    [_pct(e["overall"]["em"])                    for e in all_evals])
    row("Token F1",              [_pct(e["overall"]["f1"])                    for e in all_evals])
    row("BERTScore F1",          [_pct(e["overall"]["bertscore_f1"])          for e in all_evals])
    row("Retrieval Success",     [_pct(e["overall"]["retrieval_success_rate"]) for e in all_evals])
    row("Avg Retrieval Steps",   [f"{e['overall']['avg_retrieval_steps']:.1f}"  for e in all_evals])
    row("Avg Tokens",            [f"{e['overall']['avg_tokens']:.0f}"           for e in all_evals])
    row("Avg Latency (s)",       [f"{e['overall']['avg_latency_s']:.2f}"        for e in all_evals])

    for ds in ("hotpotqa", "musique"):
        header(f"BY DATASET — {ds.upper()}")
        row("", systems)
        row("EM",           [_pct(e["by_dataset"].get(ds, {}).get("em",  0)) for e in all_evals])
        row("Token F1",     [_pct(e["by_dataset"].get(ds, {}).get("f1",  0)) for e in all_evals])
        row("BERTScore F1", [_pct(e["by_dataset"].get(ds, {}).get("bertscore_f1", 0)) for e in all_evals])

    header("BY HOP COUNT (MuSiQue)")
    row("", systems)
    for h in ("2", "3", "4"):
        row(f"EM ({h}-hop)", [_pct(e["by_hops"].get(h, {}).get("em", 0)) for e in all_evals])

    print(f"\n{'─' * 70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _discover_result_files(results_dir: str) -> List[Tuple[str, str]]:
    """Return [(system_name, filepath)] for JSON files in results_dir."""
    pairs = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json") and fname != "evaluation_summary.json":
            name = fname.replace(".json", "")
            pairs.append((name, os.path.join(results_dir, fname)))
    return pairs


def main(
    file_pairs:  List[Tuple[str, str]],
    output_path: str,
) -> None:
    all_evals: List[Dict[str, Any]] = []

    for system_name, fpath in file_pairs:
        print(f"\n[metrics] Loading {fpath} ...")
        with open(fpath, encoding="utf-8") as f:
            results = json.load(f)
        ev = evaluate_system(results, system_name)
        all_evals.append(ev)

    print_comparison(all_evals)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_evals, f, ensure_ascii=False, indent=2)
    print(f"[metrics] Summary saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG systems.")
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Directory containing naive_rag.json, iterative_rag.json, agentic_rag.json",
    )
    parser.add_argument(
        "--files", nargs="+", default=None,
        help="Explicit list of result JSON files (evaluated in order given).",
    )
    parser.add_argument(
        "--output", type=str, default="results/evaluation_summary.json",
    )
    args = parser.parse_args()

    if args.files:
        pairs = [(os.path.basename(f).replace(".json", ""), f) for f in args.files]
    elif args.results_dir:
        pairs = _discover_result_files(args.results_dir)
    else:
        parser.error("Provide --results_dir or --files.")

    if not pairs:
        print("[metrics] No result files found.")
    else:
        main(pairs, args.output)
