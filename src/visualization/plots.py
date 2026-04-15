"""
plots.py
--------
Generate comparison figures from evaluation_summary.json.

Figures produced (saved to --output_dir):
    1. accuracy_overall.png      — EM / F1 / BERTScore bar chart (all systems)
    2. accuracy_by_dataset.png   — EM side-by-side for HotpotQA vs MuSiQue
    3. accuracy_by_hops.png      — EM line chart by hop count (MuSiQue)
    4. efficiency.png            — Retrieval steps / tokens / latency (3-panel)
    5. retrieval_success.png     — Gold passage hit-rate per system

Usage:
    python src/visualization/plots.py --results_dir results/
    python src/visualization/plots.py \
        --summary results/evaluation_summary.json \
        --output_dir results/figures/
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

PALETTE   = ["#4C72B0", "#55A868", "#C44E52"]   # blue / green / red
SYSTEM_LABELS = {
    "naive_rag":      "Naive RAG",
    "iterative_rag":  "Iterative RAG",
    "agentic_rag":    "Agentic RAG",
}
FIG_DPI = 150


def _style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi":      FIG_DPI,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    })


def _label(system: str) -> str:
    return SYSTEM_LABELS.get(system, system.replace("_", " ").title())


def _save(fig: plt.Figure, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Figure 1 — Overall accuracy
# ---------------------------------------------------------------------------

def plot_accuracy_overall(
    evals: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    systems = [_label(e["system"]) for e in evals]
    metrics = ["em", "f1", "bertscore_f1"]
    labels  = ["Exact Match", "Token F1", "BERTScore F1"]

    x     = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (ev, color) in enumerate(zip(evals, PALETTE)):
        values = [ev["overall"][m] * 100 for m in metrics]
        bars   = ax.bar(x + i * width, values, width, label=systems[i],
                        color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_title("Accuracy Comparison — Overall")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left")

    _save(fig, os.path.join(output_dir, "accuracy_overall.png"))


# ---------------------------------------------------------------------------
# Figure 2 — Accuracy by dataset
# ---------------------------------------------------------------------------

def plot_accuracy_by_dataset(
    evals: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    datasets      = ["hotpotqa", "musique"]
    dataset_labels = ["HotpotQA (2-hop)", "MuSiQue (2–4-hop)"]
    systems       = [_label(e["system"]) for e in evals]

    x     = np.arange(len(datasets))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, metric, title in zip(
        axes,
        ["em", "f1"],
        ["Exact Match (%)", "Token F1 (%)"],
    ):
        for i, (ev, color) in enumerate(zip(evals, PALETTE)):
            values = [
                ev["by_dataset"].get(ds, {}).get(metric, 0) * 100
                for ds in datasets
            ]
            bars = ax.bar(x + i * width, values, width,
                          label=systems[i], color=color, alpha=0.85)
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=7.5,
                )

        ax.set_xticks(x + width)
        ax.set_xticklabels(dataset_labels)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)

    fig.suptitle("Accuracy by Dataset", fontsize=13, fontweight="bold")
    _save(fig, os.path.join(output_dir, "accuracy_by_dataset.png"))


# ---------------------------------------------------------------------------
# Figure 3 — Accuracy by hop count (MuSiQue)
# ---------------------------------------------------------------------------

def plot_accuracy_by_hops(
    evals: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    hops    = ["2", "3", "4"]
    systems = [_label(e["system"]) for e in evals]

    fig, ax = plt.subplots(figsize=(7, 5))

    for ev, color in zip(evals, PALETTE):
        em_by_hop = [
            ev["by_hops"].get(h, {}).get("em", 0) * 100
            for h in hops
        ]
        ax.plot(
            hops, em_by_hop,
            marker="o", linewidth=2, markersize=7,
            label=_label(ev["system"]), color=color,
        )
        for h, val in zip(hops, em_by_hop):
            ax.annotate(
                f"{val:.1f}",
                xy=(h, val),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlabel("Number of Reasoning Hops")
    ax.set_ylabel("Exact Match (%)")
    ax.set_title("EM by Hop Count — MuSiQue")
    ax.set_ylim(0, max(
        ev["by_hops"].get("2", {}).get("em", 0) for ev in evals
    ) * 100 * 1.25 + 5)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    _save(fig, os.path.join(output_dir, "accuracy_by_hops.png"))


# ---------------------------------------------------------------------------
# Figure 4 — Efficiency (3-panel)
# ---------------------------------------------------------------------------

def plot_efficiency(
    evals: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    systems = [_label(e["system"]) for e in evals]
    x       = np.arange(len(systems))

    panels = [
        ("avg_retrieval_steps", "Avg Retrieval Steps",  "Steps"),
        ("avg_tokens",          "Avg Tokens / Question", "Tokens"),
        ("avg_latency_s",       "Avg Latency (s)",       "Seconds"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    for ax, (key, title, ylabel) in zip(axes, panels):
        values = [e["overall"][key] for e in evals]
        bars   = ax.bar(x, values, color=PALETTE[:len(evals)], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:.1f}" if key != "avg_tokens" else f"{val:.0f}",
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(systems, rotation=12, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    fig.suptitle("Efficiency Comparison", fontsize=13, fontweight="bold")
    _save(fig, os.path.join(output_dir, "efficiency.png"))


# ---------------------------------------------------------------------------
# Figure 5 — Retrieval success rate
# ---------------------------------------------------------------------------

def plot_retrieval_success(
    evals: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    systems = [_label(e["system"]) for e in evals]
    values  = [e["overall"]["retrieval_success_rate"] * 100 for e in evals]
    x       = np.arange(len(systems))

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x, values, color=PALETTE[:len(evals)], alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Retrieval Success Rate\n(≥1 Gold Passage Retrieved)")
    ax.set_ylim(0, 110)

    _save(fig, os.path.join(output_dir, "retrieval_success.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(summary_path: str, output_dir: str) -> None:
    _style()
    os.makedirs(output_dir, exist_ok=True)

    with open(summary_path, encoding="utf-8") as f:
        evals: List[Dict[str, Any]] = json.load(f)

    print(f"[plots] Generating figures for {len(evals)} system(s) ...")

    plot_accuracy_overall(evals, output_dir)
    plot_accuracy_by_dataset(evals, output_dir)
    plot_accuracy_by_hops(evals, output_dir)
    plot_efficiency(evals, output_dir)
    plot_retrieval_success(evals, output_dir)

    print(f"[plots] All figures saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise RAG evaluation results.")
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Directory that contains evaluation_summary.json (auto-detected).",
    )
    parser.add_argument(
        "--summary", type=str, default=None,
        help="Explicit path to evaluation_summary.json.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to write PNG files (defaults to <results_dir>/figures/).",
    )
    args = parser.parse_args()

    if args.summary:
        summary_path = args.summary
        results_dir  = os.path.dirname(args.summary)
    elif args.results_dir:
        summary_path = os.path.join(args.results_dir, "evaluation_summary.json")
        results_dir  = args.results_dir
    else:
        parser.error("Provide --results_dir or --summary.")

    output_dir = args.output_dir or os.path.join(results_dir, "figures")
    main(summary_path, output_dir)
