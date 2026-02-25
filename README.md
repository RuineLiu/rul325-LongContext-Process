# 📄 Lost in the Middle: Replication & Mitigation of Positional Bias in Long-Context LLMs

> **Course Final Project — Large Language Models**
> Topic: Long Context Processing

---

## 📌 Table of Contents

1. [Overview](#overview)
2. [Motivation & Background](#motivation--background)
3. [What's New Compared to the Original Work](#whats-new-compared-to-the-original-work)
4. [Dataset](#dataset)
5. [Mitigation Strategies](#mitigation-strategies)
6. [Evaluation Methods](#evaluation-methods)
7. [Project Structure](#project-structure)
8. [Installation & Setup](#installation--setup)
9. [How to Run](#how-to-run)
10. [Results & Visualizations](#results--visualizations)
11. [Discussion](#discussion)
12. [References](#references)

---

## Overview

This project investigates and mitigates the **"Lost in the Middle"** phenomenon in Large Language Models — the well-documented tendency of LLMs to underperform when relevant information is positioned in the **middle** of a long input context, even when that same information placed at the beginning or end yields correct answers.

We:
1. **Replicate** the original findings of Liu et al. (2023) on modern LLMs (GPT-3.5-turbo, Mistral-7B)
2. **Propose and evaluate** multiple novel mitigation strategies beyond what the original paper addressed
3. **Provide a modular, reproducible pipeline** that any researcher can extend

---

## Motivation & Background

### The Core Problem

Modern LLMs support very long contexts (32K–128K tokens), but raw context length does not equal raw reasoning quality. Research has shown that models tend to exhibit a **U-shaped accuracy curve** when relevant information is distributed across a long context:

```
Accuracy
  |
  |  ★                         ★
  |    ★                     ★
  |       ★               ★
  |          ★           ★
  |             ★     ★
  |                ★
  |________________________________
     Beginning   Middle       End
              Document Position
```

This means information buried in the middle is frequently **ignored or forgotten**, regardless of the model's official context window size. This is a critical limitation for real-world applications like document QA, legal review, and multi-document reasoning.

### Original Paper

> **Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023).**
> *Lost in the Middle: How Language Models Use Long Contexts.*
> arXiv:2307.03172

The original paper used the **NaturalQuestions (NQ)** and **KV-retrieval** datasets, tested GPT-3.5 and Claude 2, and focused primarily on **documenting** the phenomenon rather than solving it.

---

## What's New Compared to the Original Work

| Dimension | Original Paper (Liu et al., 2023) | Our Work |
|---|---|---|
| **Goal** | Document the problem | Document + actively mitigate |
| **Datasets** | NaturalQuestions, KV-Retrieval | NaturalQuestions + **HotpotQA** (multi-hop) |
| **Models Tested** | GPT-3.5, Claude 2 | GPT-3.5, Mistral-7B-Instruct |
| **Mitigation** | None proposed | 3 novel strategies implemented & compared |
| **Evaluation** | Exact Match only | Exact Match + **F1 + BERTScore** |
| **Context lengths** | Fixed at 10–20 docs | Variable sweep: 5, 10, 20, 30 documents |
| **Reproducibility** | No public pipeline | Fully open-source, modular code |
| **Multi-hop reasoning** | Not tested | Tested on HotpotQA (cross-document reasoning) |

### Our Key Contributions

1. **Three mitigation strategies** implemented and compared head-to-head on the same benchmarks
2. **Multi-hop extension**: We test the phenomenon on HotpotQA, where answering requires synthesizing *multiple* relevant passages — a more realistic and harder scenario
3. **Scalability analysis**: We vary the number of distractor documents from 5 to 30 to understand how the U-curve degrades as context grows
4. **Lightweight wrapper**: Our mitigation pipeline is model-agnostic and API-compatible — it works as a plug-in preprocessing layer for any LLM backend

---

## Dataset

### Primary Dataset — NaturalQuestions (NQ) Open

| Property | Detail |
|---|---|
| **Source** | Google's Natural Questions (open-domain version) |
| **Size used** | 500 test examples (randomly sampled) |
| **Format** | Question + gold answer + distractor passages |
| **Why** | Directly used in the original paper, allows apples-to-apples comparison |
| **Link** | https://huggingface.co/datasets/nq_open |

Each example is structured as a question paired with one gold-answer passage and K−1 distractor passages (retrieved via BM25 from Wikipedia). We systematically vary the position of the gold passage among K total passages.

### Secondary Dataset — HotpotQA

| Property | Detail |
|---|---|
| **Source** | HotpotQA (distractor setting) |
| **Size used** | 300 test examples |
| **Format** | Multi-hop question + 10 candidate paragraphs (2 gold, 8 distractor) |
| **Why** | Tests whether positional bias worsens when *multiple* relevant passages are needed |
| **Link** | https://huggingface.co/datasets/hotpot_qa |

HotpotQA extends our analysis beyond single-passage retrieval into genuine multi-document reasoning, which is closer to real-world use cases.

### Dataset Preprocessing

All raw datasets are preprocessed using the `src/data/prepare_dataset.py` script, which:
- Samples the specified number of examples
- Builds context windows by inserting gold passage(s) at controlled positions (indices 0 through K−1)
- Serializes each constructed input as a JSON record for reproducibility

---

## Mitigation Strategies

We implement and evaluate **three strategies**, each representing a different approach to solving the problem.

---

### Strategy 1 — Relevance-Based Reordering (Rerank)

**Core idea:** Before feeding documents to the LLM, re-rank them by relevance to the query using a lightweight cross-encoder. Place the most relevant documents at the **beginning and end** of the context, pushing the least relevant to the middle.

**Implementation:**
- Cross-encoder model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (via `sentence-transformers`)
- Each document is scored against the query
- Documents are reordered so that top-scoring docs appear at positions 0 and K−1, the second tier at positions 1 and K−2, and so on (a "relevance sandwich" layout)

**Expected effect:** Moves the gold passage toward the edges, where LLMs pay more attention, without modifying any content.

```
Before reordering:  [D3, D1, GOLD, D5, D2]   ← gold stuck in middle
After reordering:   [GOLD, D5, D3, D1, D2]   ← gold moved to front
```

---

### Strategy 2 — Sandwich Prompting (Repetition)

**Core idea:** Repeat the most likely relevant passage(s) **both at the beginning and the end** of the context, surrounding all other documents. This exploits the model's known primacy and recency biases.

**Implementation:**
- The top-1 passage by BM25 score is duplicated
- It appears as the first and last document in the context
- A brief system instruction is added: *"The key information is provided at the start and end of the context for emphasis."*

**Expected effect:** Guarantees that the most important information is seen at a high-attention position. Increases token count slightly (one passage duplicated).

```
Context layout: [GOLD_COPY | D1, D2, D3, D4 | GOLD_COPY]
```

---

### Strategy 3 — Hierarchical Compression (Summarize-then-Answer)

**Core idea:** Instead of feeding all raw documents directly, first use the LLM to compress each document into a 1–2 sentence summary. Then feed only the summaries as the final context for answering.

**Implementation:**
- Each document is individually summarized using a cheap model call: *"Summarize this passage in 1-2 sentences, focusing on information relevant to: {question}"*
- The summaries (much shorter) are then fed as the context to the final answering call
- Total context length is reduced by ~80%, effectively eliminating the "middle" problem

**Expected effect:** Dramatically reduces context length, removes the positional bias problem, but trades off fine-grained detail. Tests the accuracy-efficiency frontier.

```
Phase 1: Doc1 → Summary1, Doc2 → Summary2, ..., DocK → SummaryK
Phase 2: [Summary1, Summary2, ..., SummaryK] → Answer
```

---

### Strategies At a Glance

| Strategy | Modifies Content? | Extra API Calls | Token Overhead | Target Problem |
|---|---|---|---|---|
| Baseline | No | 0 | 0 | — |
| Rerank | No | 0 | 0 | Position of gold passage |
| Sandwich | No | 0 | +1 passage | Recency/primacy attention |
| Compression | Yes (lossy) | +K calls | −80% | Context length itself |

---

## Evaluation Methods

We use three complementary metrics to capture different aspects of answer quality.

### 1. Exact Match (EM)

The strictest metric. The model's answer is considered correct only if it **exactly matches** (case-insensitive, punctuation-stripped) the gold answer string.

```
EM = 1  if  normalize(predicted) == normalize(gold)
EM = 0  otherwise
```

Used as the primary metric for NaturalQuestions, consistent with the original paper.

### 2. Token-Level F1

A softer metric that measures word overlap between the predicted and gold answers. Particularly useful when the model produces a correct but slightly differently phrased answer.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

where:
  Precision = |predicted_tokens ∩ gold_tokens| / |predicted_tokens|
  Recall    = |predicted_tokens ∩ gold_tokens| / |gold_tokens|
```

### 3. BERTScore (Semantic Similarity)

Captures **semantic** rather than lexical similarity by comparing contextual embeddings of predicted and gold answers using a pretrained BERT model. Important when surface-form variation is expected (e.g., "the United States" vs. "the US").

```
BERTScore-F1 = harmonic mean of precision and recall
               computed over token-level cosine similarities
               using bert-base-uncased embeddings
```

### Evaluation Dimensions

Beyond per-example scoring, we analyze results across two axes:

**Axis 1 — Position:** Accuracy as a function of where the gold passage appears in the context (position 0 through K−1). Produces the signature U-curve plot.

**Axis 2 — Context Length:** Accuracy as a function of total number of documents K (5, 10, 20, 30). Shows how each strategy degrades as context grows.

---

## Project Structure

```
lost-in-the-middle/
│
├── README.md
│
├── src/
│   ├── data/
│   │   ├── prepare_dataset.py        # Download & preprocess NQ and HotpotQA
│   │   └── context_builder.py        # Insert gold passage at controlled positions
│   │
│   ├── models/
│   │   ├── llm_interface.py          # Unified API wrapper (OpenAI / HuggingFace)
│   │   └── reranker.py               # Cross-encoder scoring (sentence-transformers)
│   │
│   ├── strategies/
│   │   ├── baseline.py               # Raw context, no modification
│   │   ├── rerank.py                 # Strategy 1: Relevance-based reordering
│   │   ├── sandwich.py               # Strategy 2: Sandwich prompting
│   │   └── compression.py            # Strategy 3: Hierarchical compression
│   │
│   ├── evaluation/
│   │   ├── metrics.py                # EM, F1, BERTScore implementations
│   │   └── evaluate.py               # Run evaluation over all strategies
│   │
│   └── visualization/
│       ├── plot_ucurve.py            # Position vs. accuracy plots
│       └── plot_comparison.py        # Strategy comparison bar charts
│
├── experiments/
│   ├── run_baseline.sh
│   ├── run_rerank.sh
│   ├── run_sandwich.sh
│   └── run_compression.sh
│
├── results/
│   ├── baseline_nq.json
│   ├── rerank_nq.json
│   ├── sandwich_nq.json
│   └── compression_nq.json
│
├── notebooks/
│   └── analysis.ipynb                # End-to-end analysis and plots
│
├── requirements.txt
└── .env.example                      # API key configuration
```

---

## Installation & Setup

### Prerequisites

- Python 3.9+
- An OpenAI API key (for GPT-3.5-turbo)
- Optional: HuggingFace account token (for Mistral-7B)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/lost-in-the-middle.git
cd lost-in-the-middle

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Key Dependencies (`requirements.txt`)

```
openai>=1.0.0
transformers>=4.35.0
sentence-transformers>=2.2.2
datasets>=2.14.0
evaluate>=0.4.0
bert-score>=0.3.13
rank-bm25>=0.2.2
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
python-dotenv>=1.0.0
```

---

## How to Run

### Step 1 — Prepare Data

```bash
python src/data/prepare_dataset.py \
  --dataset nq \
  --n_samples 500 \
  --n_docs 20 \
  --output_dir data/processed/
```

### Step 2 — Run Baseline (Replication)

```bash
python src/evaluation/evaluate.py \
  --strategy baseline \
  --dataset data/processed/nq_500.json \
  --model gpt-3.5-turbo \
  --output results/baseline_nq.json
```

### Step 3 — Run Mitigation Strategies

```bash
# Strategy 1: Reranking
python src/evaluation/evaluate.py --strategy rerank --dataset data/processed/nq_500.json

# Strategy 2: Sandwich
python src/evaluation/evaluate.py --strategy sandwich --dataset data/processed/nq_500.json

# Strategy 3: Compression
python src/evaluation/evaluate.py --strategy compression --dataset data/processed/nq_500.json
```

### Step 4 — Generate Plots

```bash
# U-curve plot (position vs. accuracy)
python src/visualization/plot_ucurve.py --results_dir results/

# Strategy comparison
python src/visualization/plot_comparison.py --results_dir results/
```

Or run the full interactive analysis in:

```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## Results & Visualizations

### Expected Result 1 — U-Shaped Curve (Replication)

The baseline should reproduce the U-curve: accuracy is highest when the gold document is at position 0 (beginning) or position K−1 (end), and drops significantly in the middle positions.

### Expected Result 2 — Strategy Comparison Table

| Strategy | EM (Overall) | EM (Middle) | F1 | BERTScore | Tokens Used |
|---|---|---|---|---|---|
| Baseline | ~45% | ~30% | ~52% | ~82% | 1× |
| Rerank | ~52% | ~46% | ~59% | ~85% | 1× |
| Sandwich | ~50% | ~44% | ~57% | ~84% | ~1.1× |
| Compression | ~48% | ~47% | ~54% | ~80% | ~0.3× |

> Note: Exact numbers will depend on your sampled subset, model version, and temperature settings. The table above is an expected approximation based on related work.

### Key Expected Findings

- **Reranking** provides the best overall accuracy improvement with zero token overhead — it is the most practical strategy
- **Sandwich prompting** effectively eliminates the middle dip but slightly reduces overall accuracy due to redundant content confusing the model in some cases
- **Compression** nearly eliminates positional bias (the U-curve flattens) but loses fine-grained detail, hurting F1 slightly
- All three strategies meaningfully close the gap between middle-position and edge-position accuracy, reducing the delta by 40–60%

---

## Discussion

### Why Does This Happen?

The positional bias is believed to stem from a combination of:

1. **Training data distribution**: Most training documents are structured so the key answer is at the start or end (e.g., abstracts, conclusions, introductory paragraphs)
2. **Attention decay**: Transformer attention, while theoretically global, shows empirical decay in information propagation over very long sequences
3. **Instruction following bias**: Models are trained to prioritize the beginning of the user turn, inadvertently de-prioritizing mid-context content

### Limitations of Our Work

- We test only on English QA benchmarks; multilingual behavior may differ
- GPT-3.5's exact context handling is opaque (closed-source); Mistral results are more interpretable
- Compression strategy uses the same model for summarization and answering, which may introduce a self-consistency bias
- BERTScore results depend on the reference BERT model version

### Future Work

- Apply mitigations to longer contexts (64K+ tokens) using Claude 3 or GPT-4-turbo
- Explore **fine-tuning** approaches that teach the model to uniformly attend to all context positions
- Test on domain-specific corpora (legal contracts, medical records) where the middle-loss problem has direct economic consequences

---

## References

1. Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023). **Lost in the Middle: How Language Models Use Long Contexts.** arXiv:2307.03172.

2. Kwiatkowski, T., et al. (2019). **Natural Questions: A Benchmark for Question Answering Research.** TACL.

3. Yang, Z., et al. (2018). **HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.** EMNLP.

4. Reimers, N., & Gurevych, I. (2019). **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.** EMNLP.

5. Zhang, T., et al. (2020). **BERTScore: Evaluating Text Generation with BERT.** ICLR.

6. Nogueira, R., & Cho, K. (2019). **Passage Re-ranking with BERT.** arXiv:1901.04085.

---

> **Authors:** [Your Name(s)]
> **Course:** Large Language Models — Final Project
> **Institution:** [Your University]
> **Date:** 2025
