# From RAG to Agentic RAG: A Comparative Study on Multi-hop Question Answering

> **Course Final Project — Large Language Models**
> Topic: Retrieval-Augmented Generation & LLM Agents

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Related Work](#2-related-work)
3. [Dataset](#3-dataset)
4. [Methodology](#4-methodology)
5. [Evaluation](#5-evaluation)
6. [Project Structure](#6-project-structure)
7. [Installation & Setup](#7-installation--setup)
8. [How to Run](#8-how-to-run)
9. [Results](#9-results)
10. [References](#10-references)

---

## 1. Problem Statement

### Background

Retrieval-Augmented Generation (RAG) has become the dominant paradigm for grounding Large Language Model (LLM) responses in external knowledge. By retrieving relevant documents at inference time, RAG reduces hallucination and enables the model to answer questions beyond its training knowledge cutoff.

However, standard RAG follows a rigid **single-shot retrieval** pipeline:

```
Query → Retrieve (once) → Concatenate → Generate Answer
```

This design has a critical limitation: **it assumes a single retrieval step is always sufficient**. For simple, factoid questions (e.g., *"Who directed Inception?"*), this works well. But for **multi-hop questions** that require synthesizing information across multiple documents (e.g., *"What year was the university where Einstein worked founded?"*), a single retrieval pass often misses key intermediate facts, leading to incorrect or incomplete answers.

### The Core Challenge

Multi-hop reasoning requires the model to:
1. Identify what is known from the first retrieved document
2. Formulate a follow-up query based on that partial knowledge
3. Retrieve additional evidence iteratively
4. Synthesize all retrieved pieces into a final answer

Standard RAG cannot do this — it has no mechanism for iterative retrieval or self-directed reasoning.

### Research Questions

This project investigates the following questions:

- **RQ1**: How does standard RAG perform on multi-hop questions compared to single-hop questions?
- **RQ2**: Does an Agentic RAG system — capable of iterative retrieval and self-verification — outperform standard RAG on multi-hop QA?
- **RQ3**: What is the cost-accuracy trade-off between Naive RAG, Iterative RAG, and Agentic RAG in terms of retrieval steps and token consumption?

---

## 2. Related Work

### 2.1 Retrieval-Augmented Generation (RAG)

**Lewis et al. (2020)** introduced RAG as a general framework combining a parametric memory (the LLM) with a non-parametric memory (a dense retrieval index). The original RAG model retrieves the top-k passages using a DPR retriever and conditions generation on them. This established the retrieve-then-read paradigm that underpins most modern RAG systems.

**Limitation addressed by our work**: The original RAG performs only a single retrieval step, which is insufficient for multi-hop reasoning tasks.

### 2.2 ReAct — Reasoning + Acting

**Yao et al. (2023)** proposed ReAct, a framework that interleaves chain-of-thought reasoning with action execution (e.g., search, lookup). The model generates reasoning traces and actions in a unified sequence, enabling it to dynamically retrieve information based on intermediate conclusions. ReAct demonstrated significant improvements on multi-hop QA benchmarks including HotpotQA.

**Relevance**: Our Agentic RAG implementation adopts the ReAct paradigm, where the LLM agent decides *when* and *what* to retrieve.

### 2.3 Self-RAG

**Asai et al. (2023)** proposed Self-RAG, which trains the model to adaptively retrieve and critically evaluate retrieved passages through special reflection tokens (Retrieve, ISREL, ISSUP, ISUSE). The model learns to retrieve only when necessary and to filter out irrelevant passages.

**Relevance**: Self-RAG motivates our self-verification component in Agentic RAG, where the agent evaluates whether retrieved evidence is sufficient before generating a final answer.

### 2.4 Iterative and Adaptive Retrieval

**Trivedi et al. (2022)** proposed IRCoT (Interleaving Retrieval with Chain-of-Thought), which alternates between generating a reasoning step and retrieving relevant documents. This demonstrates that tightly coupling retrieval with reasoning substantially improves multi-hop performance.

**Shao et al. (2023)** introduced FLARE (Forward-Looking Active REtrieval), which proactively decides when to retrieve by monitoring the model's confidence in its own generation.

**Relevance**: These works validate the iterative retrieval paradigm and inform the design of our Agentic RAG agent.

### 2.5 RAG Survey

**Gao et al. (2023)** provided a comprehensive survey categorizing RAG into Naive RAG, Advanced RAG, and Modular RAG. This taxonomy directly informs our experimental design, where we implement and compare systems across these three categories.

### Summary of Related Work

| Work | Key Contribution | Limitation |
|------|-----------------|------------|
| Lewis et al. (2020) | Foundational RAG framework | Single-step retrieval only |
| Yao et al. (2023) ReAct | Interleaved reasoning and retrieval | Requires capable base LLM |
| Asai et al. (2023) Self-RAG | Self-reflective retrieval | Requires fine-tuned model |
| Trivedi et al. (2022) IRCoT | CoT-guided iterative retrieval | Specialized for multi-hop only |
| Gao et al. (2023) | RAG taxonomy and survey | No empirical comparison |
| **Ours** | **Systematic comparison of RAG paradigms on multi-hop QA using LangChain** | — |

---

## 3. Dataset

We evaluate on two complementary multi-hop QA benchmarks, totalling **2,000 examples**.

### 3.1 HotpotQA (Primary Benchmark)

| Property | Detail |
|----------|--------|
| **Source** | Yang et al. (2018), Stanford / CMU |
| **Setting** | Distractor setting |
| **Split used** | Validation |
| **Size** | 1,000 examples (randomly sampled) |
| **Task type** | 2-hop open-domain QA |
| **Paragraphs per example** | 10 (2 gold + 8 distractors) |
| **Link** | https://huggingface.co/datasets/hotpot_qa |

HotpotQA is specifically designed for multi-hop reasoning. Each example consists of:
- A question requiring **two-hop reasoning** across two Wikipedia paragraphs
- **10 candidate paragraphs**: 2 gold (containing the answer evidence) + 8 distractors
- Gold answer and supporting fact annotations

**Example**:
```
Question: "In what year was the university where Sergei Tokarev was a professor founded?"
Gold paragraphs:
  [1] "Sergei Tokarev ... professor at Lomonosov Moscow State University ..."
  [2] "Lomonosov Moscow State University ... founded in 1755 ..."
Answer: 1755
```

### 3.2 MuSiQue (Challenging Benchmark)

| Property | Detail |
|----------|--------|
| **Source** | Trivedi et al. (2022), Allen AI |
| **Setting** | Answerable split |
| **Split used** | Validation |
| **Size** | 1,000 examples (randomly sampled) |
| **Task type** | 2–4-hop open-domain QA |
| **Paragraphs per example** | 20 (2–4 gold + rest distractors) |
| **Link** | https://huggingface.co/datasets/dgslibisey/MuSiQue |

MuSiQue extends the difficulty of HotpotQA in two key ways: questions require up to **4 reasoning hops**, and the distractor pool is larger (20 paragraphs vs. 10). Questions are constructed by composing single-hop sub-questions, making them harder to shortcut.

**Hop distribution in our sample**:
| Hops | Count | % |
|------|-------|---|
| 2 | 535 | 53.5% |
| 3 | 318 | 31.8% |
| 4 | 147 | 14.7% |

### 3.3 Why These Two Datasets?

| Criterion | HotpotQA | MuSiQue |
|-----------|----------|---------|
| Established benchmark | ✅ Widely cited since 2018 | ✅ Recent, increasingly adopted |
| Real Wikipedia passages | ✅ | ✅ |
| Fixed 2-hop difficulty | ✅ Good for baseline | ❌ Variable (more realistic) |
| Variable hop count | ❌ | ✅ Enables difficulty analysis |
| Distractor density | Medium (8 distractors) | High (16–18 distractors) |

Together, they allow us to measure not only whether Agentic RAG outperforms Naive RAG, but also **how performance gaps widen as reasoning complexity increases**.

---

## 4. Methodology

### System Overview

We implement and compare **three RAG paradigms** of increasing sophistication, all built with LangChain:

```
┌─────────────────────────────────────────────────────────┐
│                     Query (Question)                     │
└───────────────────┬─────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   Naive RAG   Iterative   Agentic RAG
               RAG
        │           │           │
        ▼           ▼           ▼
    Answer      Answer      Answer
```

### 4.1 Shared Infrastructure

**Vector Store**: All systems share the same retrieval backend — passages from each dataset are embedded using `text-embedding-3-small` and indexed in FAISS for efficient similarity search. Separate indexes are built for HotpotQA and MuSiQue to ensure per-question passage isolation.

**LLM Backend**: `gpt-4o-mini` for generation across all systems, ensuring fair comparison.

**LangChain**: Used as the unified orchestration framework.

### 4.2 Baseline — Naive RAG

The standard single-shot retrieve-then-read pipeline.

```
Query
  │
  ▼
Retrieve top-k passages (k=3) from FAISS
  │
  ▼
Concatenate passages + question into prompt
  │
  ▼
LLM generates answer
```

**Implementation**: `LangChain RetrievalQA` chain with a custom prompt template.

**Expected limitation**: For multi-hop questions, the first retrieval may return only one of the two required gold passages, leading to incomplete answers.

### 4.3 System 2 — Iterative RAG

Extends Naive RAG by performing multiple fixed rounds of retrieval before answering.

```
Query
  │
  ▼
Round 1: Retrieve top-k → extract intermediate facts
  │
  ▼
Round 2: Re-query with updated context → retrieve more
  │
  ▼
(repeat N rounds)
  │
  ▼
LLM generates final answer from accumulated context
```

**Implementation**: `LangChain SequentialChain` with N=2 retrieval rounds.

**Expected improvement**: Two retrieval rounds allow the system to first identify an intermediate entity, then look it up explicitly.

### 4.4 System 3 — Agentic RAG (Core Contribution)

A LangChain ReAct agent that autonomously decides when and what to retrieve, and verifies its own answer before committing.

```
Query
  │
  ▼
Agent thinks: "What do I need to find?"
  │
  ▼
Agent calls Search tool → retrieves documents
  │
  ▼
Agent thinks: "Do I have enough to answer?"
     ├─ No → reformulate query → Search again
     └─ Yes → generate answer
              │
              ▼
         Self-verify: "Does the answer follow from the evidence?"
              ├─ No → retrieve more
              └─ Yes → return final answer
```

**Tools available to the agent**:
- `DocumentSearch`: Semantic search over the FAISS vector store
- `AnswerVerifier`: Checks whether the current evidence supports a candidate answer

**Implementation**: `LangChain AgentExecutor` with `ReAct` prompt template and custom tools.

### System Comparison

| Property | Naive RAG | Iterative RAG | Agentic RAG |
|----------|-----------|---------------|-------------|
| Retrieval steps | 1 (fixed) | N (fixed) | Variable (dynamic) |
| Query reformulation | No | No | Yes |
| Self-verification | No | No | Yes |
| LangChain component | RetrievalQA | SequentialChain | AgentExecutor |
| Token overhead | 1× | ~2× | Variable |
| Framework | LangChain | LangChain | LangChain |

---

## 5. Evaluation

### 5.1 Accuracy Metrics

**Exact Match (EM)**
```
EM = 1  if  normalize(predicted) == normalize(gold)
EM = 0  otherwise
(normalization: lowercase, strip punctuation and articles)
```

**Token-level F1**
```
F1 = 2 × Precision × Recall / (Precision + Recall)
where overlap is computed on whitespace-tokenized strings
```

**BERTScore**
Semantic similarity between predicted and gold answers using contextual embeddings, capturing cases where surface form differs but meaning is preserved.

### 5.2 Efficiency Metrics

| Metric | Description |
|--------|-------------|
| Avg. retrieval steps | Mean number of retrieval calls per question |
| Avg. tokens consumed | Mean total tokens (prompt + completion) per question |
| Latency | Mean wall-clock time per question |

### 5.3 Analysis Dimensions

**By dataset**: HotpotQA vs. MuSiQue results reported separately to assess generalization across benchmarks.

**By question type (HotpotQA)**: *Bridge* questions (find entity A, then look up A's property) vs. *comparison* questions (compare two entities). We report metrics separately to understand where each system excels.

**By hop count (MuSiQue)**: Results broken down by 2-hop, 3-hop, and 4-hop questions to quantify how performance gaps between systems widen as reasoning complexity increases.

**Retrieval success rate**: Whether all required gold passages appeared in the retrieved context (measures retrieval quality independent of generation quality).

**Failure analysis**: Qualitative categorization of errors — missing evidence, wrong retrieval, correct retrieval but wrong generation, etc.

---

## 6. Project Structure

```
rul325-Retrieval-Augmented-Generation-LLM-Agents/
│
├── README.md
│
├── src/
│   ├── data/
│   │   └── prepare_dataset.py      # Download & preprocess HotpotQA + MuSiQue
│   │
│   ├── retriever/
│   │   └── vector_store.py         # FAISS index build & query interface
│   │
│   ├── rag/
│   │   ├── naive_rag.py            # System 1: Naive RAG (RetrievalQA)
│   │   ├── iterative_rag.py        # System 2: Iterative RAG (SequentialChain)
│   │   └── agentic_rag.py          # System 3: Agentic RAG (AgentExecutor)
│   │
│   ├── evaluation/
│   │   └── metrics.py              # EM, F1, BERTScore, efficiency metrics
│   │
│   └── visualization/
│       └── plots.py                # Result charts and comparison figures
│
├── experiments/
│   └── run_all.sh                  # End-to-end experiment runner
│
├── results/                        # JSON output files per system
│
├── notebooks/
│   └── analysis.ipynb              # Interactive analysis and visualization
│
├── data/
│   └── processed/
│       ├── hotpotqa_1000.json      # 1,000 HotpotQA records
│       ├── musique_1000.json       # 1,000 MuSiQue records
│       └── combined_2000.json      # Combined dataset (2,000 total)
│
├── requirements.txt
└── .env.example
```

---

## 7. Installation & Setup

### Prerequisites

- Python 3.9+
- OpenAI API key

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd rul325-LongContext-Process

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

---

## 8. How to Run

### Step 1 — Prepare Datasets

```bash
python src/data/prepare_dataset.py --n_hotpot 1000 --n_musique 1000 --output_dir data/processed/
# Outputs: hotpotqa_1000.json, musique_1000.json, combined_2000.json
```

### Step 2 — Build Vector Store

```bash
python src/retriever/vector_store.py --input data/processed/combined_2000.json --output data/faiss_index/
```

### Step 3 — Run All Systems

```bash
# Run individual systems
python src/rag/naive_rag.py      --data data/processed/combined_2000.json --output results/naive_rag.json
python src/rag/iterative_rag.py  --data data/processed/combined_2000.json --output results/iterative_rag.json
python src/rag/agentic_rag.py    --data data/processed/combined_2000.json --output results/agentic_rag.json

# Or run everything at once
bash experiments/run_all.sh
```

### Step 4 — Evaluate

```bash
python src/evaluation/metrics.py --results_dir results/
```

### Step 5 — Visualize

```bash
python src/visualization/plots.py --results_dir results/
# or
jupyter notebook notebooks/analysis.ipynb
```

---

## 9. Results

*To be updated after experiments.*

### Expected Comparison — HotpotQA (2-hop)

| System | EM | F1 | Avg. Retrieval Steps | Avg. Tokens |
|--------|----|----|----------------------|-------------|
| Naive RAG     | ~30% | ~40% | 1    | ~800   |
| Iterative RAG | ~40% | ~52% | 2    | ~1,400 |
| Agentic RAG   | ~50% | ~62% | ~2.5 | ~2,200 |

### Expected Comparison — MuSiQue (2–4-hop)

| System | EM (2-hop) | EM (3-hop) | EM (4-hop) |
|--------|-----------|-----------|-----------|
| Naive RAG     | ~25% | ~15% | ~8%  |
| Iterative RAG | ~35% | ~25% | ~15% |
| Agentic RAG   | ~45% | ~35% | ~22% |

*Numbers are estimates based on related work. Actual results will be reported upon completion.*

---

## 10. References

1. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.** *NeurIPS 2020.*

2. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). **ReAct: Synergizing Reasoning and Acting in Language Models.** *ICLR 2023.*

3. Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023). **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.** *ICLR 2024.*

4. Trivedi, H., Balasubramanian, N., Khot, T., & Sabharwal, A. (2022). **Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions.** *ACL 2023.*

5. Shao, Z., Gong, Y., Shen, Y., Huang, M., Dolan, B., Jiao, J., & Chen, W. (2023). **Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy.** *EMNLP 2023.*

6. Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., ... & Wang, H. (2023). **Retrieval-Augmented Generation for Large Language Models: A Survey.** *arXiv:2312.10997.*

7. Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W., Salakhutdinov, R., & Manning, C. D. (2018). **HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.** *EMNLP 2018.*

8. Trivedi, H., Balasubramanian, N., Khot, T., & Sabharwal, A. (2022). **MuSiQue: Multihop Questions via Single-hop Question Composition.** *TACL 2022.*

9. Chase, H. (2022). **LangChain.** GitHub. https://github.com/langchain-ai/langchain

---

> **Author:** [Ruimeng Liu]
> **Course:** Large Language Models — Final Project
> **Institution:** [Lehigh University]
> **Date:** 2025
