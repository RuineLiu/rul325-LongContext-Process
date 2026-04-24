# From RAG to Agentic RAG: A Comparative Study on Multi-hop Question Answering

> **Course Final Project — Large Language Models**  
> Topic: Retrieval-Augmented Generation & LLM Agents

---

## Table of Contents

1. [Project Description](#1-project-description)
2. [Data Sources](#2-data-sources)
3. [Required Packages](#3-required-packages)
4. [How to Run](#4-how-to-run)
5. [Results](#5-results)
6. [Project Structure](#6-project-structure)
7. [References](#7-references)

---

## 1. Project Description

### Background

Retrieval-Augmented Generation (RAG) grounds Large Language Model (LLM) responses in external knowledge by retrieving relevant documents at inference time, reducing hallucination and extending the model's effective knowledge beyond its training cutoff.

Standard RAG follows a rigid **single-shot retrieval** pipeline:

```
Query → Retrieve (once) → Concatenate → Generate Answer
```

This design has a critical limitation for **multi-hop questions** — questions that require synthesizing information across multiple documents (e.g., *"What year was the university where Einstein worked founded?"*). A single retrieval pass often misses key intermediate facts, leading to incorrect answers.

### What This Project Does

We implement and systematically compare **three RAG paradigms** of increasing sophistication on multi-hop QA benchmarks:

| System | Description |
|--------|-------------|
| **Naive RAG** | Single-shot retrieve-then-read: retrieve top-k passages once, generate answer |
| **Iterative RAG** | Fixed 2-round retrieval: first round finds intermediate facts, second round refines |
| **Agentic RAG** | LangGraph ReAct agent that dynamically decides when/what to retrieve and self-verifies its answer |

### Research Questions

- **RQ1**: How does standard RAG perform on multi-hop questions compared to simple questions?
- **RQ2**: Does Agentic RAG — with iterative retrieval and self-verification — outperform Naive RAG on multi-hop QA?
- **RQ3**: What is the cost-accuracy trade-off across the three paradigms?

### Key Findings

- All three metrics (EM, Token F1, BERTScore) improve consistently: **Naive < Iterative < Agentic**
- Agentic RAG's advantage grows with question complexity: **+9.3pp on 2-hop HotpotQA, +16.8pp on 2–4-hop MuSiQue**
- The trade-off: Agentic uses **14× more tokens** than Naive (6,465 vs 454 per question)
- Iterative RAG offers the best cost-efficiency: **2× token cost for +7pp EM gain**

---

## 2. Data Sources

We evaluate on two complementary multi-hop QA benchmarks totalling **2,000 examples**, sampled from HuggingFace Datasets.

### HotpotQA

| Property | Detail |
|----------|--------|
| **Paper** | Yang et al., EMNLP 2018 |
| **HuggingFace** | [`hotpot_qa`](https://huggingface.co/datasets/hotpot_qa) |
| **Setting** | Distractor setting, validation split |
| **Size used** | 1,000 examples (randomly sampled) |
| **Task** | 2-hop open-domain QA |
| **Passages per question** | 10 (2 gold + 8 distractors) |

Each question requires reasoning across exactly **two Wikipedia paragraphs**. The 8 distractor paragraphs are designed to mislead retrieval systems.

**Example**:
```
Q: "In what year was the university where Sergei Tokarev was a professor founded?"
Hop 1: Sergei Tokarev → professor at Lomonosov Moscow State University
Hop 2: Lomonosov Moscow State University → founded in 1755
Answer: 1755
```

### MuSiQue

| Property | Detail |
|----------|--------|
| **Paper** | Trivedi et al., TACL 2022 |
| **HuggingFace** | [`dgslibisey/MuSiQue`](https://huggingface.co/datasets/dgslibisey/MuSiQue) |
| **Setting** | Answerable split, validation |
| **Size used** | 1,000 examples (randomly sampled) |
| **Task** | 2–4-hop open-domain QA |
| **Passages per question** | 20 (2–4 gold + rest distractors) |

MuSiQue extends difficulty in two ways: questions require up to **4 reasoning hops**, and the distractor pool is larger. Hop distribution in our sample:

| Hops | Count | % |
|------|-------|---|
| 2 | 535 | 53.5% |
| 3 | 318 | 31.8% |
| 4 | 147 | 14.7% |

### Why These Two Datasets?

HotpotQA provides a well-established 2-hop baseline; MuSiQue adds variable hop counts (2–4), enabling analysis of **how performance gaps widen as reasoning complexity increases**.

---

## 3. Required Packages

Python **3.9+** and an **OpenAI API key** are required.

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Full package list:

| Package | Version | Purpose |
|---------|---------|---------|
| `openai` | ≥1.0 | OpenAI API client |
| `langchain` | ≥0.2 | RAG orchestration framework |
| `langchain-openai` | ≥0.1 | LangChain × OpenAI integration |
| `langchain-community` | ≥0.2 | Callbacks (token counting) |
| `langgraph` | ≥0.1 | ReAct agent for Agentic RAG |
| `faiss-cpu` | ≥1.7.4 | Vector similarity search |
| `sentence-transformers` | ≥2.2 | Sentence embeddings (BERTScore) |
| `transformers` | ≥4.30 | RoBERTa model for BERTScore |
| `torch` | ≥2.0 | Backend for transformers |
| `bert-score` | ≥0.3.13 | Semantic similarity evaluation |
| `datasets` | ≥2.14 | HuggingFace dataset loading |
| `numpy` | ≥1.24 | Numerical operations |
| `pandas` | ≥2.0 | Results analysis |
| `matplotlib` | ≥3.7 | Plotting |
| `seaborn` | ≥0.12 | Statistical visualization |
| `scipy` | ≥1.10 | Paired t-test (significance testing) |
| `tqdm` | ≥4.65 | Progress bars |
| `python-dotenv` | ≥1.0 | Load `.env` API key |
| `jupyter` | ≥1.0 | Interactive notebook analysis |
| `ipykernel` | ≥6.0 | Jupyter kernel |

> **macOS note**: `faiss-cpu` and `numpy` each ship their own OpenMP runtime, which can conflict. The code automatically sets `KMP_DUPLICATE_LIB_OK=TRUE` to suppress this.

---

## 4. How to Run

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/RuineLiu/rul325-Retrieval-Augmented-Generation-LLM-Agents.git
cd rul325-Retrieval-Augmented-Generation-LLM-Agents

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your OpenAI API key
cp .env.example .env
# Edit .env and set: OPENAI_API_KEY=sk-...
```

---

### Option A — Run Everything at Once (Recommended)

```bash
bash experiments/run_all.sh
```

This script runs all 7 steps automatically. It skips steps that are already done (safe to re-run):

```
Step 1: Download & preprocess datasets  (~2 min)
Step 2: Build FAISS embedding index     (~$0.50, ~10 min)
Step 3: Run Naive RAG                   (~$0.30, ~30 min)
Step 4: Run Iterative RAG               (~$0.60, ~60 min)
Step 5: Run Agentic RAG                 (~$3.00, ~2–3 hrs)
Step 6: Evaluate (EM / F1 / BERTScore)  (~5 min)
Step 7: Generate figures                (~1 min)
```

**Useful flags:**

```bash
bash experiments/run_all.sh --limit 50    # Quick smoke-test on 50 questions
bash experiments/run_all.sh --resume      # Resume an interrupted run
```

---

### Option B — Run Steps Individually

**Step 1 — Prepare datasets**
```bash
python src/data/prepare_dataset.py \
    --n_hotpot 1000 --n_musique 1000 \
    --output_dir data/processed/
```

**Step 2 — Build FAISS vector index**
```bash
python src/retriever/vector_store.py \
    --input  data/processed/combined_2000.json \
    --output data/faiss_index/
```

**Step 3 — Run RAG systems**
```bash
python src/rag/naive_rag.py \
    --data  data/processed/combined_2000.json \
    --index data/faiss_index/ \
    --output results/naive_rag.json

python src/rag/iterative_rag.py \
    --data  data/processed/combined_2000.json \
    --index data/faiss_index/ \
    --output results/iterative_rag.json

python src/rag/agentic_rag.py \
    --data  data/processed/combined_2000.json \
    --index data/faiss_index/ \
    --output results/agentic_rag.json
```

Add `--resume` to any of the above to continue an interrupted run without re-processing completed questions.

**Step 4 — Evaluate**
```bash
python src/evaluation/metrics.py --results_dir results/
```

**Step 5 — Visualize**
```bash
# Generate all comparison figures
python src/visualization/plots.py \
    --summary    results/evaluation_summary.json \
    --output_dir results/figures/

# Or open the interactive notebook
jupyter notebook notebooks/analysis.ipynb
jupyter notebook notebooks/extra_analysis.ipynb   # cost & retrieval quality analysis
```

---

### Estimated Cost (Full 2,000-question Run)

| Step | Model | Estimated Cost |
|------|-------|----------------|
| Build index | text-embedding-3-small | ~$0.50 |
| Naive RAG | gpt-4o-mini | ~$0.30 |
| Iterative RAG | gpt-4o-mini | ~$0.60 |
| Agentic RAG | gpt-4o-mini | ~$3.00 |
| **Total** | | **~$4.40** |

> **API rate limits**: The free tier of OpenAI allows 10,000 requests/day. Agentic RAG makes ~3 requests per question, so 2,000 questions requires ~6,000 requests — likely needing 2 days if on the free tier. Use `--resume` to continue across sessions.

---

## 5. Results

All experiments run on 2,000 questions (1,000 HotpotQA + 1,000 MuSiQue) using `gpt-4o-mini`.

### Overall Performance

| System | EM | Token F1 | BERTScore F1 | Retrieval Success | Avg Tokens | Avg Latency |
|--------|-----|----------|--------------|-------------------|------------|-------------|
| Naive RAG | 33.3% | 44.0% | 88.3% | 95.2% | 454 | 0.68s |
| Iterative RAG | 40.2% | 52.2% | 90.0% | 96.9% | 985 | 2.00s |
| **Agentic RAG** | **46.4%** | **60.2%** | **91.9%** | **99.1%** | 6,465 | 6.73s |

### By Dataset

| System | HotpotQA EM | MuSiQue EM |
|--------|------------|-----------|
| Naive RAG | 45.5% | 21.1% |
| Iterative RAG | 52.4% | 28.0% |
| **Agentic RAG** | **54.8%** | **37.9%** |

Agentic RAG's gain over Naive is **+9.3pp on HotpotQA** vs **+16.8pp on MuSiQue** — demonstrating that dynamic retrieval helps most on harder, multi-hop questions.

### By Hop Count (MuSiQue)

| System | 2-hop EM | 3-hop EM | 4-hop EM |
|--------|---------|---------|---------|
| Naive RAG | 25.1% | 15.7% | 18.4% |
| Iterative RAG | 33.8% | 22.0% | 19.7% |
| **Agentic RAG** | **45.8%** | **28.0%** | **30.6%** |

---

## 6. Project Structure

```
rul325-Retrieval-Augmented-Generation-LLM-Agents/
│
├── .env.example                        # Template — copy to .env and add API key
├── requirements.txt                    # All Python dependencies
├── experiments/
│   └── run_all.sh                      # End-to-end experiment runner
│
├── src/
│   ├── data/
│   │   └── prepare_dataset.py          # Download HotpotQA + MuSiQue from HuggingFace
│   ├── retriever/
│   │   └── vector_store.py             # Build & query FAISS embedding index
│   ├── rag/
│   │   ├── naive_rag.py                # System 1: single-shot RAG
│   │   ├── iterative_rag.py            # System 2: 2-round iterative RAG
│   │   └── agentic_rag.py              # System 3: LangGraph ReAct agent
│   ├── evaluation/
│   │   └── metrics.py                  # EM, Token F1, BERTScore, efficiency
│   └── visualization/
│       └── plots.py                    # Comparison figures
│
├── notebooks/
│   ├── analysis.ipynb                  # Main analysis (accuracy, efficiency, errors)
│   └── extra_analysis.ipynb            # Cost-accuracy trade-off + retrieval impact
│
├── data/
│   └── processed/
│       ├── hotpotqa_1000.json
│       ├── musique_1000.json
│       └── combined_2000.json
│
└── results/
    ├── naive_rag.json
    ├── iterative_rag.json
    ├── agentic_rag.json
    ├── evaluation_summary.json
    └── figures/                        # PNG charts
```

---

## 7. References

1. Lewis, P., et al. (2020). **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.** *NeurIPS 2020.*

2. Yao, S., et al. (2023). **ReAct: Synergizing Reasoning and Acting in Language Models.** *ICLR 2023.*

3. Asai, A., et al. (2023). **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.** *ICLR 2024.*

4. Trivedi, H., et al. (2022). **Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions.** *ACL 2023.*

5. Gao, Y., et al. (2023). **Retrieval-Augmented Generation for Large Language Models: A Survey.** *arXiv:2312.10997.*

6. Yang, Z., et al. (2018). **HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.** *EMNLP 2018.*

7. Trivedi, H., et al. (2022). **MuSiQue: Multihop Questions via Single-hop Question Composition.** *TACL 2022.*

8. Chase, H. (2022). **LangChain.** https://github.com/langchain-ai/langchain

---

> **Author:** Ruimeng Liu  
> **Course:** Large Language Models — Final Project  
> **Institution:** Lehigh University  
> **Date:** 2025
