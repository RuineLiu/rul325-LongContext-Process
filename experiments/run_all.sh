#!/usr/bin/env bash
# =============================================================================
# run_all.sh — End-to-end experiment runner
#
# Runs all five pipeline stages in order:
#   1. Prepare datasets       (skip if data already exists)
#   2. Build vector index     (skip if index already exists)
#   3. Run Naive RAG
#   4. Run Iterative RAG
#   5. Run Agentic RAG
#   6. Evaluate all systems
#   7. Generate figures
#
# Usage:
#   bash experiments/run_all.sh                  # full run (2 000 questions)
#   bash experiments/run_all.sh --limit 50       # quick smoke-test
#   bash experiments/run_all.sh --resume         # skip already-done questions
#   bash experiments/run_all.sh --limit 50 --resume
#
# Must be run from the project root:
#   cd rul325-Retrieval-Augmented-Generation-LLM-Agents
#   bash experiments/run_all.sh
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN="\033[0;32m"; YELLOW="\033[1;33m"; RED="\033[0;31m"; NC="\033[0m"
info()    { echo -e "${GREEN}[run_all]${NC} $*"; }
warn()    { echo -e "${YELLOW}[run_all]${NC} $*"; }
section() { echo -e "\n${GREEN}━━━ $* ━━━${NC}"; }
fail()    { echo -e "${RED}[run_all] ERROR:${NC} $*" >&2; exit 1; }

# ── Argument parsing ──────────────────────────────────────────────────────────
LIMIT=""
RESUME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)   LIMIT="$2"; shift 2 ;;
    --limit=*) LIMIT="${1#*=}"; shift ;;
    --resume)  RESUME="--resume"; shift ;;
    *) warn "Unknown argument: $1"; shift ;;
  esac
done

LIMIT_FLAG=""
[[ -n "$LIMIT" ]] && LIMIT_FLAG="--limit $LIMIT"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR="data/processed"
INDEX_DIR="data/faiss_index"
RESULTS_DIR="results"
HOTPOT_JSON="$DATA_DIR/hotpotqa_1000.json"
MUSIQUE_JSON="$DATA_DIR/musique_1000.json"
COMBINED_JSON="$DATA_DIR/combined_2000.json"
INDEX_MARKER="$INDEX_DIR/embeddings.npy"

# ── Sanity checks ─────────────────────────────────────────────────────────────
[[ -f ".env" ]] || fail ".env not found. Copy .env.example and add your OPENAI_API_KEY."
command -v python3 &>/dev/null || fail "python3 not found."

START_TIME=$SECONDS

# ─────────────────────────────────────────────────────────────────────────────
section "STEP 1 — Prepare datasets"
# ─────────────────────────────────────────────────────────────────────────────
if [[ -f "$HOTPOT_JSON" && -f "$MUSIQUE_JSON" && -f "$COMBINED_JSON" ]]; then
  warn "Datasets already exist — skipping download."
else
  info "Downloading HotpotQA + MuSiQue from HuggingFace ..."
  python3 src/data/prepare_dataset.py \
    --n_hotpot 1000 \
    --n_musique 1000 \
    --output_dir "$DATA_DIR"
fi

# ─────────────────────────────────────────────────────────────────────────────
section "STEP 2 — Build FAISS vector index"
# ─────────────────────────────────────────────────────────────────────────────
if [[ -f "$INDEX_MARKER" && -f "$INDEX_DIR/metadata.json" ]]; then
  warn "Index already exists at $INDEX_DIR — skipping embedding."
else
  info "Embedding passages with text-embedding-3-small ..."
  python3 src/retriever/vector_store.py \
    --input  "$COMBINED_JSON" \
    --output "$INDEX_DIR"
fi

mkdir -p "$RESULTS_DIR"

# ─────────────────────────────────────────────────────────────────────────────
section "STEP 3 — Naive RAG"
# ─────────────────────────────────────────────────────────────────────────────
info "Running Naive RAG ... $LIMIT_FLAG $RESUME"
python3 src/rag/naive_rag.py \
  --data   "$COMBINED_JSON" \
  --index  "$INDEX_DIR" \
  --output "$RESULTS_DIR/naive_rag.json" \
  $LIMIT_FLAG $RESUME

# ─────────────────────────────────────────────────────────────────────────────
section "STEP 4 — Iterative RAG"
# ─────────────────────────────────────────────────────────────────────────────
info "Running Iterative RAG ... $LIMIT_FLAG $RESUME"
python3 src/rag/iterative_rag.py \
  --data   "$COMBINED_JSON" \
  --index  "$INDEX_DIR" \
  --output "$RESULTS_DIR/iterative_rag.json" \
  $LIMIT_FLAG $RESUME

# ─────────────────────────────────────────────────────────────────────────────
section "STEP 5 — Agentic RAG"
# ─────────────────────────────────────────────────────────────────────────────
info "Running Agentic RAG ... $LIMIT_FLAG $RESUME"
python3 src/rag/agentic_rag.py \
  --data   "$COMBINED_JSON" \
  --index  "$INDEX_DIR" \
  --output "$RESULTS_DIR/agentic_rag.json" \
  $LIMIT_FLAG $RESUME

# ─────────────────────────────────────────────────────────────────────────────
section "STEP 6 — Evaluate"
# ─────────────────────────────────────────────────────────────────────────────
info "Computing EM / F1 / BERTScore ..."
python3 src/evaluation/metrics.py \
  --results_dir "$RESULTS_DIR" \
  --output      "$RESULTS_DIR/evaluation_summary.json"

# ─────────────────────────────────────────────────────────────────────────────
section "STEP 7 — Generate figures"
# ─────────────────────────────────────────────────────────────────────────────
info "Plotting comparison charts ..."
python3 src/visualization/plots.py \
  --summary    "$RESULTS_DIR/evaluation_summary.json" \
  --output_dir "$RESULTS_DIR/figures"

# ─────────────────────────────────────────────────────────────────────────────
ELAPSED=$(( SECONDS - START_TIME ))
section "ALL DONE  (${ELAPSED}s)"
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "  Results  → $RESULTS_DIR/"
echo "  Figures  → $RESULTS_DIR/figures/"
echo "  Summary  → $RESULTS_DIR/evaluation_summary.json"
echo ""
