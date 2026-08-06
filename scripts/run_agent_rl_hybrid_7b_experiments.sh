#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/personal-rag}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/agentrl-venv/bin/python}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen3-1.7B-modelscope}"
FINALIZER_MODEL="${FINALIZER_MODEL:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-/root/autodl-tmp/models/bge-small-en-v1.5}"
RERANKER_MODEL="${RERANKER_MODEL:-/root/autodl-tmp/models/bge-reranker-base}"
DATA_DIR="${DATA_DIR:-$ROOT/data/agent_rl/hotpotqa_validation_1k}"
RESULT_DIR="${RESULT_DIR:-/root/autodl-tmp/results/agent-rl-hybrid-7b}"
TASKS="${TASKS:-1000}"
CONTROLLERS="${CONTROLLERS:-prompt sft741 sftv2 grpo}"

declare -A ADAPTERS=(
  [prompt]=""
  [sft741]="/root/autodl-tmp/results/agent-sft-2k/qwen3-1.7b-qlora/adapter"
  [sftv2]="/root/autodl-tmp/results/agent-sft-v2/qwen3-1.7b-qlora/adapter"
  [grpo]="/root/autodl-tmp/results/agent-grpo/qwen3-1.7b-sftv2-grpo/adapter/policy"
)

mkdir -p "$RESULT_DIR/logs"
cd "$ROOT"

run_args=()
for controller in $CONTROLLERS; do
  if [[ -z "${ADAPTERS[$controller]+x}" ]]; then
    echo "Unknown controller: $controller" >&2
    exit 2
  fi
  adapter_args=()
  if [[ -n "${ADAPTERS[$controller]}" ]]; then
    adapter_args=(--controller-adapter "${ADAPTERS[$controller]}")
  fi
  stem="${controller}_hybrid_rerank_7b_validation${TASKS}"
  "$PYTHON_BIN" scripts/eval_hotpotqa_prompt_policy.py \
    --completion-backend transformers \
    --controller-model "$BASE_MODEL" \
    "${adapter_args[@]}" \
    --finalizer-model "$FINALIZER_MODEL" \
    --data-dir "$DATA_DIR" \
    --partition test \
    --max-tasks "$TASKS" \
    --retrieval-backend hybrid-rerank \
    --vector-store lancedb \
    --retrieval-candidate-top-k 15 \
    --retrieval-rrf-k 60 \
    --reranker-top-k 6 \
    --embedding-model "$EMBEDDING_MODEL" \
    --embedding-device cuda \
    --embedding-batch-size 64 \
    --reranker-model "$RERANKER_MODEL" \
    --reranker-device cuda \
    --reranker-batch-size 16 \
    --max-steps 5 \
    --controller-max-new-tokens 128 \
    --finalizer-max-new-tokens 64 \
    --controller-temperature 0.7 \
    --controller-top-p 0.8 \
    --controller-top-k 20 \
    --seed 42 \
    --dtype bfloat16 \
    --output "$RESULT_DIR/${stem}.json" \
    --trajectories "$RESULT_DIR/${stem}.jsonl" \
    --overwrite \
    > "$RESULT_DIR/logs/${stem}.log" 2>&1
  run_args+=(--run "$controller" "$RESULT_DIR/${stem}.json" "$RESULT_DIR/${stem}.jsonl")
done

if (( ${#run_args[@]} >= 6 )); then
  "$PYTHON_BIN" scripts/compare_agent_evals.py \
    "${run_args[@]}" \
    --bootstrap-samples 2000 \
    --seed 42 \
    --output "$RESULT_DIR/comparison_hybrid_rerank_7b_validation${TASKS}.json" \
    --overwrite
fi
