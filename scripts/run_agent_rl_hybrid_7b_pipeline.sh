#!/usr/bin/env bash
set -euo pipefail

DOWNLOAD_PID="${DOWNLOAD_PID:-}"
MODEL_DIR="${FINALIZER_MODEL:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
RESULT_DIR="${RESULT_DIR:-/root/autodl-tmp/results/agent-rl-hybrid-7b}"
CONTROLLERS="${CONTROLLERS:-prompt sft741 sftv2 grpo}"
SMOKE_TASKS="${SMOKE_TASKS:-10}"
FULL_TASKS="${FULL_TASKS:-1000}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/agentrl-venv/bin/python}"
ROOT="${ROOT:-/root/autodl-tmp/personal-rag}"

if [[ -n "$DOWNLOAD_PID" ]]; then
  while kill -0 "$DOWNLOAD_PID" 2>/dev/null; do
    sleep 60
  done
fi

cd "$MODEL_DIR"
sha256sum --check <<'EOF'
a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7  model-00001-of-00004.safetensors
f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185  model-00002-of-00004.safetensors
8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5  model-00003-of-00004.safetensors
1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd  model-00004-of-00004.safetensors
EOF

for required in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json; do
  if [[ ! -f "$required" ]]; then
    echo "Missing finalizer artifact: $MODEL_DIR/$required" >&2
    exit 1
  fi
done

"$PYTHON_BIN" -c '
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
path = "'"$MODEL_DIR"'"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
).eval()
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Return only: OK"}],
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.get_input_embeddings().weight.device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=4)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
'

mkdir -p "$RESULT_DIR"
cd "$ROOT"

TASKS="$SMOKE_TASKS" \
CONTROLLERS="$CONTROLLERS" \
RESULT_DIR="$RESULT_DIR" \
bash scripts/run_agent_rl_hybrid_7b_experiments.sh

TASKS="$FULL_TASKS" \
CONTROLLERS="$CONTROLLERS" \
RESULT_DIR="$RESULT_DIR" \
bash scripts/run_agent_rl_hybrid_7b_experiments.sh
