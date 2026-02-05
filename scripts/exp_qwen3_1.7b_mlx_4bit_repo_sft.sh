#!/usr/bin/env bash
set -euo pipefail

# Repo-specific SFT for Qwen/Qwen3-1.7B-MLX-4bit.
#
# This builds a small "MLXSmith repo assistant" dataset from prompts and
# generates responses using `codex exec` (batched), then fine-tunes Qwen with LoRA.
#
# Usage:
#   ./scripts/exp_qwen3_1.7b_mlx_4bit_repo_sft.sh
#
# Optional knobs:
#   NUM=300 BATCH=6 ITEX=2000 LR=2e-4 ./scripts/exp_qwen3_1.7b_mlx_4bit_repo_sft.sh
#   SMOKE_SERVE=1 ./scripts/exp_qwen3_1.7b_mlx_4bit_repo_sft.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_ID="Qwen/Qwen3-1.7B-MLX-4bit"
MODEL_DIR="cache/mlx/Qwen__Qwen3-1.7B-MLX-4bit"
CFG="qwen3_1.7b_mlx_4bit_repo.yaml"

PROMPTS="${PROMPTS:-data/mlxsmith_prompts.jsonl}"
SFT_JSONL="${SFT_JSONL:-data/mlxsmith_sft.jsonl}"
SFT_DIR="${SFT_DIR:-data/repo_sft}"

NUM="${NUM:-300}"
BATCH="${BATCH:-4}"
ITERS="${ITERS:-2000}"
LR="${LR:-2e-4}"

if [[ ! -d "$MODEL_DIR" ]]; then
  mlxsmith pull "$MODEL_ID"
fi

if [[ ! -s "$PROMPTS" ]]; then
  echo "Missing prompts: $PROMPTS"
  echo "Tip: generate via ./scripts/make_repo_seed_prompts.py and/or ./scripts/codex_batch_prompts.py"
  exit 1
fi

echo "==> Generating SFT pairs (num=$NUM batch=$BATCH) -> $SFT_JSONL"
./scripts/codex_batch_sft.py \
  --prompts "$PROMPTS" \
  --out "$SFT_JSONL" \
  --num "$NUM" \
  --batch-size "$BATCH"

echo "==> Splitting -> $SFT_DIR"
python3 -c "import shutil, pathlib; shutil.rmtree(pathlib.Path('${SFT_DIR}'), ignore_errors=True)"
mlxsmith data split --in "$SFT_JSONL" --out-dir "$SFT_DIR" --valid 0.05 --test 0.02

echo "==> Training Qwen3-1.7B repo adapter (iters=$ITERS lr=$LR)"
mlxsmith sft -c "$CFG" --model "$MODEL_DIR" --data "$SFT_DIR" --iters "$ITERS" --lr "$LR"

LATEST_ADAPTER="$(ls -dt runs/sft_* | head -n 1)/adapter"
echo "==> Latest adapter: $LATEST_ADAPTER"

if [[ "${SMOKE_SERVE:-0}" == "1" ]]; then
  PORT="${PORT:-8099}"
  HOST="127.0.0.1"
  echo "==> Serving $LATEST_ADAPTER on http://$HOST:$PORT"
  mlxsmith serve -c "$CFG" --model "$LATEST_ADAPTER" --host "$HOST" --port "$PORT" >"/tmp/mlxsmith_serve_${PORT}.log" 2>&1 &
  PID="$!"
  trap 'kill "$PID" 2>/dev/null || true' EXIT

  for _ in $(seq 1 30); do
    if curl -sSf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  curl -sSf "http://$HOST:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"mlxsmith-qwen3-1.7b","messages":[{"role":"user","content":"What does `mlxsmith pipeline` do? Give a minimal example."}],"max_tokens":220,"temperature":0.2}' \
    | head -c 800
  echo

  kill "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
  trap - EXIT
fi
