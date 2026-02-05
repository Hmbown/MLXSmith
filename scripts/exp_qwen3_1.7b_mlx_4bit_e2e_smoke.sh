#!/usr/bin/env bash
set -euo pipefail

# End-to-end smoke test for MLXSmith on Qwen/Qwen3-1.7B-MLX-4bit:
#   pull -> pipeline (SFT -> Pref -> RFT -> RLM)
#
# Usage:
#   ./scripts/exp_qwen3_1.7b_mlx_4bit_e2e_smoke.sh
#
# Optional (also smoke-test serving after training):
#   SMOKE_SERVE=1 ./scripts/exp_qwen3_1.7b_mlx_4bit_e2e_smoke.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_ID="Qwen/Qwen3-1.7B-MLX-4bit"
MODEL_DIR="cache/mlx/Qwen__Qwen3-1.7B-MLX-4bit"
CFG="examples/qwen3_1.7b_mlx_4bit_smoke.yaml"

if [[ ! -d "$MODEL_DIR" ]]; then
  mlxsmith pull "$MODEL_ID"
fi

mlxsmith pipeline \
  -c "$CFG" \
  --model "$MODEL_DIR" \
  --data-sft data/sft \
  --data-pref data/prefs \
  --env envs/coding.yaml \
  --verifier verifiers/regex.py

if [[ "${SMOKE_SERVE:-0}" == "1" ]]; then
  ADAPTER_DIR="$(ls -dt runs/rft_* | head -n 1)/adapter"
  PORT="${PORT:-8099}"
  HOST="127.0.0.1"

  echo "==> Serving $ADAPTER_DIR on http://$HOST:$PORT"
  mlxsmith serve -c "$CFG" --model "$ADAPTER_DIR" --host "$HOST" --port "$PORT" >"/tmp/mlxsmith_serve_${PORT}.log" 2>&1 &
  PID="$!"
  trap 'kill "$PID" 2>/dev/null || true' EXIT

  # Wait for /health
  for _ in $(seq 1 30); do
    if curl -sSf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  curl -sSf "http://$HOST:$PORT/health"
  echo

  curl -sSf "http://$HOST:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3-1.7b","messages":[{"role":"user","content":"Say hello in exactly three words."}],"max_tokens":16,"temperature":0}' \
    | head -c 400
  echo

  kill "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
  trap - EXIT
fi

