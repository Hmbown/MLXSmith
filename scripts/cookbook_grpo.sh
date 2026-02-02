#!/bin/bash
set -euo pipefail

# Cookbook: GRPO-style verifier-driven RL
# Expected: envs/coding.yaml exists

mlxsmith rft --model runs/sft_0001/adapter --env envs/coding.yaml --verifier verifiers/regex.py
