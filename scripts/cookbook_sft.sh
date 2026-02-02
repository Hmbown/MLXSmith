#!/bin/bash
set -euo pipefail

# Cookbook: SFT on a JSONL dataset
# Expected: data/sft/train.jsonl exists with {prompt, response}

mlxsmith sft --model cache/mlx/YOUR_MODEL --data data/sft
