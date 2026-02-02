#!/bin/bash
set -euo pipefail

# Cookbook: OPD teacher/student distillation (preference-style)
# Expected: data/distill/prompts.jsonl with {prompt}

mlxsmith distill --data data/distill/prompts.jsonl --teacher cache/mlx/TEACHER --student cache/mlx/STUDENT --mode opd
