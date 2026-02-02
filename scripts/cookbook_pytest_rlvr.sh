#!/bin/bash
set -euo pipefail

# Cookbook: RL with pytest verifier
# Expected: envs/coding.yaml tests entries and verifiers/pytest.py

mlxsmith rft --model runs/sft_0001/adapter --env envs/coding.yaml --verifier verifiers/pytest.py
