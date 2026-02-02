#!/bin/bash
set -euo pipefail

# Cookbook: DPO preference tuning
# Expected: data/prefs/train.jsonl with {prompt, chosen, rejected}

mlxsmith pref --model runs/sft_0001/adapter --data data/prefs
