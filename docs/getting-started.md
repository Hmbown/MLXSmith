# Getting Started

This guide walks through setting up MLXSmith and running your first fine-tuning job from a clean install.

## Prerequisites

- macOS with Apple Silicon (M1 or later)
- Python 3.10, 3.11, or 3.12
- At least 16 GB unified memory (32 GB recommended for 4B+ models)

## Installation

Create a virtual environment and install MLXSmith with all dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install "mlxsmith[all]"
```

## Create a project

```bash
mlxsmith init myproj
cd myproj
```

This creates a workspace with the standard directory layout:

```text
myproj/
  mlxsmith.yaml          # Configuration
  data/sft/              # SFT training data
  data/prefs/            # Preference data
  envs/                  # RL environments
  verifiers/             # Verifier scripts
  eval/suites/           # Evaluation suites
  runs/                  # Training outputs
  cache/                 # Downloaded models
```

## Check your environment

```bash
mlxsmith doctor
```

Verify that `metal: True` and `mlx: True` appear in the output. If MLX is not detected, reinstall with `pip install "mlxsmith[mlx,llm]"`.

## Pull a model

Download and convert a Hugging Face model to MLX format:

```bash
mlxsmith pull mlx-community/Qwen3-4B-Instruct-2507-4bit
```

The model is saved to `cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit`. Pre-quantized community models avoid the conversion step and are fastest to get started with.

To pull and quantize a full-precision model instead:

```bash
mlxsmith pull mlx-community/Qwen3-4B-Instruct-2507-4bit --quantize --q-bits 4
```

Swap in a full-precision model id when you want to quantize from FP16/FP32 weights.

## Pull training data

MLXSmith includes built-in dataset presets:

```bash
mlxsmith data pull --preset alpaca
```

This downloads the Alpaca instruction-following dataset to `data/sft/`. Other presets include `hh-rlhf`, `ultrachat-200k`, `ultrafeedback-binarized-prefs`, and `ultrafeedback-binarized-sft`.

To see all presets:

```bash
mlxsmith data presets
```

## Run SFT

Fine-tune the model with LoRA:

```bash
mlxsmith sft \
  --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --data data/sft
```

Training output is written to `runs/sft_0001/` including:
- `adapter/` — LoRA adapter weights
- `metrics.jsonl` — loss and throughput per step
- `config.snapshot.yaml` — exact configuration used

## Serve the model

Start an OpenAI-compatible API server with the trained adapter:

```bash
mlxsmith serve --model runs/sft_0001/adapter --port 8080
```

Test it:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

## What's next

- [Concepts](concepts.md) — understand the training modes
- [Preference training](cli/preference-training.md) — align with DPO or ORPO
- [Reinforcement training](cli/reinforcement-training.md) — GRPO with verifier rewards
- [Synthetic data](cli/synthetic-data.md) — generate your own training data
- [CLI Reference](cli/README.md) — full command listing
