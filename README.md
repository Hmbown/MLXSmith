# MLXSmith

[![PyPI](https://img.shields.io/pypi/v/mlxsmith)](https://pypi.org/project/mlxsmith/)
[![CI](https://github.com/Hmbown/MLXSmith/actions/workflows/ci.yml/badge.svg)](https://github.com/Hmbown/MLXSmith/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/Hmbown/MLXSmith)](LICENSE)

Fine-tune language models on Apple Silicon. SFT, preference optimization, reinforcement learning, distillation, and serving — all native to MLX.

**Status:** Alpha (v0.1.8) · Validated on Qwen3-4B

---

## Features

- **Supervised fine-tuning** — LoRA and QLoRA with configurable optimizers
- **Preference optimization** — DPO, ORPO, IPO, CPO, SimPO, and more
- **Reinforcement learning** — GRPO with verifier-based rewards
- **Knowledge distillation** — Offline and online preference distillation
- **KTO** — Kahneman-Tversky Optimization from binary feedback
- **Online DPO** — Live preference tuning with LLM judge scoring
- **Self-verification training** — Policy gradient from self-assessed rewards
- **Synthetic data generation** — Generate, evolve, and filter training data
- **External model backends** — Use Codex, Claude, Gemini CLIs or any OpenAI-compatible API for data generation and judging
- **Recursive training** — Self-improving RLM loop with task generation and gating
- **Serving** — OpenAI-compatible API with streaming
- **Web dashboard (Next.js)** — Models, adapters, training, eval, chat, and serving UI
- **Environment plugins** — Reusable task and verifier packages for RL training
- **Experimental mHC adapters** — Optional block-local mHC patching for MLX transformer blocks (not a speedup)

## Requirements

- macOS with Apple Silicon (M1 or later)
- Python 3.10+

Data tools, configuration, and project scaffolding work on any platform.

## Install

```bash
pip install "mlxsmith[all]"
```

<details>
<summary>Selective install</summary>

```bash
# Core only (data tools, config, scaffolding)
pip install mlxsmith

# Apple Silicon training
pip install "mlxsmith[mlx,llm]"

# Training + serving
pip install "mlxsmith[mlx,llm,serve]"
```

</details>

## Quickstart

```bash
# 1. Create a project
mlxsmith init myproj && cd myproj

# 2. Verify your environment
mlxsmith doctor

# 3. Pull a model
mlxsmith pull mlx-community/Qwen3-4B-Instruct-2507-4bit

# 4. Pull training data
mlxsmith data pull --preset alpaca

# 5. Fine-tune
mlxsmith sft \
  --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --data data/sft

# 6. Serve the result
mlxsmith serve --model runs/sft_0001/adapter --port 8080
```

See [Getting Started](docs/getting-started.md) for a complete walkthrough.

## Web Dashboard (Optional)

Run the API server, then start the Next.js dashboard:

```bash
# Terminal 1: start the OpenAI-compatible API
mlxsmith serve --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit --port 8080

# Terminal 2: start the dashboard
cd apps/web
npm install
npm run dev
```

The dashboard defaults to `http://localhost:8080` for the API base URL (change in Settings if needed).

## Training Modes

| Mode | Command | Input Format | Use Case |
|------|---------|-------------|----------|
| [SFT](docs/cli/sft.md) | `mlxsmith sft` | `{prompt, response}` | Instruction-following via LoRA |
| [Preference](docs/cli/preference-training.md) | `mlxsmith pref` | `{prompt, chosen, rejected}` | Alignment with DPO, ORPO, and others |
| [KTO](docs/cli/kto.md) | `mlxsmith kto` | `{prompt, response, label}` | Binary good/bad feedback |
| [GRPO](docs/cli/reinforcement-training.md) | `mlxsmith rft` | Environment + verifier | Reward-driven reinforcement learning |
| [Online DPO](docs/cli/online-dpo.md) | `mlxsmith online-dpo` | `{prompt}` | Online preference with LLM judge |
| [Self-verify](docs/cli/self-verify.md) | `mlxsmith self-verify` | `{prompt}` | Self-verification reward signal |
| [Distillation](docs/cli/distillation.md) | `mlxsmith distill` | `{prompt}` | Teacher-to-student transfer |
| [Judge](docs/cli/judge.md) | `mlxsmith judge` | Judge-format data | Train a scoring model |
| [Pipeline](docs/cli/sft.md#pipeline) | `mlxsmith pipeline` | Combined | SFT then Pref then RFT then RLM |

See [Concepts](docs/concepts.md) for an explanation of each training mode.

## Tools

| Tool | Command | Description |
|------|---------|-------------|
| [Data](docs/cli/data.md) | `mlxsmith data` | Import, split, validate, and pull datasets |
| [Synthetic](docs/cli/synthetic-data.md) | `mlxsmith synthetic` | Generate and evolve training data |
| [Eval](docs/cli/eval-and-bench.md) | `mlxsmith eval` | Run evaluation suites with pass@k |
| [Bench](docs/cli/eval-and-bench.md) | `mlxsmith bench` | Benchmark inference and training throughput |
| [Serve](docs/cli/serving.md) | `mlxsmith serve` | OpenAI-compatible model server |
| [RLM](docs/cli/rlm.md) | `mlxsmith rlm` | Recursive training loop + REPL-based inference |

## External Model Backends

MLXSmith can use powerful cloud models for synthetic data generation and judging while keeping fine-tuning local on Apple Silicon.

Supported backends:

- `cli` — shell out to Codex/Claude/Gemini CLIs (or any command you provide)
- `openai` — call any OpenAI-compatible Chat Completions endpoint

Note: training commands (`sft`, `pref`, `rft`, `rlm` loop) still require a local training backend like `mlx-lm`.

**CLI Backend** — Shell out to Codex, Claude, or Gemini CLIs:

```bash
# Use a CLI model for prompt generation
export MLXSMITH__MODEL__BACKEND=cli
export MLXSMITH_CLI_CODEX_CMD='codex exec --full-auto --model gpt-5.2'

# If your CLI expects the prompt as an argument instead of stdin:
# export MLXSMITH_CLI_PROMPT_FLAG='--prompt'

mlxsmith synthetic prompts \
  --model codex \
  --seed-prompts data/seeds.jsonl \
  --num 100 \
  --out data/prompts.jsonl

# Use a CLI model as judge for filtering
mlxsmith synthetic sft \
  --model codex \
  --judge-backend cli \
  --judge-model claude \
  --prompts data/prompts.jsonl \
  --out data/sft.jsonl
```

**OpenAI Backend** — Use any OpenAI-compatible API:

```bash
export MLXSMITH__MODEL__BACKEND=openai
export OPENAI_API_KEY="sk-..."
export MLXSMITH_API_BASE="https://api.openai.com/v1"  # or any compatible endpoint

mlxsmith synthetic prompts \
  --model gpt-4o \
  --out data/prompts.jsonl
```

This enables cloud-quality data generation with local training — use frontier models to create and filter training data, then fine-tune efficiently on your Mac.

## Documentation

| Section | Description |
|---------|-------------|
| [Getting Started](docs/getting-started.md) | Full setup walkthrough |
| [Concepts](docs/concepts.md) | Training modes explained |
| [CLI Reference](docs/cli/README.md) | All commands with examples |
| [Verifiers](docs/VERIFIERS.md) | Verifier API and composition |
| [Environments](docs/ENVIRONMENTS.md) | Task environment plugins |
| [Project Format](docs/PROJECT_FORMAT.md) | Run artifacts and layout |
| [Configuration](docs/cli/configuration.md) | Config system and options |
| [Compatibility](docs/COMPATIBILITY.md) | Tested versions and models |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and fixes |
| [FAQ](docs/FAQ.md) | Frequently asked questions |
| [Contributing](CONTRIBUTING.md) | How to contribute and run tests |
| [Changelog](CHANGELOG.md) | Release notes |

## License

MIT
