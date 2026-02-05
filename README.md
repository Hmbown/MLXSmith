# MLXSmith

[![PyPI](https://img.shields.io/pypi/v/mlxsmith)](https://pypi.org/project/mlxsmith/)
[![CI](https://github.com/Hmbown/MLXSmith/actions/workflows/ci.yml/badge.svg)](https://github.com/Hmbown/MLXSmith/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/mlxsmith)](https://pypi.org/project/mlxsmith/)
[![License](https://img.shields.io/github/license/Hmbown/MLXSmith)](LICENSE)

**Fine-tune language models on Apple Silicon.**

MLXSmith is a training toolkit built on [MLX](https://github.com/ml-explore/mlx) that brings supervised fine-tuning, preference optimization, reinforcement learning, knowledge distillation, and model serving to your Mac. Every training algorithm runs natively on the Metal GPU — no cloud required.

> **Status:** Alpha (v0.1.9). Validated on Qwen3-4B and Qwen3-1.7B.

## Installation

**Requirements:** macOS with Apple Silicon (M1 or later) and Python 3.10+. Data tools, configuration, and project scaffolding work on any platform.

```bash
pip install "mlxsmith[all]"
```

<details>
<summary>Install only what you need</summary>

```bash
# Core only — data tools, config, scaffolding (any platform)
pip install mlxsmith

# Training on Apple Silicon
pip install "mlxsmith[mlx,llm]"

# Training + serving
pip install "mlxsmith[mlx,llm,serve]"
```

</details>

## Quick Start

```bash
# Create a project
mlxsmith init myproj && cd myproj

# Verify your environment
mlxsmith doctor

# Download a model
mlxsmith pull mlx-community/Qwen3-4B-Instruct-2507-4bit

# Download training data
mlxsmith data pull --preset alpaca

# Fine-tune with LoRA
mlxsmith sft \
  --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --data data/sft

# Serve the result
mlxsmith serve --model runs/sft_0001/adapter --port 8080
```

See [Getting Started](docs/getting-started.md) for a complete walkthrough.

## Training

MLXSmith supports the full model improvement pipeline — from supervised learning through reinforcement and distillation.

| Mode | Command | Data Format | Description |
|------|---------|-------------|-------------|
| [SFT](docs/cli/sft.md) | `mlxsmith sft` | `{prompt, response}` | Supervised fine-tuning with LoRA/QLoRA |
| [Preference](docs/cli/preference-training.md) | `mlxsmith pref` | `{prompt, chosen, rejected}` | DPO, ORPO, IPO, CPO, SimPO, TDPO |
| [KTO](docs/cli/kto.md) | `mlxsmith kto` | `{prompt, response, label}` | Binary good/bad feedback |
| [GRPO](docs/cli/reinforcement-training.md) | `mlxsmith rft` | Environment + verifier | Reward-driven reinforcement learning |
| [Online DPO](docs/cli/online-dpo.md) | `mlxsmith online-dpo` | `{prompt}` | Live preference tuning with LLM judge |
| [Self-Verify](docs/cli/self-verify.md) | `mlxsmith self-verify` | `{prompt}` | Policy gradient from self-assessed rewards |
| [Distillation](docs/cli/distillation.md) | `mlxsmith distill` | `{prompt}` | Teacher-to-student knowledge transfer |
| [Pipeline](docs/cli/sft.md#pipeline) | `mlxsmith pipeline` | Combined | Chain SFT, preference, RFT, and RLM stages |

See [Concepts](docs/concepts.md) for an explanation of each training mode.

## Tools

| Tool | Command | Description |
|------|---------|-------------|
| [Data](docs/cli/data.md) | `mlxsmith data` | Import, split, validate, and download datasets |
| [Synthetic](docs/cli/synthetic-data.md) | `mlxsmith synthetic` | Generate and evolve training data |
| [Eval](docs/cli/eval-and-bench.md) | `mlxsmith eval` | Run evaluation suites with pass@k metrics |
| [Bench](docs/cli/eval-and-bench.md) | `mlxsmith bench` | Benchmark inference and training throughput |
| [Serve](docs/cli/serving.md) | `mlxsmith serve` | OpenAI-compatible API server with streaming |
| [RLM](docs/cli/rlm.md) | `mlxsmith rlm` | Recursive self-improving training loop |

## External Model Backends

Use cloud models for data generation and judging while keeping training local on Apple Silicon.

**CLI backend** — shell out to Codex, Claude, or Gemini:

```bash
export MLXSMITH__MODEL__BACKEND=cli
export MLXSMITH_CLI_CODEX_CMD='codex exec --full-auto --model gpt-5.2'

mlxsmith synthetic prompts \
  --model codex \
  --seed-prompts data/seeds.jsonl \
  --num 100 \
  --out data/prompts.jsonl
```

**OpenAI backend** — any OpenAI-compatible API:

```bash
export MLXSMITH__MODEL__BACKEND=openai
export OPENAI_API_KEY="sk-..."

mlxsmith synthetic prompts \
  --model gpt-4o \
  --out data/prompts.jsonl
```

Training commands (`sft`, `pref`, `rft`, `rlm`) still require a local MLX backend.

## Web Dashboard

MLXSmith includes a Next.js dashboard for managing models, training runs, evaluation, chat, and serving.

```bash
# Terminal 1: start the API
mlxsmith serve --model <model-or-adapter> --port 8080

# Terminal 2: start the dashboard
cd apps/web && npm install && npm run dev
```

A native macOS app is also available under `apps/macos/`.

## Architecture

```
src/mlxsmith/
├── cli.py                  # CLI entry point (Typer)
├── config.py               # Pydantic config with env/file/CLI precedence
├── train/                  # Training algorithms (SFT, DPO, GRPO, KTO, ...)
├── rlm/                    # Recursive language model training loop
├── llm/                    # Backend abstraction (MLX, OpenAI, CLI, mock)
├── verifiers/              # Reward verifiers (regex, pytest, JSON schema, LLM judge, ...)
├── sdk/                    # Loss registry, training/sampling clients
├── api/                    # FastAPI handlers and schemas
├── orchestrator/           # Multi-process job scheduling
├── envs/                   # Environment plugin system
└── server.py               # OpenAI-compatible serving
```

## Configuration

MLXSmith uses a layered configuration system. Settings are resolved in order of precedence:

1. **CLI arguments** (highest)
2. **Config file** (YAML, TOML, or JSON)
3. **Environment variables** (`MLXSMITH__SECTION__KEY`)
4. **Defaults**

```bash
# Show resolved configuration
mlxsmith config show

# Create a default config file
mlxsmith config init mlxsmith.yaml

# Validate a config file
mlxsmith config validate mlxsmith.yaml
```

See [Configuration](docs/cli/configuration.md) for the full reference.

## Documentation

| Section | Description |
|---------|-------------|
| [Getting Started](docs/getting-started.md) | Installation and first training run |
| [Concepts](docs/concepts.md) | Training modes explained |
| [CLI Reference](docs/cli/README.md) | All commands with examples |
| [Configuration](docs/cli/configuration.md) | Config system and options |
| [Verifiers](docs/VERIFIERS.md) | Verifier API and composition |
| [Environments](docs/ENVIRONMENTS.md) | Task environment plugins |
| [Project Format](docs/PROJECT_FORMAT.md) | Run artifacts and directory layout |
| [Compatibility](docs/COMPATIBILITY.md) | Tested versions and models |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and fixes |
| [FAQ](docs/FAQ.md) | Frequently asked questions |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and pull request guidelines.

## License

MLXSmith is released under the [MIT License](LICENSE).
