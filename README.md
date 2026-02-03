# MLXSmith

[![PyPI](https://img.shields.io/pypi/v/mlxsmith)](https://pypi.org/project/mlxsmith/)
[![CI](https://github.com/Hmbown/MLXSmith/actions/workflows/ci.yml/badge.svg)](https://github.com/Hmbown/MLXSmith/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/Hmbown/MLXSmith)](LICENSE)

Fine-tune language models on Apple Silicon. SFT, preference optimization, reinforcement learning, distillation, and serving — all native to MLX.

**Status:** Alpha (v0.1.6) · Validated on Qwen3-4B

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
- **Recursive training** — Self-improving RLM loop with task generation and gating
- **Serving** — OpenAI-compatible API with streaming
- **Environment plugins** — Reusable task and verifier packages for RL training

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
| [RLM](docs/cli/rlm.md) | `mlxsmith rlm` | Recursive self-improving training loop |

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
