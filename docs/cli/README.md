# CLI Reference

All commands are accessed through the `mlxsmith` binary. Run `mlxsmith --help` for a full listing.

## Project Setup

| Command | Guide | Description |
|---------|-------|-------------|
| `mlxsmith init <path>` | [Project Setup](project-setup.md) | Create a project with default directory layout |
| `mlxsmith doctor` | [Project Setup](project-setup.md) | Check Python, MLX, and Metal availability |

## Training

| Command | Guide | Description |
|---------|-------|-------------|
| `mlxsmith sft` | [SFT](sft.md) | Supervised fine-tuning with LoRA/QLoRA |
| `mlxsmith pref` | [Preference](preference-training.md) | DPO, ORPO, and other preference algorithms |
| `mlxsmith kto` | [KTO](kto.md) | Binary feedback training |
| `mlxsmith rft` | [GRPO](reinforcement-training.md) | Reinforcement fine-tuning with verifier rewards |
| `mlxsmith online-dpo` | [Online DPO](online-dpo.md) | Online preference with LLM judge |
| `mlxsmith self-verify` | [Self-Verify](self-verify.md) | Self-verification training |
| `mlxsmith distill` | [Distillation](distillation.md) | Teacher-to-student knowledge transfer |
| `mlxsmith judge` | [Judge](judge.md) | Train a scoring model |
| `mlxsmith pipeline` | [Pipeline](sft.md#pipeline) | Run SFT, Pref, RFT, and RLM in sequence |

## Data

| Command | Guide | Description |
|---------|-------|-------------|
| `mlxsmith data pull` | [Data](data.md) | Download a HF dataset or built-in preset |
| `mlxsmith data import` | [Data](data.md) | Convert ShareGPT format to JSONL |
| `mlxsmith data split` | [Data](data.md) | Split into train/valid/test |
| `mlxsmith data stats` | [Data](data.md) | Analyze dataset statistics |
| `mlxsmith data validate` | [Data](data.md) | Validate dataset structure |
| `mlxsmith data presets` | [Data](data.md) | List built-in dataset presets |

## Synthetic Data

| Command | Guide | Description |
|---------|-------|-------------|
| `mlxsmith synthetic prompts` | [Synthetic](synthetic-data.md) | Generate task prompts |
| `mlxsmith synthetic evolve` | [Synthetic](synthetic-data.md) | Evolve prompts for diversity |
| `mlxsmith synthetic sft` | [Synthetic](synthetic-data.md) | Generate SFT pairs with optional judge filtering |
| `mlxsmith synthetic dpo` | [Synthetic](synthetic-data.md) | Generate DPO preference pairs |

## Model Management

| Command | Guide | Description |
|---------|-------|-------------|
| `mlxsmith pull <model>` | [Model Management](model-management.md) | Download and convert a HF model to MLX format |
| `mlxsmith quantize <path>` | [Model Management](model-management.md) | Create a quantization stub |
| `mlxsmith adapters merge` | [Model Management](model-management.md) | Merge multiple LoRA adapters |

## Serving

| Command | Guide | Description |
|---------|-------|-------------|
| `mlxsmith serve` | [Serving](serving.md) | Start OpenAI-compatible API server |

## Evaluation and Benchmarking

| Command | Guide | Description |
|---------|-------|-------------|
| `mlxsmith eval` | [Eval](eval-and-bench.md) | Run evaluation suites with pass@k scoring |
| `mlxsmith bench` | [Bench](eval-and-bench.md) | Benchmark inference and training throughput |
| `mlxsmith losses` | [Losses](losses.md) | List all registered loss functions |

## RLM (Recursive Language Model)

| Command | Guide | Description |
|---------|-------|-------------|
| `mlxsmith rlm` | [RLM](rlm.md) | Run the RLM self-improving loop |
| `mlxsmith rlm status` | [RLM](rlm.md) | Show current iteration state |
| `mlxsmith rlm history` | [RLM](rlm.md) | View benchmark history |

## Configuration

| Command | Guide | Description |
|---------|-------|-------------|
| `mlxsmith config show` | [Config](configuration.md) | Display merged configuration |
| `mlxsmith config init` | [Config](configuration.md) | Create a default config file |
| `mlxsmith config validate` | [Config](configuration.md) | Validate config structure |
| `mlxsmith config env` | [Config](configuration.md) | Show environment variable mapping |

## Authentication

| Command | Guide | Description |
|---------|-------|-------------|
| `mlxsmith auth login` | [Auth](auth.md) | Save a Hugging Face token |
| `mlxsmith auth status` | [Auth](auth.md) | Check authentication status |
| `mlxsmith auth logout` | [Auth](auth.md) | Clear saved token |

## Environment Plugins

| Command | Guide | Description |
|---------|-------|-------------|
| `mlxsmith env init <name>` | [Environments](environments.md) | Scaffold a new environment |
| `mlxsmith env list` | [Environments](environments.md) | List registry entries |
| `mlxsmith env info <name>` | [Environments](environments.md) | Show environment metadata |
| `mlxsmith env install <source>` | [Environments](environments.md) | Install from directory, package, or registry |
| `mlxsmith env package <name>` | [Environments](environments.md) | Create distributable tarball |
| `mlxsmith env publish <package>` | [Environments](environments.md) | Publish to local registry |
| `mlxsmith env pull <name>` | [Environments](environments.md) | Download from registry |
| `mlxsmith env run <env>` | [Environments](environments.md) | Execute RFT with an environment |
| `mlxsmith env registry` | [Environments](environments.md) | Show registry index |

## Acceleration

| Command | Guide | Description |
|---------|-------|-------------|
| `mlxsmith accel status` | [Acceleration](accel.md) | Show acceleration backend status |
