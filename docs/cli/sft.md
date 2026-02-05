# SFT (Supervised Fine-Tuning)

Supervised fine-tuning trains a model to follow instructions by learning from prompt-response pairs. It uses LoRA or QLoRA adapters, leaving the base model weights unchanged.

## When to use

- You have a dataset of instruction-response pairs and want the model to replicate that behavior.
- This is typically the first training stage before preference tuning or reinforcement learning.

## Minimal example

```bash
mlxsmith sft \
  --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --data data/sft
```

## Data format

JSONL files in the data directory. Each line must have `prompt` and `response` fields (or `instruction` and `output`):

```json
{"prompt": "Explain what a closure is in Python.", "response": "A closure is a function that..."}
```

The data directory should contain `train.jsonl` and optionally `valid.jsonl` and `test.jsonl`. Use `mlxsmith data split` to create these from a single file.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | From config | Model path or HF id |
| `--data` | `data/sft` | Directory containing JSONL files |
| `--config`, `-c` | `mlxsmith.yaml` | Config file path |
| `--lr` | From config | Learning rate |
| `--iters` | From config | Number of training iterations |
| `--batch-size` | From config | Batch size |
| `--accel` | From config | Acceleration backend |

Additional settings are controlled via the config file under the `train` and `lora` sections. Key config options:

| Config Key | Default | Description |
|------------|---------|-------------|
| `train.lr` | `2e-4` | Learning rate |
| `train.iters` | `1000` | Training iterations |
| `train.batch_size` | `1` | Batch size |
| `train.weight_decay` | `0.0` | Weight decay |
| `train.optimizer` | `adamw` | Optimizer (`adamw`, `adam`, `qhadam`, `muon`) |
| `train.grad_accum` | `8` | Gradient accumulation steps |
| `train.max_grad_norm` | `1.0` | Gradient clipping |
| `train.save_every` | `100` | Save checkpoint interval |
| `train.eval_every` | `100` | Evaluation interval |
| `lora.r` | `16` | LoRA rank |
| `lora.alpha` | `32` | LoRA alpha |
| `lora.dropout` | `0.05` | LoRA dropout |

## Output

Training writes to a run directory (e.g., `runs/sft_0001/`):

- `adapter/` — LoRA adapter weights (MLX-LM compatible)
- `metrics.jsonl` — loss and throughput per step
- `config.snapshot.yaml` — exact configuration used

If you stop training early (Ctrl+C / SIGINT), MLXSmith saves the latest adapter before exiting.

## Typical workflows

### SFT on a preset dataset

```bash
mlxsmith data pull --preset alpaca
mlxsmith sft --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit --data data/sft
```

### SFT with custom hyperparameters

```bash
mlxsmith sft \
  --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --data data/sft \
  --lr 5e-5 \
  --iters 2000 \
  --batch-size 2
```

### SFT then serve

```bash
mlxsmith sft --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit --data data/sft
mlxsmith serve --model runs/sft_0001/adapter --port 8080
```

## Pipeline

The `pipeline` command chains SFT, preference training, RFT, and RLM in sequence:

```bash
mlxsmith pipeline \
  --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --data-sft data/sft \
  --data-pref data/prefs \
  --env envs/coding.yaml \
  --verifier verifiers/regex.py
```

Each stage passes its output adapter to the next.
