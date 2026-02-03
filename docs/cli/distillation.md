# Knowledge Distillation

Distillation transfers knowledge from a larger teacher model to a smaller student model. MLXSmith supports two modes: offline (teacher generates, student learns via SFT) and online preference distillation (OPD).

## When to use

- You have a strong large model and want to create a smaller, faster model that approaches its quality.
- You want to compress model capabilities into a more deployable form factor.

## Minimal example

```bash
mlxsmith distill \
  --teacher cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --student cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --data data/prompts.jsonl \
  --mode offline
```

In practice, set `--student` to the smaller model you want to distill into.

## Data format

JSONL with `prompt` fields:

```json
{"prompt": "Explain the concept of gradient descent."}
{"prompt": "Write a sorting algorithm in Python."}
```

## Modes

### Offline distillation

The teacher generates responses for each prompt, and the student is trained on them via SFT:

```bash
mlxsmith distill \
  --teacher cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --student cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --data data/prompts.jsonl \
  --mode offline
```

### Online Preference Distillation (OPD)

The student generates candidates, the teacher scores them, and an importance-sampled loss updates the student:

```bash
mlxsmith distill \
  --teacher cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --student cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --data data/prompts.jsonl \
  --mode opd
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--teacher` | Required | Teacher model path or id |
| `--student` | Required | Student model path or id |
| `--data` | Required | JSONL with prompts |
| `--mode` | `offline` | Distillation mode (`offline` or `opd`) |
| `--max-new-tokens` | `256` | Max generation length |
| `--temperature` | `0.7` | Sampling temperature |
| `--config`, `-c` | `mlxsmith.yaml` | Config file path |
| `--accel` | From config | Acceleration backend |

## Output

Training writes to `runs/distill_0001/`:

- **Offline mode:** Generated dataset under `runs/distill_0001/artifacts/distill_data/train.jsonl`. The adapter is produced by a child SFT run (see the `child_run` field in `runs/distill_0001/metrics.jsonl` or the printed run directory).
- **OPD mode:** Adapter weights + metrics with importance-sampled loss tracking in the distill run directory
- `config.snapshot.yaml` — exact configuration used
