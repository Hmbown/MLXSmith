# Judge Training

The `judge` command fine-tunes a model to act as a scoring model. It runs standard SFT on judge-format data — prompt-response pairs where the responses contain scoring judgments. The trained judge can then be used in online DPO, self-verification, synthetic data filtering, and evaluation.

## When to use

- You want a dedicated judge model for automated evaluation or reward scoring.
- You have judge-format training data (prompts requesting evaluation, responses containing scores and reasoning).

## Minimal example

```bash
mlxsmith judge \
  --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --data data/judge
```

## Data format

Standard SFT format — JSONL with `prompt` and `response` fields. The prompts should request evaluation of a completion, and the responses should contain structured judgments:

```json
{"prompt": "Rate the following code solution...\n[solution here]\nScore from 0.0 to 1.0.", "response": "{\"score\": 0.8, \"reasoning\": \"Correct but could be more efficient.\"}"}
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | Required | Base model for judge SFT |
| `--data` | `data/judge` | Judge training data directory |
| `--config`, `-c` | `mlxsmith.yaml` | Config file path |
| `--accel` | From config | Acceleration backend |
| `--lr` | From config | Learning rate |
| `--iters` | From config | Training iterations |

## Output

Training writes to `runs/judge_0001/` with standard SFT outputs (adapter, metrics, config snapshot). Use the adapter path printed by `mlxsmith judge`.

## Typical workflows

### Train a judge then use it for online DPO

```bash
mlxsmith judge --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit --data data/judge
mlxsmith online-dpo \
  --model runs/sft_0001/adapter \
  --data data/prompts.jsonl \
  --judge-model runs/judge_0001/adapter
```

### Train a judge for synthetic data filtering

```bash
mlxsmith judge --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit --data data/judge
mlxsmith synthetic sft \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --prompts data/prompts.jsonl \
  --judge-model runs/judge_0001/adapter \
  --min-score 0.7
```
