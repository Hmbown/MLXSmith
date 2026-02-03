# Online DPO

Online DPO generates preference data on-the-fly. At each training step, the model generates multiple candidate responses to a prompt. An LLM judge scores them, and the best and worst candidates become the chosen/rejected pair for a DPO update.

## When to use

- You want preference-based training but do not have pre-collected preference data.
- You have (or can specify) a judge model to score candidate responses.

## Minimal example

```bash
mlxsmith online-dpo \
  --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --data data/prompts.jsonl \
  --judge-model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit
```

## Data format

JSONL with `prompt` fields:

```json
{"prompt": "Explain the difference between a list and a tuple in Python."}
{"prompt": "Write a function that reverses a linked list."}
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | Required | Model to train |
| `--data` | Required | JSONL with prompts |
| `--judge-model` | Required (or set `MLXSMITH_JUDGE_MODEL`) | Model used for scoring candidates |
| `--judge-backend` | `mlx-lm` | Backend for the judge model |
| `--rubric` | None | Scoring rubric text or file path |
| `--group-size` | From config | Number of candidates per prompt |
| `--max-new-tokens` | From config | Max generation length |
| `--temperature` | From config | Sampling temperature |
| `--config`, `-c` | `mlxsmith.yaml` | Config file path |
| `--accel` | From config | Acceleration backend |

## Output

Training writes to `runs/online_dpo_0001/` with adapter weights, metrics (including best/worst reward per step), and config snapshot.

## Typical workflows

### Online DPO with a dedicated judge

```bash
mlxsmith online-dpo \
  --model runs/sft_0001/adapter \
  --data data/prompts.jsonl \
  --judge-model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit
```

### Online DPO with a custom rubric

```bash
mlxsmith online-dpo \
  --model runs/sft_0001/adapter \
  --data data/prompts.jsonl \
  --judge-model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --rubric verifiers/rubrics/coding.txt
```
