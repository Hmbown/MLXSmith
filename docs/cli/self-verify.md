# Self-Verification Training

Self-verification training uses the model (or a separate verifier model) to assess its own outputs. The verification scores serve as reward signals for policy gradient updates. This is similar to online DPO but uses a verifier-style reward rather than pairwise preferences.

## When to use

- You want the model to learn from its own self-assessment.
- You have a verifier model or want the model to judge its own responses using ThinkPRM-style process grading.

## Minimal example

```bash
mlxsmith self-verify \
  --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --data data/prompts.jsonl
```

## Data format

JSONL with `prompt` fields:

```json
{"prompt": "Write a Python function that checks if a string is a palindrome."}
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | Required | Model to train |
| `--data` | Required | JSONL with prompts |
| `--verifier-model` | Same as model | Model used for verification scoring |
| `--verifier-backend` | `mlx-lm` | Backend for the verifier model |
| `--rubric` | None | Scoring rubric text or file path |
| `--max-new-tokens` | From config | Max generation length |
| `--temperature` | From config | Sampling temperature |
| `--config`, `-c` | `mlxsmith.yaml` | Config file path |
| `--accel` | From config | Acceleration backend |

## Output

Training writes to `runs/self_verify_0001/` with adapter weights, metrics, and config snapshot.

## Typical workflows

### Self-verify with a dedicated verifier model

```bash
mlxsmith self-verify \
  --model runs/sft_0001/adapter \
  --data data/prompts.jsonl \
  --verifier-model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit
```

### Self-verify with a rubric

```bash
mlxsmith self-verify \
  --model runs/sft_0001/adapter \
  --data data/prompts.jsonl \
  --rubric verifiers/rubrics/coding.txt
```
