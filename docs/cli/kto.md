# KTO (Kahneman-Tversky Optimization)

KTO trains a model from binary feedback (good/bad labels) rather than paired preferences. It is grounded in prospect theory — the model is trained with asymmetric loss weighting that reflects loss aversion.

## When to use

- You have individual response quality labels (thumbs-up/thumbs-down) rather than side-by-side comparisons.
- Binary feedback is often easier to collect than preference pairs.

## Minimal example

```bash
mlxsmith kto \
  --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --data data/kto.jsonl
```

## Data format

JSONL with `prompt`, `response`, and `label` fields. The label is a boolean or `0`/`1`:

```json
{"prompt": "What is 2+2?", "response": "4", "label": true}
{"prompt": "What is 2+2?", "response": "Fish", "label": false}
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | Required | Model path or id |
| `--data` | Required | Path to JSONL file |
| `--config`, `-c` | `mlxsmith.yaml` | Config file path |
| `--accel` | From config | Acceleration backend |
| `--reference-model` | None | Reference model for KL penalty |
| `--beta` | `0.1` | Loss scale / inverse temperature (KTO) |
| `--loss-aversion` | `1.0` | Loss aversion coefficient (>1 penalizes bad outputs more) |
| `--gain-power` | `1.0` | Power function exponent for positive rewards |
| `--loss-power` | `1.0` | Power function exponent for negative rewards |
| `--reference-point` | `0.0` | Reference point for prospect theory calculation |

## Output

Training writes to `runs/kto_0001/` with adapter weights, metrics, and config snapshot.

## Typical workflows

### KTO after SFT

```bash
mlxsmith sft --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit --data data/sft
mlxsmith kto --model runs/sft_0001/adapter --data data/kto.jsonl
```

### KTO with loss aversion tuning

Increase `--loss-aversion` to penalize bad outputs more heavily:

```bash
mlxsmith kto \
  --model runs/sft_0001/adapter \
  --data data/kto.jsonl \
  --loss-aversion 2.0
```
