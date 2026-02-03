# Preference Training (DPO/ORPO)

Preference training aligns a model using comparison pairs. Each example contains a prompt, a preferred (chosen) response, and a rejected response. The model learns to prefer the chosen output.

## When to use

- You have preference data (chosen vs. rejected pairs) and want to improve model alignment.
- Typically run after SFT to refine model behavior.

## Minimal example

```bash
mlxsmith pref \
  --model runs/sft_0001/adapter \
  --data data/prefs \
  --loss-type dpo
```

## Data format

JSONL files with `prompt`, `chosen`, and `rejected` fields:

```json
{"prompt": "Explain recursion.", "chosen": "Recursion is when a function calls itself...", "rejected": "Recursion is a loop..."}
```

The data directory should contain `train.jsonl` and optionally `valid.jsonl`.

## Algorithms

| Algorithm | Flag | Reference Model | Description |
|-----------|------|----------------|-------------|
| DPO | `--loss-type dpo` | Optional | Standard offline preference optimization |
| ORPO | `--loss-type orpo` | Optional | Combines SFT loss with preference signal |
| IPO | `--loss-type ipo` | Optional | Identity-based preference optimization |
| CPO | `--loss-type cpo` | No | Contrastive preference — no reference model needed |
| SimPO | `--loss-type simpo` | No | Length-normalized, reference-free |
| Hinge | `--loss-type hinge` | Optional | Margin-based preference loss |
| TDPO | `--loss-type tdpo` | Optional | Token-level DPO (mean logprob) |

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | Required | Base model or adapter path (e.g., `runs/sft_0001/adapter`) |
| `--data` | `data/prefs` | Directory with preference JSONL files |
| `--loss-type` | From config | Preference algorithm (see table above) |
| `--algo` | From config | Legacy alias for algorithm selection |
| `--config`, `-c` | `mlxsmith.yaml` | Config file path |
| `--accel` | From config | Acceleration backend |

Key config options under the `pref` section:

| Config Key | Default | Description |
|------------|---------|-------------|
| `pref.loss_type` | `dpo` | Algorithm |
| `pref.beta` | `0.1` | Preference loss scale / inverse temperature |
| `pref.kl_coeff` | `0.0` | Additional KL coefficient |
| `pref.delta` | `0.0` | Margin for hinge/SimPO |

## Output

Training writes to `runs/pref_0001/` with the same structure as SFT (adapter, metrics, config snapshot).

## Typical workflows

### SFT then DPO

```bash
mlxsmith sft --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit --data data/sft
mlxsmith pref --model runs/sft_0001/adapter --data data/prefs --loss-type dpo
```

### ORPO (combined SFT + preference)

ORPO merges the SFT and preference objectives into a single loss, so it can be run without a prior SFT stage:

```bash
mlxsmith pref \
  --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --data data/prefs \
  --loss-type orpo
```

### Reference-free preference (SimPO or CPO)

SimPO and CPO do not require a reference model, reducing memory usage:

```bash
mlxsmith pref --model runs/sft_0001/adapter --data data/prefs --loss-type simpo
```
