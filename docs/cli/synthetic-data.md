# Synthetic Data Generation

The `synthetic` command group generates training data using a model. You can generate prompts from scratch, evolve existing prompts for diversity, and produce SFT or DPO pairs with optional judge filtering.

## When to use

- You need more training data than you have.
- You want to diversify existing prompts (Evol-Instruct style).
- You want to generate preference pairs without manual labeling.

## Commands

### Generate prompts

Generate task prompts from a model, optionally seeded with existing examples:

```bash
mlxsmith synthetic prompts \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --num 100 \
  --out data/prompts.jsonl
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | Required | Model id or path |
| `--num` | `50` | Number of prompts to generate |
| `--out` | `data/synthetic_prompts.jsonl` | Output path |
| `--seed-prompts` | None | JSONL with seed prompts for few-shot |
| `--system-prompt` | None | Override system prompt |
| `--max-new-tokens` | `256` | Max generation length |
| `--temperature` | `0.9` | Sampling temperature |
| `--seed` | `42` | Random seed |
| `--config`, `-c` | `mlxsmith.yaml` | Config file path |

### Evolve prompts

Apply Evol-Instruct style transformations to make prompts harder or more diverse:

```bash
mlxsmith synthetic evolve \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --seeds data/prompts.jsonl \
  --num 100 \
  --mode mix
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | Required | Model id or path |
| `--seeds` | Required | JSONL with seed prompts |
| `--num` | `50` | Number of evolved prompts |
| `--out` | `data/synthetic_evolved.jsonl` | Output path |
| `--mode` | `mix` | Evolution mode (see below) |
| `--system-prompt` | None | Override system prompt |
| `--max-new-tokens` | `256` | Max generation length |
| `--temperature` | `0.9` | Sampling temperature |
| `--seed` | `42` | Random seed |
| `--config`, `-c` | `mlxsmith.yaml` | Config file path |

Evolution modes:

| Mode | Description |
|------|-------------|
| `mix` | Randomly sample from all modes |
| `deepen` | Add multi-step reasoning |
| `broaden` | Expand scope and coverage |
| `complexify` | Increase difficulty |
| `constraints` | Add evaluation criteria |
| `multi_turn` | Convert to multi-turn dialogue |

### Generate SFT pairs

Generate prompt-response pairs with optional judge-based filtering:

```bash
mlxsmith synthetic sft \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --prompts data/prompts.jsonl \
  --candidates 4 \
  --judge-model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --min-score 0.7
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | Required | Model id or path |
| `--prompts` | Required | Input JSONL with prompts |
| `--out` | `data/synthetic_sft.jsonl` | Output path |
| `--candidates` | `1` | Candidates per prompt (for rejection sampling) |
| `--judge-model` | Required when `--candidates > 1` or filtering is enabled (or set `MLXSMITH_JUDGE_MODEL`) | Judge model for filtering |
| `--judge-backend` | `mlx-lm` | Judge backend |
| `--rubric` | None | Scoring rubric text or file path |
| `--min-score` | None | Filter responses below this score |
| `--max-prompts` | None | Limit prompts processed |
| `--max-new-tokens` | `512` | Max generation length |
| `--temperature` | `0.7` | Sampling temperature |
| `--seed` | `42` | Random seed |
| `--config`, `-c` | `mlxsmith.yaml` | Config file path |

### Generate DPO pairs

Generate preference pairs by scoring multiple candidates per prompt:

```bash
mlxsmith synthetic dpo \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --prompts data/prompts.jsonl \
  --candidates 4 \
  --judge-model mlx-community/Qwen3-4B-Instruct-2507-4bit
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | Required | Model id or path |
| `--prompts` | Required | Input JSONL with prompts |
| `--out` | `data/synthetic_dpo.jsonl` | Output path |
| `--candidates` | `4` | Candidates per prompt |
| `--judge-model` | Required (or set `MLXSMITH_JUDGE_MODEL`) | Judge model for ranking |
| `--judge-backend` | `mlx-lm` | Judge backend |
| `--rubric` | None | Scoring rubric text or file path |
| `--min-margin` | None | Skip pairs below this score margin |
| `--max-prompts` | None | Limit prompts processed |
| `--max-new-tokens` | `512` | Max generation length |
| `--temperature` | `0.8` | Sampling temperature |
| `--seed` | `42` | Random seed |
| `--config`, `-c` | `mlxsmith.yaml` | Config file path |

## Typical workflows

### Full synthetic pipeline

```bash
# 1. Generate prompts
mlxsmith synthetic prompts --model mlx-community/Qwen3-4B-Instruct-2507-4bit --num 500

# 2. Evolve for diversity
mlxsmith synthetic evolve \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --seeds data/synthetic_prompts.jsonl --num 500

# 3. Generate SFT pairs with judge filtering
mlxsmith synthetic sft \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --prompts data/synthetic_evolved.jsonl \
  --candidates 4 \
  --judge-model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --min-score 0.7

# 4. Train on the generated data
mlxsmith data split --in data/synthetic_sft.jsonl --out-dir data/synthetic_sft
mlxsmith sft --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit --data data/synthetic_sft
```

### Generate preference data for DPO

```bash
mlxsmith synthetic dpo \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --prompts data/prompts.jsonl \
  --candidates 4 \
  --judge-model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --min-margin 0.3

mlxsmith data split --in data/synthetic_dpo.jsonl --out-dir data/synthetic_prefs
mlxsmith pref --model runs/sft_0001/adapter --data data/synthetic_prefs
```
