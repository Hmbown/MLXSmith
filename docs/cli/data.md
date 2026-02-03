# Data Tools

The `data` command group provides utilities for importing, splitting, validating, and downloading training datasets.

## Commands

### Pull a dataset

Download a Hugging Face dataset or a built-in preset:

```bash
# Pull a preset
mlxsmith data pull --preset alpaca

# Pull a specific HF dataset
mlxsmith data pull --dataset tatsu-lab/alpaca --split train --out-dir data/sft
```

| Option | Default | Description |
|--------|---------|-------------|
| `--preset` | None | Built-in preset name |
| `--dataset` | None | HF dataset name |
| `--split` | `train` | Dataset split |
| `--out-dir` | `data/sft` | Output directory |
| `--kind` | `sft` | Dataset kind (`sft` or `prefs`) |
| `--limit` | None | Limit number of rows |
| `--prompt-field` | Auto | Field name for prompts |
| `--response-field` | Auto | Field name for responses |
| `--chosen-field` | Auto | Field name for chosen (prefs) |
| `--rejected-field` | Auto | Field name for rejected (prefs) |
| `--config` | None | HF dataset config |
| `--revision` | None | HF dataset revision |

### List presets

```bash
mlxsmith data presets
```

Built-in presets:

| Preset | Dataset | Kind |
|--------|---------|------|
| `alpaca` | Stanford Alpaca (52K instructions) | SFT |
| `hh-rlhf` | Anthropic HH-RLHF | Prefs |
| `ultrachat-200k` | UltraChat 200K | SFT |
| `ultrafeedback-binarized-prefs` | UltraFeedback preference pairs | Prefs |
| `ultrafeedback-binarized-sft` | UltraFeedback SFT data | SFT |

### Import from ShareGPT format

Convert a ShareGPT-format JSON file to JSONL:

```bash
mlxsmith data import --in raw.json --format sharegpt --out data/sft/train.jsonl
```

| Option | Default | Description |
|--------|---------|-------------|
| `--in` | Required | Input file path |
| `--format` | `sharegpt` | Input format |
| `--out` | Required | Output JSONL path |

### Split a dataset

Split a JSONL file into train, validation, and test sets:

```bash
mlxsmith data split --in data/sft/train.jsonl --valid 0.05 --test 0.05
```

| Option | Default | Description |
|--------|---------|-------------|
| `--in` | Required | Input JSONL file |
| `--out-dir` | `data/sft` | Output directory |
| `--valid` | `0.02` | Validation fraction |
| `--test` | `0.02` | Test fraction |
| `--seed` | `1337` | Random seed |

### Analyze dataset statistics

```bash
mlxsmith data stats --in data/sft/train.jsonl
```

Shows row count, field coverage, average character lengths, and error counts.

| Option | Default | Description |
|--------|---------|-------------|
| `--in` | Required | Input JSONL file |
| `--kind` | Auto | Dataset kind (`sft` or `prefs`) |
| `--limit` | None | Limit rows analyzed |

### Validate dataset structure

```bash
mlxsmith data validate --in data/sft/train.jsonl
```

Checks for malformed JSON, missing fields, and structural issues. Exits with code 1 in strict mode if issues are found.

| Option | Default | Description |
|--------|---------|-------------|
| `--in` | Required | Input JSONL file |
| `--kind` | Auto | Dataset kind (`sft` or `prefs`) |
| `--strict` / `--no-strict` | `--strict` | Exit with error on issues |
| `--limit` | None | Limit rows checked |

## Typical workflows

### Prepare SFT data from a preset

```bash
mlxsmith data pull --preset alpaca
mlxsmith data stats --in data/sft/train.jsonl
mlxsmith data validate --in data/sft/train.jsonl
```

### Import and split custom data

```bash
mlxsmith data import --in raw_conversations.json --format sharegpt --out data/sft/all.jsonl
mlxsmith data split --in data/sft/all.jsonl --valid 0.05 --test 0.05
mlxsmith data validate --in data/sft/train.jsonl
```

### Pull preference data

```bash
mlxsmith data pull --preset ultrafeedback-binarized-prefs --out-dir data/prefs
mlxsmith data stats --in data/prefs/train.jsonl --kind prefs
```
