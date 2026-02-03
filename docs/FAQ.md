# FAQ

## Where is the configuration stored?

The default config file is `mlxsmith.yaml` in your project root. You can override values with CLI flags or environment variables like `MLXSMITH__MODEL__ID`.

## How do I use an adapter from a previous run?

Most commands accept `--model` and can point directly at an adapter directory:

```bash
mlxsmith serve --model runs/sft_0001/adapter
mlxsmith pref --model runs/sft_0001/adapter --data data/prefs
```

## How do I resume RLM?

Use the `--resume` flag to continue from the latest RLM state:

```bash
mlxsmith rlm --resume
```

## Can I run without MLX or Metal?

Data tools and configuration commands work on any platform. Training and inference require MLX; if it is unavailable, MLXSmith will report the backend as unavailable and skip training.

## How do I set a default judge model?

Set the environment variable once:

```bash
export MLXSMITH_JUDGE_MODEL=cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit
```

## Where are datasets expected?

Training commands expect a directory with `train.jsonl` (and optional `valid.jsonl`/`test.jsonl`). Use `mlxsmith data split` to create the directory from a single JSONL file.
