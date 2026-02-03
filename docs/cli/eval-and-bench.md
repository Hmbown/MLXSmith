# Evaluation and Benchmarking

MLXSmith provides two measurement tools: `eval` for task-based evaluation suites and `bench` for throughput benchmarking.

## Eval

Run a YAML-defined evaluation suite that tests a model against tasks with verifier-based scoring and pass@k metrics.

### Minimal example

```bash
mlxsmith eval --suite eval/suites/coding.yaml --model runs/sft_0001/adapter
```

### Evaluation suite format

```yaml
name: coding-eval-sample
notes: |
  Minimal eval suite for smoke testing.
tasks:
  - id: add
    prompt: |
      Write a Python function `add(a, b)` that returns the sum.
    k: 2
    max_new_tokens: 128
    verifier: verifiers/regex.py
    verifier_kwargs:
      pattern: "def\\s+add\\("
```

Each task defines a prompt, a verifier, and the number of attempts (`k`) for pass@k scoring.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--suite` | `eval/suites/coding.yaml` | Path to evaluation suite YAML |
| `--model` | Required | Model or adapter path |

## Bench

Benchmark inference throughput, training step speed, or end-to-end performance.

### Minimal example

```bash
mlxsmith bench --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit --mode inference
```

### Modes

| Mode | Description |
|------|-------------|
| `inference` | Measure tokens/second during generation |
| `trainer` | Measure steps/second during training |
| `end_to_end` | Measure full pipeline throughput |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | From config | Model path or id |
| `--mode` | `inference` | Benchmark mode |
| `--prompt` | `"Hello"` | Prompt for inference mode |
| `--max-tokens` | `128` | Max tokens per generation |
| `--reps` | `3` | Number of repetitions |
| `--steps` | `5` | Steps per rep (trainer mode) |
| `--config`, `-c` | `mlxsmith.yaml` | Config file path |
| `--accel` | From config | Acceleration backend |

### Output

Reports are written to `bench/bench_<timestamp>.json` with per-rep tokens/sec or steps/sec statistics.

## Typical workflows

### Evaluate a trained adapter

```bash
mlxsmith eval --suite eval/suites/coding.yaml --model runs/sft_0001/adapter
```

### Compare base vs. fine-tuned throughput

```bash
mlxsmith bench --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit --mode inference
mlxsmith bench --model runs/sft_0001/adapter --mode inference
```

### Benchmark training throughput

```bash
mlxsmith bench --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit --mode trainer --steps 10
```
