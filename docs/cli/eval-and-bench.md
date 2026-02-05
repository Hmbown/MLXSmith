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

      Return only Python code.
    k: 2
    max_new_tokens: 128
    tests: |
      from main import add

      def test_add():
          assert add(1, 2) == 3
          assert add(-1, 5) == 4
```

Each task defines a prompt, a pass@k attempt count (`k`), and either:

- `tests` — embedded pytest tests (recommended)
- `verifier` — a custom verifier module (see `docs/VERIFIERS.md`)

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
| `--mhc` | From config | Enable experimental mHC adapters (not a speedup) |
| `--mhc-n` | From config | mHC stream expansion rate (n) |
| `--mhc-tmax` | From config | mHC Sinkhorn iterations (tmax) |

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
