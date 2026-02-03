# RLM (Recursive Language Model)

The RLM loop is a self-improving training cycle that generates tasks, collects rollouts, verifies solutions, trains on the results, and gates adapter promotion based on benchmark scores. It runs iteratively, with each cycle potentially improving the model.

## When to use

- You want automated, iterative model improvement over many cycles.
- You have a verifier that can score model outputs and a benchmark suite for evaluation.
- You want the model to generate its own training tasks and learn from them.

## Minimal example

```bash
mlxsmith rlm \
  --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --iterations 50
```

## How it works

Each iteration:

1. **Task generation** — the model generates coding/reasoning tasks (or loads from an environment)
2. **Task mutation** — Evol-Instruct style variation for diversity
3. **Rollout collection** — N candidate solutions per task
4. **Verification** — grade each solution via the configured verifier
5. **Training** — GRPO policy gradient update on graded rollouts
6. **Evaluation** — benchmark against held-out task suite
7. **Gating** — accept or reject the adapter based on score improvement

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | From config | Model path or id |
| `--iterations` | From config | Number of RLM iterations (0 = infinite) |
| `--resume` | `false` | Resume from last completed iteration |
| `--orchestrated` | `false` | Use multi-process orchestrator mode |
| `--config`, `-c` | `mlxsmith.yaml` | Config file path |

Key config options under the `rlm` section:

| Config Key | Default | Description |
|------------|---------|-------------|
| `rlm.iterations` | `50` | Number of iterations |
| `rlm.rollouts_per_task` | `8` | Solutions per task |
| `rlm.corpus_max` | `8000` | Max rolling corpus size |
| `rlm.mix_old_ratio` | `0.4` | Fraction of old data mixed in |
| `rlm.hard_ratio` | `0.6` | Hard sample weighting |
| `rlm.gating` | `strict` | Gating strategy (`strict`, `threshold`, `ema`) |

## Subcommands

### Check status

```bash
mlxsmith rlm status
```

Shows the current iteration, active adapter, best adapter, best score, and EMA score.

### View history

```bash
mlxsmith rlm history --limit 20
```

Shows benchmark results across iterations (JSONL log).

## Multi-process mode

The `--orchestrated` flag runs the RLM loop with separate inference and trainer processes for better throughput:

```bash
mlxsmith rlm --orchestrated --iterations 50
```

This uses the [orchestrator](../orchestrator.md) architecture with:
- Non-blocking inference via a separate server process
- Hot weight reloading without restart
- Asynchronous training with process isolation
- Weight pointer IPC for staleness control

## Output

RLM writes to `runs/rlm_0001/` with:

- `adapter/` — current adapter weights
- `metrics.jsonl` — per-iteration metrics
- `config.snapshot.yaml` — configuration at run start

Cross-iteration state is stored in:

- `runs/rlm_state.json` — gating state (best adapter, scores)
- `runs/rlm_history.jsonl` — benchmark history log
- `runs/rlm_weights/` — weight pointers for inference/trainer coordination

## Typical workflows

### Basic RLM loop

```bash
mlxsmith init myproj && cd myproj
mlxsmith pull mlx-community/Qwen3-4B-Instruct-2507-4bit
mlxsmith rlm --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit --iterations 50
```

### Full pipeline into RLM

```bash
mlxsmith pipeline \
  --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --data-sft data/sft \
  --data-pref data/prefs \
  --env envs/coding.yaml \
  --verifier verifiers/regex.py \
  --orchestrated
```

### Resume a stopped run

```bash
mlxsmith rlm --resume
```

## Gating strategies

| Strategy | Description |
|----------|-------------|
| `strict` | Only promote if score exceeds historical best |
| `threshold` | Promote if score exceeds a configurable threshold |
| `ema` | Promote if score exceeds an exponential moving average of past scores |
