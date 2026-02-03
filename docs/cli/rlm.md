# RLM (Recursive Language Model)

MLXSmith implements two complementary RLM paradigms:

1. **RLM Training Loop** — Self-improving training cycle with task generation, verification, and GRPO training
2. **RLM Inference** — REPL-based inference following the Zhang et al. paradigm (arXiv:2512.24601)

---

## RLM Training Loop

The RLM training loop is a self-improving cycle that generates tasks, collects rollouts, verifies solutions, trains on the results, and gates adapter promotion based on benchmark scores.

### When to use

- You want automated, iterative model improvement over many cycles.
- You have a verifier that can score model outputs and a benchmark suite for evaluation.
- You want the model to generate its own training tasks and learn from them.

### Minimal example

```bash
mlxsmith rlm \
  --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --iterations 50
```

### How it works

Each iteration:

1. **Task generation** — the model generates coding/reasoning tasks
2. **Task mutation** — Evol-Instruct style variation for diversity
3. **Rollout collection** — N candidate solutions per task
4. **Verification** — grade each solution via the configured verifier
5. **Training** — GRPO policy gradient update on graded rollouts
6. **Evaluation** — benchmark against held-out task suite
7. **Gating** — accept or reject the adapter based on score improvement

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | From config | Model path or id |
| `--iterations` | From config | Number of RLM iterations (0 = infinite) |
| `--resume` | `false` | Resume from last completed iteration |
| `--orchestrated` | `false` | Use multi-process orchestrator mode |
| `--config`, `-c` | `mlxsmith.yaml` | Config file path |

---

## RLM Inference (REPL-based)

The canonical RLM inference paradigm allows the model to interact with a Python REPL during generation. The model can execute code, observe output, make recursive sub-calls, and signal completion.

### When to use

- You need to process very long contexts (100K+ tokens)
- The task benefits from decomposition and recursive processing
- You want the model to programmatically examine and manipulate data

### Available tools

The model has access to:

- `context` — The input text stored as a string variable
- `llm_query(prompt)` — Make recursive sub-calls to a language model
- `llm_batch(prompts)` — Make parallel sub-calls (list in, list out)
- `FINAL(answer)` — Signal completion with a direct answer
- `FINAL_VAR(varname)` — Signal completion, answer is in the named variable
- Standard Python execution with output capture

### Example inference

```bash
# Process a long document
mlxsmith rlm infer "Summarize this research paper: ..." \
  --model qwen3 \
  --sandbox local

# Process from file
mlxsmith rlm infer @input.txt \
  --model qwen3 \
  --out trajectory.json
```

### How it works

1. Model receives context and system prompt explaining available tools
2. Model generates code in ```repl blocks
3. Code executes in sandbox, output returned to model
4. Model continues reasoning based on output
5. Model calls `FINAL()` when answer is ready
6. Trajectory saved for training

### Collect trajectories for training

```bash
# Generate trajectories from prompts
mlxsmith rlm collect data/prompts.jsonl \
  --model qwen3 \
  --out data/rlm_trajectories \
  --sandbox docker

# Output includes:
# - data/rlm_trajectories/trajectory_0000.json
# - data/rlm_trajectories/trajectory_0001.json
# - data/rlm_trajectories/training_pairs.jsonl
```

The `training_pairs.jsonl` can be used directly for SFT training on RLM trajectories.

### Sandbox options

| Sandbox | Description |
|---------|-------------|
| `local` | Execute in the current Python process (fast, not isolated) |
| `docker` | Execute in a Docker container (recommended for untrusted code) |

### RLM Inference Config

| Config Key | Default | Description |
|------------|---------|-------------|
| `rlm.repl_max_turns` | `20` | Maximum model↔REPL turns |
| `rlm.repl_max_tokens_per_turn` | `1024` | Tokens per model response |
| `rlm.repl_temperature` | `0.7` | Generation temperature |
| `rlm.repl_sandbox` | `local` | Sandbox: `local` or `docker` |
| `rlm.repl_max_output_chars` | `4000` | Max REPL stdout/stderr per execution |
| `rlm.repl_max_exec_iterations` | `50` | Max REPL executions per run |
| `rlm.repl_timeout_per_exec_s` | `30.0` | REPL execution timeout (best-effort) |
| `rlm.repl_sub_call_max_tokens` | `512` | Tokens for `llm_query()` sub-calls |
| `rlm.repl_sub_call_temperature` | `0.3` | Temperature for sub-calls |
| `rlm.repl_system_prompt` | `null` | Override the default RLM system prompt |
| `rlm.docker_image` | `python:3.11-slim` | Docker image for `--sandbox docker` |
| `rlm.docker_memory_mb` | `512` | Memory limit for `--sandbox docker` |
| `rlm.docker_cpus` | `1.0` | CPU limit for `--sandbox docker` |
| `rlm.docker_pids` | `128` | PID limit for `--sandbox docker` |

---

## Subcommands

### Run training loop

```bash
mlxsmith rlm --iterations 50
```

### Run inference

```bash
mlxsmith rlm infer "Process this text..." --model qwen3
```

### Collect trajectories

```bash
mlxsmith rlm collect data/prompts.jsonl --out data/trajectories
```

### Check status

```bash
mlxsmith rlm status
```

### View history

```bash
mlxsmith rlm history --limit 20
```

---

## Multi-process mode

The `--orchestrated` flag runs the training loop with separate inference and trainer processes:

```bash
mlxsmith rlm --orchestrated --iterations 50
```

This uses the [orchestrator](../orchestrator.md) architecture with:
- Non-blocking inference via a separate server process
- Hot weight reloading without restart
- Asynchronous training with process isolation

---

## Output

### Training loop output

RLM training writes to `runs/rlm_0001/` with:

- `adapter/` — current adapter weights
- `metrics.jsonl` — per-iteration metrics
- `config.snapshot.yaml` — configuration at run start

Cross-iteration state:

- `runs/rlm_state.json` — gating state (best adapter, scores)
- `runs/rlm_history.jsonl` — benchmark history log

### Inference output

Trajectories are saved as JSON with:

- `context` — original input
- `turns` — list of model outputs and REPL results
- `final_answer` — the computed answer
- `success` — whether FINAL was called
- `sub_calls` — number of llm_query calls

---

## Gating strategies

| Strategy | Description |
|----------|-------------|
| `strict` | Only promote if score exceeds historical best |
| `threshold` | Promote if score exceeds a configurable threshold |
| `ema` | Promote if score exceeds an exponential moving average |

---

## References

- Zhang et al., "Recursive Language Models" (arXiv:2512.24601)
- [alexzhang13/rlm](https://github.com/alexzhang13/rlm) — Reference implementation
- [Prime Intellect RLMEnv](https://www.primeintellect.ai/blog/rlm) — Industry adoption
