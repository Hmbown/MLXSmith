# RLM Training with MLXSmith

Date: 2026-02-01

## Goal

Define the full set of capabilities needed for MLXSmith to support end-to-end
**Recursive Language Model (RLM)** training on Apple Silicon. An RLM loop
generates tasks, solves them, verifies solutions, and trains on the signal —
iteratively improving the model's reasoning ability.

This document lists what MLXSmith already has, what it needs, and where the gaps
are relative to a complete RLM training pipeline.

Update (2026-02-01): core RLM loop, task generation/mutation, gating, corpus,
history logging, and a basic monitor UI are now implemented. Remaining gaps are
called out in the partial status table below.

## What RLM training requires

An RLM training loop has five stages. Each stage maps to specific MLXSmith
capabilities.

```
 1. Task Generation     Model produces coding/reasoning tasks
 2. Rollout Collection  Model generates N candidate solutions per task
 3. Verification        Grade each solution (pass/fail, partial reward)
 4. RL Update           Policy gradient on graded rollouts (not just SFT)
 5. Evaluation          Benchmark against held-out tasks, gate promotion
```

## Current status (what works)

### Fully implemented

| Capability | Module | Notes |
|---|---|---|
| SFT LoRA/QLoRA training | `train/sft.py` | Standard supervised fine-tuning on (prompt, response) pairs |
| DPO/ORPO preference tuning | `train/pref.py` | Learning from (prompt, chosen, rejected) triplets |
| GRPO-style RL training | `train/rft.py` | Policy gradient with multi-rollout sampling |
| Sequence log-probability | `llm/backend.py` | `sequence_logprob()` — required for DPO, GRPO, KL penalty |
| Reference model KL penalty | `train/pref.py`, `train/rft.py` | Prevents catastrophic drift during RL |
| Advantage normalization | `train/rft.py` | Mean/std normalization across rollouts |
| Multi-rollout sampling | `train/rft.py` | N completions per task, reward aggregation |
| Gradient accumulation | `train/*.py` | Effective batch size > 1 on memory-constrained hardware |
| Pluggable verifiers | `verifiers/` | Regex, JSON Schema, pytest — structured VerifyResult with graded reward |
| Custom verifier support | `verifiers/` | Any Python function returning VerifyResult |
| Adapter management | `train/lora.py`, `models.py` | Save/load/stack LoRA adapters |
| Run artifact tracking | `runs.py` | Metrics, config snapshots, accepted trajectories |
| Eval suites with Pass@k | `eval.py` | YAML-defined task suites, per-task verifier config |
| Model pull + conversion | `models.py` | HF download, MLX conversion, quantization |
| OpenAI-compatible serving | `server.py` | Deploy trained models via API |
| Throughput benchmarking | `bench.py` | Tokens/sec measurement for base vs tuned |
| RLM task generation | `rlm/generate.py` | Structured task generation with JSON extraction + fallback |
| Task mutation | `rlm/mutate.py` | Evol-Instruct-style task variation |
| RLM orchestrator | `rlm/loop.py` | Generate → rollout → verify → train → eval → gate |
| Acceptance gating | `rlm/gating.py` | Strict/threshold/EMA gating with state tracking |
| Corpus management | `rlm/corpus.py` | Rolling corpus + difficulty mix |
| Task dedup/similarity (basic) | `rlm/generate.py` | Jaccard similarity + content hash dedup |
| Benchmark history tracking | `rlm/history.py` | JSONL history log across iterations |
| Docker verifier | `verifiers/docker_verifier.py` | Containerized pytest execution |
| RLM monitor (basic) | `server.py` | UI + history/state endpoints |
| Weight pointers | `rlm/weights.py` | Inference/trainer pointer files + staleness |

### Partially implemented (needs extension for RLM)

| Capability | Current state | Gap |
|---|---|---|
| Task generation quality filters | Basic heuristics | Add stronger constraints + semantic filters |
| Similarity filtering | Jaccard overlap | Add embedding-based similarity + benchmark leakage checks |
| Async staleness | Pointer-based staleness | Add off-policy corrections + async worker processes |
| Monitoring dashboard | Basic UI | Add charts for pass rate, reward distribution, and logs |

## Required additions (prioritized)

### P0 — Core RLM loop

These are required to run any RLM training.

#### 1. Task generation module

A module that prompts the loaded model to produce structured coding/reasoning
tasks. Output format:

```json
{
  "id": "task_001",
  "description": "Write a recursive function that ...",
  "function_name": "solve",
  "signature": "def solve(n: int) -> int",
  "tests": [
    {"input": [5], "expected": 120},
    {"input": [0], "expected": 1}
  ]
}
```

Requirements:
- Configurable number of tasks per iteration (default 20-100).
- Multiple generation rounds with prompt diversity.
- Quality filters: description length, test count, input size.
- Deduplication by task ID, function name, and content hash.
- Similarity filtering against benchmark tasks (Jaccard or embedding).
- Fallback to hardcoded seed tasks if generation fails.

Location: `src/mlxsmith/rlm/generate.py`

#### 2. RLM loop orchestrator

A top-level command that runs the full cycle:

```bash
mlxsmith rlm --config rlm.yaml --iterations 50
```

Per iteration:
1. Generate tasks (or load from environment YAML).
2. Collect rollouts (N solutions per task via `rft.py` sampling).
3. Verify rollouts (via configured verifier).
4. Train on graded rollouts (GRPO policy gradient update).
5. Evaluate on held-out benchmark suite.
6. Gate: accept/reject adapter based on benchmark delta.

Requirements:
- Configurable iteration count (0 = infinite).
- Sleep between iterations (for thermal management on laptops).
- Graceful shutdown on SIGINT/SIGTERM.
- Per-iteration run directory with full artifacts.
- Resume from last completed iteration.

Location: `src/mlxsmith/rlm/loop.py`
CLI: `src/mlxsmith/cli.py` (new `rlm` subcommand)

#### 3. Acceptance gating

After each iteration's benchmark:
- Compare new adapter score against historical best.
- If improved: promote to production adapter, update best checkpoint.
- If regressed: rollback to best adapter.
- Record gating decision in iteration metadata.

Requirements:
- Configurable gating strategy (strict improvement, threshold, EMA).
- Best adapter tracking with metadata (iteration, score, timestamp).
- Adapter staging area (train to staging, promote on acceptance).

Location: `src/mlxsmith/rlm/gating.py`

#### 4. Corpus management

Maintain a rolling corpus of verified (prompt, solution) pairs across
iterations:

- Append passing solutions from each iteration.
- Cap corpus size (default 5000-8000 samples).
- FIFO eviction when cap exceeded.
- Mix old corpus into new training data (configurable ratio, default 30-40%).
- Difficulty-aware sampling: weight harder tasks higher (default 60/40 split).

Location: `src/mlxsmith/rlm/corpus.py`

### P1 — Quality and robustness

These improve training signal quality and stability.

#### 5. Task mutation (Evol-Instruct)

Apply transformation operators to generated tasks before solving:

| Operator | Effect |
|---|---|
| `constraints` | Add input validation requirements and ValueError tests |
| `edge_cases` | Add boundary condition test cases |
| `composition` | Compose function with additional operations |
| `difficulty` | Increase complexity (nested structures, larger inputs) |

Requirements:
- Configurable mutations per task (default 1).
- Mutation operators are model-prompted (not hardcoded transforms).
- Mutated tasks go through the same quality filters as generated tasks.

Location: `src/mlxsmith/rlm/mutate.py`

#### 6. Docker sandbox for code verification

The current pytest verifier runs in a lightweight sandbox (per-rollout workdir,
timeout, env isolation). For RLM training where the model generates arbitrary
code, stronger isolation is needed:

- Docker container execution (configurable image, default `python:3.11-slim`).
- Resource limits: memory (512MB), CPU (1.0), PIDs (128).
- Network disabled (`--network none`).
- Read-only filesystem with noexec /tmp.
- Dropped capabilities, no privilege escalation.
- Fallback to local sandbox if Docker unavailable.

This could be a new verifier backend or an option on the existing pytest
verifier.

Location: `src/mlxsmith/verifiers/docker_verifier.py` or extension of
`pytest_verifier.py`

#### 7. Benchmark history and trend tracking

Append-only JSONL log of benchmark results across iterations:

```json
{
  "iteration": 25,
  "timestamp": "2026-01-29T13:49:06",
  "base_score": 3,
  "adapter_score": 8,
  "holdout_score": 5,
  "best_score": 7,
  "accepted": true
}
```

Requirements:
- Separate from per-run metrics.jsonl (this is cross-iteration).
- Queryable for trend analysis.
- Used by gating logic to detect plateaus or regressions.

Location: `src/mlxsmith/rlm/history.py`

### P2 — User experience

These make RLM training accessible as an open-source tool.

#### 8. RLM configuration schema

Extend `config.py` with an `RlmConfig` Pydantic model:

```python
class RlmConfig(BaseModel):
    iterations: int = 50           # 0 = infinite
    sleep_between: int = 0         # seconds between iterations
    tasks_per_iter: int = 80       # synthetic tasks to generate
    rollouts_per_task: int = 8     # solutions per task
    attempts_per_task: int = 3     # retries on failure
    corpus_max: int = 8000         # rolling corpus cap
    mix_old_ratio: float = 0.4     # old data mixing ratio
    hard_ratio: float = 0.6        # hard sample weighting
    mutations_per_task: int = 1    # Evol-Instruct mutations
    gating: str = "strict"         # strict | threshold | ema
    require_recursion: bool = False # enforce recursion in tasks
    task_domains: list[str] = ["strings", "arrays", "math", "dp", "graphs"]
    benchmark_suite: str = "eval/suites/rlm_bench.yaml"
    holdout_suite: str = "eval/suites/rlm_holdout.yaml"
```

#### 9. Monitoring dashboard

Web UI showing real-time RLM training progress:
- Current iteration number and status.
- Benchmark score trend (line chart).
- Pass rate, acceptance rate, reward distribution.
- Recent log output.
- Adapter promotion history.

This can be a simple HTML page served alongside the existing `mlxsmith serve`.

Location: `src/mlxsmith/rlm/monitor.py`

#### 10. CLI integration

```bash
# Initialize an RLM project
mlxsmith init myproject --template rlm

# Run the full loop
mlxsmith rlm --config mlxsmith.yaml --iterations 50

# Check status of a running loop
mlxsmith rlm status

# View benchmark history
mlxsmith rlm history

# Resume from last iteration
mlxsmith rlm --resume
```

### P3 — Advanced

These are stretch goals for competitive parity with cloud RL platforms.

#### 11. Multi-suite evaluation

Run multiple eval suites per iteration (main benchmark + holdout + domain-specific):
- Aggregate scores with configurable weights.
- Gate on composite score, not just one suite.

#### 12. Reward shaping

Support richer reward signals beyond binary pass/fail:
- Partial credit for partially correct solutions.
- Style/efficiency bonuses (e.g., O(n) vs O(n^2)).
- Multi-verifier composition (tests pass AND output matches schema).

#### 13. Curriculum scheduling

Adjust task difficulty across iterations:
- Start with simple tasks, increase complexity as model improves.
- Difficulty tracked per domain.
- Auto-adjust based on pass rate (if pass rate > 80%, increase difficulty).

#### 14. Speculative decoding for rollout collection

Use a smaller draft model to speed up rollout generation:
- Draft model generates candidate tokens.
- Policy model verifies/corrects.
- Reduces time-to-rollout on long completions.

#### 15. Adapter merging

After N iterations, merge accumulated LoRA weights into the base model:
- Reduces inference overhead from adapter application.
- Creates a new base checkpoint for the next training phase.
- `mlxsmith merge --base <model> --adapter <adapter> --output <merged>`

## Proposed directory structure

```
src/mlxsmith/
  rlm/
    __init__.py
    loop.py          # Main RLM orchestrator
    generate.py      # Task generation via model self-play
    mutate.py        # Evol-Instruct task mutation
    corpus.py        # Rolling corpus management
    gating.py        # Acceptance gating logic
    history.py       # Cross-iteration benchmark tracking
    monitor.py       # Web dashboard (optional)
    config.py        # RlmConfig schema
```

## Implementation order

1. **RlmConfig** in `config.py` — define the schema first.
2. **Task generation** — the model needs to produce its own tasks.
3. **Corpus management** — accumulate verified solutions.
4. **Acceptance gating** — promote/rollback adapters.
5. **Loop orchestrator** — wire stages together into `mlxsmith rlm`.
6. **Benchmark history** — trend tracking for gating decisions.
7. **Task mutation** — improve task diversity.
8. **Docker sandbox** — stronger isolation for arbitrary code.
9. **CLI integration** — `mlxsmith rlm` subcommand with status/history/resume.
10. **Monitoring dashboard** — web UI for overnight runs.

## Reference implementation

The Qwen3-4B-RLM project (`/Volumes/VIXinSSD/Qwen3-4B-RLM`) is a working
RLM loop built on raw mlx-lm. It implements task generation, self-critique,
Evol-Instruct mutation, acceptance gating, corpus management, difficulty-aware
sampling, Docker sandboxing, and a monitoring dashboard — all in a single
`rsi/evolve.py` file. The features listed above are informed by what that
implementation proved necessary in practice.

Key differences from MLXCTL's current approach:
- Uses SFT on filtered solutions (rejection sampling) rather than GRPO policy
  gradients. MLXCTL's `rft.py` already has the better approach.
- Shells out to `mlx_lm lora --train` rather than using an in-process training
  loop. MLXCTL's backend abstraction avoids this.
- No preference tuning — discards failed solutions. MLXCTL can use them as
  negative examples via `pref.py`.

The goal is to combine MLXSmith's stronger training algorithms (GRPO, DPO, log-probs,
KL penalty) with the practical RLM orchestration that the reference implementation
proved necessary.
