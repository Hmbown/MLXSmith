# mlxsmith

Apple Silicon MLX fine-tuning toolkit — SFT, DPO/ORPO, GRPO, distillation, and OpenAI-compatible serving.

**Status:** alpha (v0.1.1). Full training pipeline validated on Qwen3-4B.

## Install

MLX training and serving require macOS on Apple Silicon.
Other platforms can use data tools and mock backends.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip

# Core CLI (data tools, config, project scaffolding)
pip install mlxsmith

# Apple Silicon training + serving
pip install "mlxsmith[mlx,llm,serve]"

# Everything
pip install "mlxsmith[all]"
```

## Quickstart

```bash
mlxsmith init myproj
cd myproj
mlxsmith doctor        # check Python, MLX, Metal
```

## Training

### SFT (LoRA/QLoRA)

```bash
mlxsmith sft --model cache/mlx/Qwen__Qwen3-4B-Instruct-2507 --data data/sft
```

Produces run artifacts under `runs/sft_NNNN/` (adapter weights, `metrics.jsonl`, config snapshot).

### Preference tuning (DPO/ORPO)

```bash
mlxsmith pref --model cache/mlx/Qwen__Qwen3-4B-Instruct-2507 \
  --data data/prefs --algo dpo
```

Supports DPO and ORPO algorithms with configurable beta and KL coefficients. Expects `{prompt, chosen, rejected}` data format.

### Reinforced fine-tuning (GRPO)

```bash
mlxsmith rft --model cache/mlx/Qwen__Qwen3-4B-Instruct-2507 \
  --env envs/coding.yaml --verifier verifiers/pytest.py
```

GRPO-style RL training with token-level environment integration and verifier-based rewards. Rollout acceptance/rejection gating with reward tracking.

### Knowledge distillation

```bash
# Offline distillation (teacher generates, student learns)
mlxsmith distill --teacher large-model --student small-model --mode offline

# Online preference distillation (OPD)
mlxsmith distill --teacher large-model --student small-model --mode opd
```

### Full pipeline

```bash
# Run SFT → Pref → RFT in sequence
mlxsmith pipeline
```

## Serving

OpenAI-compatible `/v1/chat/completions` endpoint.

```bash
mlxsmith serve --model runs/sft_0001/adapter --port 8080
```

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

Supports streaming (`"stream": true`), logprobs, stop sequences, and an optional UI dashboard (`serve.ui: true` in config).

## Data tools

```bash
mlxsmith data presets                                     # list built-in datasets
mlxsmith data pull alpaca                                 # pull a preset
mlxsmith data import raw.json --out data/sft/train.jsonl  # import ShareGPT → JSONL
mlxsmith data split data/sft/train.jsonl --fractions 0.9 0.05 0.05
mlxsmith data stats data/sft/train.jsonl                  # token counts, field analysis
mlxsmith data validate data/sft/train.jsonl               # structure check
```

Built-in presets: `alpaca`, `hh-rlhf`, `ultrachat-200k`, `ultrafeedback-binarized-prefs`, `ultrafeedback-binarized-sft`.

## Model management

```bash
# Pull + convert HF model to MLX
mlxsmith pull Qwen/Qwen3-4B-Instruct-2507

# With quantization
mlxsmith pull Qwen/Qwen3-4B-Instruct-2507 --quantize --q-bits 4

# Merge adapters
mlxsmith adapters merge runs/sft_0001/adapter runs/pref_0001/adapter --weights 0.7 0.3
```

## HF auth

```bash
mlxsmith auth login --token "$HF_TOKEN"
mlxsmith auth status
mlxsmith auth logout
```

## Eval and bench

```bash
# Evaluation suite (pass@k with verifier checks)
mlxsmith eval --suite eval/suites/coding.yaml

# Benchmark inference or training throughput
mlxsmith bench --mode inference
mlxsmith bench --mode trainer
mlxsmith bench --mode end_to_end
```

## Verifiers

Built-in verifiers for eval, RFT, and preference tuning:

- **regex** — pattern matching on completions
- **jsonschema** — JSON structure validation
- **pytest** — sandboxed test execution
- **docker** — containerized verification
- **compose** — multi-verifier composition (AND/OR/weighted)

See `docs/VERIFIERS.md` for the verifier API.

## Environment plugin system

```bash
mlxsmith env list                  # list available environments
mlxsmith env info envs/coding.yaml # show manifest (tasks, verifier, version)
mlxsmith env init my_env           # scaffold a new environment
mlxsmith env install ./my_env      # install from directory
mlxsmith env package ./my_env      # create distributable tarball
mlxsmith env run envs/coding.yaml  # execute RFT with this environment
```

Environments define tasks, verifiers, and reward functions for RFT training. See `docs/ENVIRONMENTS.md`.

## Config system

```bash
mlxsmith config show              # display merged config (YAML/JSON/TOML)
mlxsmith config show --sources    # show where each value comes from
mlxsmith config init              # create default mlxsmith.yaml
mlxsmith config validate          # check config structure
mlxsmith config env               # show environment variable mapping
```

Config sources (in priority order): CLI flags > environment variables (`MLXSMITH__SECTION__KEY`) > config file > defaults.

## SDK (programmatic API)

For building custom training loops:

```python
from mlxsmith.sdk import load_model, SamplingClient, TrainingClient, TrainingBatch

loaded = load_model("path/to/model", config)

# Sampling with logprobs
sampler = SamplingClient(loaded.backend)
result = sampler.sample("prompt", logprobs_k=5)

# Training operations
trainer = TrainingClient(loaded.backend)
trainer.create_optimizer(lr=1e-4, weight_decay=0.01)
fb = trainer.forward_backward(batch)
trainer.optim_step(fb.result().grads)
```

Loss functions: DPO, ORPO, GRPO, CISPO, DRO, PPO, importance sampling, cross-entropy.

## Research

### RLM self-play loop

RLM (Recursive Language Model) is a research feature — the infrastructure runs but has not produced measured gains yet.

```bash
mlxsmith rlm                       # single-process RLM
mlxsmith pipeline --orchestrated   # multi-process orchestrated RLM
mlxsmith rlm status                # check iteration state
mlxsmith rlm history               # view history
```

Includes task generation, mutation for data diversity, corpus management, EMA-based gating, and weight pointer IPC for multi-process coordination. See `docs/orchestrator.md`.

## Docs

- `docs/PROJECT_FORMAT.md` — project layout and artifacts
- `docs/VERIFIERS.md` — verifier API and sandbox behavior
- `docs/COMPATIBILITY.md` — tested versions and model families
- `docs/ENVIRONMENTS.md` — environment plugin system
- `docs/orchestrator.md` — multi-process RLM orchestrator
- `docs/rlm-ctl.md` — RLM training guide
- `docs/ROADMAP.md` — product direction and milestones
- `docs/README.md` — full docs index

## License

MIT
