# mlxsmith

A **self-hostable** CLI + API for fine-tuning and verifier-driven RL on **Apple Silicon** using **MLX**, with optional acceleration backends.

Features:

- **SFT (LoRA/QLoRA)** training with run tracking + artifacts.
- **Preference tuning** (DPO / ORPO) with optional KL penalty.
- **Verifier-driven RL (GRPO-style)** with per-rollout sandboxing + metrics.
- **OpenAI-compatible** `/v1/chat/completions` API + optional streaming.
- **HF → MLX conversion** via `mlx_lm.convert`.

> Designed to be compatible with the MLX + mlx-lm ecosystem as of **Jan 2026**.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip

# CLI (dev) + serving deps
pip install -e ".[dev,serve]"

# MLX + model tooling (on Apple Silicon)
pip install -e ".[mlx,llm]"

mlxsmith init myproj
cd myproj
mlxsmith doctor
```

## HF auth (optional)

```bash
mlxsmith auth login --token "$HF_TOKEN"
mlxsmith auth status
mlxsmith auth logout
```

## Pull + convert a model (HF → MLX)

```bash
mlxsmith pull Qwen/Qwen3-4B-Instruct-2507
# outputs to cache/mlx/Qwen__Qwen3-4B-Instruct-2507
```

Optional quantization:

```bash
mlxsmith pull Qwen/Qwen3-4B-Instruct-2507 --quantize --q-bits 4
```

## SFT (LoRA)

```bash
mlxsmith sft --model cache/mlx/Qwen__Qwen3-4B-Instruct-2507 --data data/sft
```

## Preference tuning (DPO / ORPO)

```bash
mlxsmith pref --model runs/sft_0001/adapter --data data/prefs
```

## Verifier-driven RL (GRPO-style)

```bash
mlxsmith rft --model runs/sft_0001/adapter --env envs/coding.yaml --verifier verifiers/regex.py --rollouts 4
```

## Distillation (offline / OPD)

```bash
mlxsmith distill --data data/distill/prompts.jsonl --teacher cache/mlx/TEACHER --student cache/mlx/STUDENT --mode offline
mlxsmith distill --data data/distill/prompts.jsonl --teacher cache/mlx/TEACHER --student cache/mlx/STUDENT --mode opd
```

## Eval

```bash
mlxsmith eval --model runs/rft_0001/adapter --suite eval/suites/coding.yaml
```

## Serve (OpenAI-compatible)

```bash
mlxsmith serve --model runs/rft_0001/adapter --port 8080
```

To enable the optional UI/monitor dashboard, set `serve.ui: true` in `mlxsmith.yaml`.

## Environments

```bash
mlxsmith env init myenv
mlxsmith env run myenv --model runs/sft_0001/adapter
```

### Sample request

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

## Project layout

See `docs/PROJECT_FORMAT.md` for full details.

## Verifiers

See `docs/VERIFIERS.md` for the verifier API and sandbox behavior.

## Compatibility

See `docs/COMPATIBILITY.md` for tested versions and model families.

## Environments

See `docs/ENVIRONMENTS.md` for the environment plugin system.

## Roadmap

See `docs/ROADMAP.md` for the latest product direction and milestones.

## Docs index

See `docs/README.md` for a full documentation index.

## Workplan

See `docs/WORKPLAN.md` for parity gaps, app plans, and handoff checklists.

## Prime Intellect Design Notes

See `docs/prime-intellect-design-notes.md` for alignment notes and deltas.

## License

MIT
