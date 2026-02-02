# mlxsmith

Apple Silicon MLX fine-tuning and OpenAI-compatible serving.
SFT + serving are stable. Preference/RL/RLM features are experimental.

Status: alpha (2026-02-02).

## Stable features
- Project init, config, data tools, HF auth, model pull/convert.
- SFT (LoRA/QLoRA) training with run tracking and adapters.
- Inference and OpenAI-compatible /v1/chat/completions serving.
- Basic eval/bench and verifier plumbing (regex/jsonschema/pytest).

## Experimental features
- Preference tuning (DPO/ORPO).
- GRPO-style RFT.
- RLM self-play loop (research).
- Distill/OPD and orchestrated RLM.

## Install

MLX is only available on Apple Silicon. Other platforms can still use data tools
and mock backends, but MLX training and serving require macOS on Apple Silicon.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip

# Core CLI
pip install mlxsmith

# Apple Silicon training + serving
pip install "mlxsmith[mlx,llm,serve]"
```

## Quickstart

```bash
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

## Pull + convert a model (HF -> MLX)

```bash
mlxsmith pull Qwen/Qwen3-4B-Instruct-2507
# outputs to cache/mlx/Qwen__Qwen3-4B-Instruct-2507
```

Optional quantization:

```bash
mlxsmith pull Qwen/Qwen3-4B-Instruct-2507 --quantize --q-bits 4
```

## SFT (LoRA/QLoRA)

```bash
mlxsmith sft --model cache/mlx/Qwen__Qwen3-4B-Instruct-2507 --data data/sft
```

## Serve (OpenAI-compatible)

```bash
mlxsmith serve --model runs/sft_0001/adapter --port 8080
```

Sample request:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

To enable the optional UI/monitor dashboard, set `serve.ui: true` in `mlxsmith.yaml`.

## Experimental commands

- `mlxsmith pref` (DPO/ORPO)
- `mlxsmith rft` (GRPO-style)
- `mlxsmith rlm` / `mlxsmith pipeline` (self-play loop)
- `mlxsmith distill` (offline/OPD)
- `mlxsmith eval` / `mlxsmith bench`

## Docs

- `docs/PROJECT_FORMAT.md` for project layout and artifacts.
- `docs/VERIFIERS.md` for verifier API and sandbox behavior.
- `docs/COMPATIBILITY.md` for tested versions and model families.
- `docs/ENVIRONMENTS.md` for the environment plugin system.
- `docs/ROADMAP.md` for product direction and milestones.
- `docs/README.md` for the full docs index.

## License

MIT
