# Status (Alpha)

Date: 2026-02-02

## Features
- Core CLI (`init`, `doctor`, `pull`, `quantize`, config tooling, data tools).
- SFT (LoRA/QLoRA) with run tracking and adapters.
- Preference tuning (DPO/ORPO) with configurable beta and KL coefficients.
- Reinforced fine-tuning (GRPO) with token-level environments and verifier-based rewards.
- Knowledge distillation (offline and OPD modes).
- OpenAI-compatible `/v1/chat/completions` endpoint + streaming.
- mlx-lm-lora passthrough (advanced training modes + synthetic datasets).
- HF auth helpers: `mlxsmith auth login/status/logout`.
- Dataset presets, pull, import, split, stats, validation.
- Built-in verifiers: regex, jsonschema, pytest (sandboxed), docker, compose, llm_judge.
- Environment plugin system for RFT task/verifier packaging.
- SDK: SamplingClient, TrainingClient, loss registry (DPO, ORPO, GRPO, CISPO, DRO, PPO).
- Adapter merging, eval suites (pass@k), benchmarking.
- Run tracking: adapter artifacts, metrics, config snapshots.

## Research
- RLM self-play loop (infrastructure runs, no measured gains yet).
- Orchestrated RLM mode (queue-driven inference + trainer workers).
- ZMLX acceleration (optional, best-effort).

## Validated
- Full pipeline validated locally on Qwen3-4B Instruct 2507: SFT (`runs/sft_0002`), DPO (`runs/pref_0002`), RFT (`runs/rft_0002`), RLM (`runs/rlm_0003`–`runs/rlm_0005`), serve on `runs/rlm_0003/adapter`.

## Remaining limitations
- Eval suite runner is minimal (task-level pass@k + verifier checks).
- Production-grade sandboxing is out of scope (documented in `docs/VERIFIERS.md`).

## Next
- Expand eval suite tooling (benchmark packs, multi-metric reports).
- Add first-class support for speculative decoding.
- See `docs/ROADMAP.md` for the broader product roadmap.
- See `docs/WORKPLAN.md` for parity + app execution details.
- See `docs/orchestrator.md` for multi-process orchestrator design.
