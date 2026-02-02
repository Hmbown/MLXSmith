# Status (Alpha)

Date: 2026-02-02

## Stable (ship-ready)
- Core CLI (`init`, `doctor`, `pull`, config tooling, data tools).
- SFT (LoRA/QLoRA) with run tracking and adapters.
- OpenAI-compatible `/v1/chat/completions` endpoint + optional streaming.
- HF auth helpers: `mlxsmith auth login/status/logout` for token storage.
- Dataset pull via `mlxsmith data pull` with presets (`mlxsmith data presets`) + quick stats/validation (`mlxsmith data stats/validate`).
- Built-in verifiers: regex, jsonschema, pytest (sandboxed).
- Run tracking: adapter artifacts, metrics, config snapshots, accepted rollouts.

## Experimental (research/dev only)
- Preference tuning (DPO/ORPO) and GRPO-style RFT loops.
- Orchestrated RLM mode (queue-driven inference + trainer workers).
- RLM self-play (no measured gains yet).
- Distill/OPD workflows.
- HF → MLX conversion via `mlx_lm.convert` in `mlxsmith pull`.
## Notes
- Real model loading via `mlx_lm.load`, with adapter application.
- Token-level RL env interface (`token_env`) with tasks shim for RFT.
- Tests + CI workflow added.
- Qwen3-4B Instruct 2507 pipeline validated locally (SFT `runs/sft_0002`, DPO `runs/pref_0002`, RFT `runs/rft_0002`, RLM `runs/rlm_0003`–`runs/rlm_0005`) with serve on `runs/rlm_0003/adapter`.

## Remaining limitations
- ZMLX acceleration is optional and currently best-effort only.
- Eval suite runner is minimal (task-level pass@k + verifier checks).
- Production-grade sandboxing is out of scope (documented in `docs/VERIFIERS.md`).

## Next
- Expand eval suite tooling (benchmark packs, multi-metric reports).
- Add first-class support for speculative decoding.
- See `docs/ROADMAP.md` for the broader product roadmap.
- See `docs/WORKPLAN.md` for parity + app execution details.
- See `docs/orchestrator.md` for multi-process orchestrator design.
