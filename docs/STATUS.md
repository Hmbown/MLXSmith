# Status (Ready)

Date: 2026-02-02

## What works now
- CLI commands run end-to-end (`init`, `doctor`, `pull`, `sft`, `pref`, `rft`, `eval`, `serve`, `bench`).
- HF → MLX conversion via `mlx_lm.convert` in `mlxsmith pull`.
- HF auth helpers: `mlxsmith auth login/status/logout` for token storage.
- Real model loading via `mlx_lm.load`, with adapter application.
- SFT (LoRA/QLoRA), preference tuning (DPO/ORPO), and GRPO-style RFT loops.
- Built-in verifiers: regex, jsonschema, pytest (sandboxed).
- OpenAI-compatible `/v1/chat/completions` endpoint + optional streaming.
- Run tracking: adapter artifacts, metrics, config snapshots, accepted rollouts.
- Orchestrated RLM mode (queue-driven inference + trainer workers).
- Tests + CI workflow added.

## Remaining limitations
- ZMLX acceleration is optional and currently best-effort only.
- Eval suite runner is minimal (task-level pass@k + verifier checks).
- Production-grade sandboxing is out of scope (documented in `docs/VERIFIERS.md`).

## Next
- Expand eval suite tooling (benchmark packs, multi-metric reports).
- Add first-class support for speculative decoding.
- See `docs/ROADMAP.md` for the broader product roadmap.
- See `docs/WORKPLAN.md` for parity + app execution details.
- See `docs/prime-intellect-design-notes.md` for design alignment notes.
