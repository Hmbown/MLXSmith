# Changelog

## Unreleased

- (none)

## 0.1.8

- Web dashboard (Next.js): full workflow UI (Dashboard, models, adapters, training, eval, chat, serving).
- Models API: delete cached models via `/internal/models/delete`.
- Experimental mHC adapters:
  - New `mlxsmith.mhc` module (block-local patching + Sinkhorn-Knopp mixing).
  - Opt-in via `accel.mhc` (plus `mhc_n`, `mhc_tmax`) and CLI flags on `serve`/`bench`.
- Utility scripts for repo prompt seeding and Codex batch generation.

## 0.1.7

- External model backends for data/judging:
  - `cli` backend for shelling out to Codex/Claude/Gemini CLIs
  - `openai` backend for OpenAI-compatible Chat Completions APIs
- Canonical RLM REPL inference:
  - `mlxsmith rlm infer` and `mlxsmith rlm collect`
  - Local and Docker sandbox execution
- Synthetic generation: `--system-prompt` supports `@file`/file path inputs.
- Docs accuracy fixes across training commands, configuration defaults, and run paths.
- Judge training runs now write to `runs/judge_####/`.
- Added troubleshooting, FAQ, and CLI coverage for model management, auth, environments, acceleration, and losses.
- Fix: ruff `F541` in `mlxsmith rlm collect`.

## 0.1.6

- Initial public alpha release.
