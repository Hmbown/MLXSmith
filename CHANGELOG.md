# Changelog

All notable changes to MLXSmith are documented in this file.

## Unreleased

_No unreleased changes._

## 0.1.9

- **Qwen3-1.7B support:** End-to-end smoke scripts and configs; repo-grounded SFT workflow.
- **Output cleanup:** Optional `infer.strip_think` strips `<think>` blocks and chat markers like `<|im_end|>` from API and orchestrator output.
- **Eval improvements:** Suites can embed pytest `tests`; added `eval/suites/coding.yaml`; `mlxsmith serve` exposes `/eval/last/results.json`.
- **Fix:** RLM weight pointers reset correctly when starting fresh or switching base models (prevents LoRA shape mismatches).
- **Fix:** SFT saves the latest adapter on SIGINT/SIGTERM for safer interruption of long runs.

## 0.1.8

- **Web dashboard (Next.js):** Full workflow UI covering models, adapters, training, eval, chat, and serving.
- **Models API:** Delete cached models via `/internal/models/delete`.
- **Experimental mHC adapters:** Block-local patching with Sinkhorn-Knopp mixing. Opt-in via `accel.mhc` config and CLI flags on `serve`/`bench`.
- **Utility scripts:** Repo prompt seeding and Codex batch generation helpers.

## 0.1.7

- **External model backends:** `cli` backend for Codex/Claude/Gemini CLIs; `openai` backend for any OpenAI-compatible API.
- **RLM REPL inference:** `mlxsmith rlm infer` and `mlxsmith rlm collect` with local and Docker sandbox execution.
- **Synthetic generation:** `--system-prompt` supports `@file` and file path inputs.
- **Documentation:** Added troubleshooting, FAQ, and CLI coverage for model management, auth, environments, acceleration, and losses.
- **Fix:** Judge training runs now write to `runs/judge_####/`.
- **Fix:** Ruff `F541` in `mlxsmith rlm collect`.

## 0.1.6

- Initial public alpha release.
