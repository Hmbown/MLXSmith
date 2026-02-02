# MLXSMITH Roadmap (2026)

Date: 2026-02-02

## Vision
MLXSMITH is the single, stable entrypoint for MLX training and serving on Apple
Silicon: a CLI + API + desktop app that makes the Prime‑RL workflow approachable,
repeatable, and observable.

## Principles (mirror PrimeIntellect setup)
- **Orchestrator / Inference / Trainer split** with explicit weight pointers and
  staleness control.
- **Verifier‑first RL** with deterministic interfaces, composition, and safe
  execution (Docker by default where possible).
- **Environment‑driven tasks** (local registry, package/publish workflow).
- **Observable training**: metrics, history, gating decisions, dashboards.
- **Reproducible runs**: config snapshots, run artifacts, bench data.
- **Local‑first** with optional HF authentication for pull/convert/serve.

## Current state (implemented)
- RLM loop: generate → rollout → verify → train → eval → gate.
- Task mutation + basic similarity filtering.
- Docker verifier backend + composition layer.
- Loss registry with RL losses (IS / PPO / CISPO / DRO) and SFT/Pref losses.
- Distillation (offline + OPD).
- Bench modes for inference/trainer/end‑to‑end.
- Serve UI + RLM monitor endpoints.
- Adapter merge tool.
- Environment plugin system + local registry.

## Roadmap (next priorities)

### P0 — Product polish
1) **Rename CLI + docs to mlxsmith** (binary, package, docs, examples). ✅ Done
2) **Config schema cleanup** with explicit defaults and validation warnings.
3) **Prompt/task quality filters**: richer heuristics and embedding‑based
   similarity to avoid benchmark leakage.
4) **Verifier reporting**: standardized latency histograms + failure taxonomy.

### P1 — Training quality
5) **Async workers**: decouple inference and trainer processes with queueing.
6) **Off‑policy corrections**: configurable PPO/CISPO/DRO in RLM trainer.
7) **Task generation quality gates**: robustness tests and cross‑domain mixes.

### P2 — Distribution & packaging
8) **Environment registry UX**: browsing, versioning, import/export.
9) **Model registry view**: detect cached models, metadata, and size.
10) **Release artifacts**: prebuilt wheels + notarized macOS app bundle.

### P2.5 — Research add-ons
11) **PARL (agentic training)**: tool-using rollouts with verifier-gated rewards (under consideration).
12) **PRIME-style process rewards**: implicit process reward shaping + stepwise scoring. ✅ Implemented
13) **ThinkPRM / self-verification**: LLM judge verifier with process grading. ✅ Implemented
14) **Muon optimizer**: Newton-Schulz orthogonalization optimizer. ✅ Implemented
15) **Online-DPO**: online preference optimization from judge rewards. ✅ Implemented
16) **Self-verify training**: policy gradient with LLM judge reward signal. ✅ Implemented

### P3 — macOS Swift app (mlxsmith Studio)
Goal: a native UI that mirrors the CLI workflows and streamlines onboarding.

**Core features**
- **Hugging Face token storage** (Keychain) + validation.
- **Model pull manager** using `mlxsmith pull` with progress/log streaming.
- **Serve control**: start/stop, health status, chat UI w/ streaming.
- **RLM monitor**: show history, gating decisions, and current weight pointers.
- **Environment registry**: browse, install, and run envs.

**Architecture**
- SwiftUI + Combine/async for streaming logs and SSE.
- `Process` wrapper around CLI with stdout/stderr ingestion.
- Keychain helper for HF token; never store plaintext.
- Local settings for project root, cache paths, and model defaults.

## Deliverables checklist
- [x] Rename CLI package/binary to mlxsmith.
- [ ] SwiftUI macOS app scaffold + Keychain token storage.
- [ ] Async inference/trainer queues (multi‑process).
- [ ] Embedding‑based task similarity filters.
- [ ] Verifier performance dashboards (latency + pass/fail).
