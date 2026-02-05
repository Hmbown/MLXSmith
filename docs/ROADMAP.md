# Roadmap

Last updated: 2026-02-02

## Vision

MLXSmith is the single, stable entry point for MLX training and serving on Apple Silicon: a CLI, API, and desktop app that makes advanced training workflows approachable, repeatable, and observable.

## Principles

- **Orchestrator / Inference / Trainer split** with explicit weight pointers and staleness control.
- **Verifier-first RL** with deterministic interfaces, composition, and safe execution.
- **Environment-driven tasks** with a local registry, package, and publish workflow.
- **Observable training** through metrics, history, gating decisions, and dashboards.
- **Reproducible runs** via config snapshots, run artifacts, and benchmark data.
- **Local-first** with optional Hugging Face authentication for pull/convert/serve.

## Implemented

- RLM loop: generate, rollout, verify, train, eval, gate.
- Task mutation and basic similarity filtering.
- Docker verifier backend and composition layer.
- Loss registry with RL losses (IS, PPO, CISPO, DRO) and SFT/preference losses.
- Distillation (offline and OPD).
- Benchmark modes for inference, trainer, and end-to-end pipelines.
- Serve UI and RLM monitor endpoints.
- Adapter merge tool.
- Environment plugin system with local registry.
- Process reward models (ThinkPRM) and self-verification training.
- Online DPO with LLM judge.
- Muon optimizer.

## Planned

### Product Polish

- Config schema cleanup with explicit defaults and validation warnings.
- Prompt and task quality filters: richer heuristics and embedding-based similarity to avoid benchmark leakage.
- Verifier reporting: standardized latency histograms and failure taxonomy.

### Training Quality

- Async workers: decouple inference and trainer processes with queuing.
- Off-policy corrections: configurable PPO/CISPO/DRO in the RLM trainer.
- Task generation quality gates: robustness tests and cross-domain mixes.

### Distribution and Packaging

- Environment registry UX: browsing, versioning, import/export.
- Model registry view: detect cached models, metadata, and size.
- Release artifacts: prebuilt wheels and notarized macOS app bundle.

### Native macOS App

A SwiftUI desktop app that mirrors CLI workflows and streamlines onboarding. Includes Hugging Face token storage (Keychain), model management, serve control, training UI, and RLM monitoring.

### Research

- PARL (agentic training): tool-using rollouts with verifier-gated rewards.
- Speculative decoding for faster synthetic generation.
- Expanded evaluation suite tooling (benchmark packs, multi-metric reports).
