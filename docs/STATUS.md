# Status

**Current version:** 0.1.9 (Alpha)
**Last updated:** 2026-02-02

## Implemented Features

- Core CLI: `init`, `doctor`, `pull`, `quantize`, config tooling, data tools.
- SFT (LoRA/QLoRA) with run tracking and adapter artifacts.
- Preference optimization (DPO, ORPO, IPO, CPO, SimPO, TDPO) with configurable beta and KL coefficients.
- Reinforcement fine-tuning (GRPO, DR-GRPO, DAPO) with verifier-based rewards.
- KTO (binary feedback) training pipeline.
- Knowledge distillation (offline and OPD modes).
- Online DPO with LLM judge scoring.
- Self-verification training with policy gradient rewards.
- Synthetic data generation: Evol-Instruct prompt evolution and rejection-sampled SFT.
- OpenAI-compatible `/v1/chat/completions` endpoint with streaming.
- HF auth helpers: `mlxsmith auth login/status/logout`.
- Dataset presets, pull, import, split, stats, and validation.
- Built-in verifiers: regex, JSON schema, pytest (sandboxed), Docker, compose, LLM judge.
- Environment plugin system for RFT task/verifier packaging.
- SDK: SamplingClient, TrainingClient, loss registry (DPO, ORPO, GRPO, CISPO, DRO, PPO).
- Adapter merging, eval suites (pass@k), and throughput benchmarking.
- Run tracking: adapter artifacts, metrics JSONL, and config snapshots.

## Research

- RLM self-play loop (infrastructure complete; no measured gains yet).
- Orchestrated RLM mode (queue-driven inference and trainer workers).

## Validated Models

- Qwen3-4B Instruct (full pipeline: SFT, DPO, RFT, RLM, serve).
- Qwen3-1.7B (end-to-end smoke test).

## Known Limitations

- Eval suite runner is minimal (task-level pass@k with verifier checks).
- Production-grade sandboxing is out of scope (see [Verifiers](VERIFIERS.md)).

## Next Steps

See [Roadmap](ROADMAP.md) for the full plan.
