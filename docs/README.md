# MLXSmith Documentation

## Getting Started

- [Getting Started](getting-started.md) — Install, create a project, and run your first training job
- [Concepts](concepts.md) — Training modes explained
- [Troubleshooting](troubleshooting.md) — Common issues and fixes
- [FAQ](FAQ.md) — Frequently asked questions

## CLI Reference

- [Command Index](cli/README.md) — All commands at a glance

**Training:**

- [SFT](cli/sft.md) — Supervised fine-tuning
- [Preference Training](cli/preference-training.md) — DPO, ORPO, IPO, CPO, SimPO, TDPO
- [KTO](cli/kto.md) — Kahneman-Tversky Optimization (binary feedback)
- [Reinforcement Training](cli/reinforcement-training.md) — GRPO with verifier rewards
- [Online DPO](cli/online-dpo.md) — Online preference tuning with LLM judge
- [Self-Verify](cli/self-verify.md) — Self-verification training
- [Distillation](cli/distillation.md) — Teacher-to-student knowledge transfer
- [Judge](cli/judge.md) — Train a scoring model

**Data and Evaluation:**

- [Data Tools](cli/data.md) — Import, split, validate, and download datasets
- [Synthetic Data](cli/synthetic-data.md) — Generate and evolve training data
- [Eval and Bench](cli/eval-and-bench.md) — Evaluation suites and benchmarking

**Infrastructure:**

- [Serving](cli/serving.md) — OpenAI-compatible model server
- [RLM](cli/rlm.md) — Recursive self-improving training loop
- [Configuration](cli/configuration.md) — Config system, environment variables, and options
- [Model Management](cli/model-management.md) — Pull, quantize, and merge adapters
- [Auth](cli/auth.md) — Hugging Face token management
- [Environments](cli/environments.md) — Environment registry and packaging
- [Acceleration](cli/accel.md) — Backend status
- [Losses](cli/losses.md) — Registered loss functions

## Reference

- [Verifiers](VERIFIERS.md) — Verifier API, built-in verifiers, composition, and sandboxing
- [Environments](ENVIRONMENTS.md) — Environment plugin system and local registry
- [Project Format](PROJECT_FORMAT.md) — Project layout and run artifacts
- [Compatibility](COMPATIBILITY.md) — Tested MLX, mlx-lm, and model versions
- [Orchestrator](orchestrator.md) — Multi-process RLM architecture
- [RLM Design](rlm-ctl.md) — RLM training loop design and implementation notes
- [Roadmap](ROADMAP.md) — Planned features
