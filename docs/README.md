# Documentation

## Getting Started

- [Getting Started](getting-started.md) — install, set up a project, and run your first training job
- [Concepts](concepts.md) — training modes explained
- [Troubleshooting](troubleshooting.md) — common issues and fixes
- [FAQ](FAQ.md) — frequently asked questions

## CLI Guides

- [CLI Reference](cli/README.md) — full command index
- [Project Setup](cli/project-setup.md) — init and doctor
- [SFT](cli/sft.md) — supervised fine-tuning
- [Preference Training](cli/preference-training.md) — DPO, ORPO, and other preference algorithms
- [KTO](cli/kto.md) — Kahneman-Tversky Optimization (binary feedback)
- [Reinforcement Training](cli/reinforcement-training.md) — GRPO with verifier rewards
- [Online DPO](cli/online-dpo.md) — online preference tuning with LLM judge
- [Self-Verify](cli/self-verify.md) — self-verification training
- [Distillation](cli/distillation.md) — teacher-to-student knowledge transfer
- [Synthetic Data](cli/synthetic-data.md) — generate and evolve training data
- [Judge](cli/judge.md) — train a scoring model
- [RLM](cli/rlm.md) — recursive self-improving training loop
- [Serving](cli/serving.md) — OpenAI-compatible model server
- [Data Tools](cli/data.md) — import, split, validate, and pull datasets
- [Eval and Bench](cli/eval-and-bench.md) — evaluation suites and benchmarking
- [Configuration](cli/configuration.md) — config system, environment variables, and options
- [Model Management](cli/model-management.md) — pull, quantize, merge adapters
- [Auth](cli/auth.md) — Hugging Face token helpers
- [Environments](cli/environments.md) — environment registry and packaging
- [Acceleration](cli/accel.md) — backend status
- [Losses](cli/losses.md) — registered loss functions

## Reference

- [Verifiers](VERIFIERS.md) — verifier API, built-in verifiers, composition, and sandboxing
- [Environments](ENVIRONMENTS.md) — environment plugin system and local registry
- [Project Format](PROJECT_FORMAT.md) — project layout and run artifacts
- [Compatibility](COMPATIBILITY.md) — tested MLX, mlx-lm, and model versions
- [Orchestrator](orchestrator.md) — multi-process RLM architecture
- [RLM Design](rlm-ctl.md) — RLM training loop design and implementation notes
