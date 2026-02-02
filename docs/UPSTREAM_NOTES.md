# Upstream verification notes (Jan 2026)

Sources verified via public docs for PRIME-RL, Verifiers/Environments Hub, and Tinker. Links below are the authoritative references for the design choices in mlxsmith.

## PRIME-RL entrypoints, roles, and inference weight reload
- PRIME-RL separates **orchestrator**, **trainer**, and **inference**. The orchestrator is a CPU process that collects rollouts, packs batches, and relays updated weights to inference; it uses verifiers environments and async OpenAI-compatible clients. The trainer uses FSDP2 and supports objectives like GRPO/GSPO/OPO/RLOO/CISPO. The inference service is OpenAI-compatible and exposes custom `update_weights` and `reload_weights` endpoints. Docs also note the default entrypoint can use OpenAI API for convenience.  
  https://docs.primeintellect.ai/prime-rl/entrypoints

## PRIME-RL configuration & benchmarking
- PRIME-RL configs use **pydantic-settings** with precedence: CLI args, TOML config files (via `@`), env vars prefixed with `PRIME_` (using `__` for nesting), then defaults.  
  https://docs.primeintellect.ai/prime-rl/configs
- Benchmarking uses `--bench` with SFT and RL modes; RL benchmarking can target trainer-only (fake data), inference (via orchestrator `--bench` with inference server), or full RL (`prime-rl rl ... --bench`).  
  https://docs.primeintellect.ai/prime-rl/benchmarking

## Verifiers / Environments Hub
- Environments are packaged as installable Python projects. `prime env init` scaffolds a template including `pyproject.toml`, README, and a `load_environment()` stub that returns a `vf.Environment`. Environments are uploaded with `prime env push`, optionally to a team or as private.  
  https://docs.primeintellect.ai/tutorials-environments/create
- Hub discovery & install basics: `prime env list`, `prime env info`, `prime env install`, `prime env init`.  
  https://docs.primeintellect.ai/tutorials-environments/getting-started
- Install/use & versioning: install latest or pinned versions, upgrade by re-install, pull source with `prime env pull`, and load via `verifiers.load_environment`.  
  https://docs.primeintellect.ai/tutorials-environments/install
- PRIME-RL can train/evaluate any verifiers environment; installation uses `prime env install` or `uv pip install` against the Hub index.  
  https://docs.primeintellect.ai/prime-rl/environments
- Verifiers overview shows `prime env install` for local or Hub environments and `prime env push` for publishing; evaluations run via `prime eval run`.  
  https://docs.primeintellect.ai/verifiers/overview

## Tinker training API (core functions, futures, logprobs, losses)
- Tinker’s public interface centers on `forward_backward`, `optim_step`, `sample`, and `save_state`.  
  https://tinker-docs.thinkingmachines.ai/  
  https://thinkingmachines.ai/tinker/
- `TrainingClient` provides `forward_backward` (returns an **APIFuture**), `optim_step`, and `save_state` / `load_state`.  
  https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient
- Training+sampling guide: `forward_backward`/`optim_step` return futures; **logprobs** can be computed with the sampling client; **top‑k prompt logprobs** are supported for distillation.  
  https://tinker-docs.thinkingmachines.ai/training-sampling
- Supported Tinker loss functions include **cross_entropy**, **importance_sampling**, **ppo**, **cispo**, and **dro** (token‑level; includes ratio form for importance sampling).  
  https://tinker-docs.thinkingmachines.ai/losses

## Tinker RL environments
- RL env interface uses an `Env` with **`initial_observation`** and **`step`** methods; environments operate on **tokens**, not strings.  
  https://tinker-docs.thinkingmachines.ai/rl/rl-envs

## On‑policy distillation (teacher/student)
- OPD workflow: sample trajectories from the **student**, compute **teacher logprobs** via `compute_logprobs`, set per‑token advantages to **negative reverse KL**, and train via **importance_sampling**. Teacher uses a sampling client (no backprop).  
  https://thinkingmachines.ai/blog/on-policy-distillation/
