# MLXSmith Code Review & Enhancement Prompt

You are reviewing and extending **MLXSmith**, an Apple Silicon MLX fine-tuning toolkit. A recent refactor replaced a subprocess-based `mlx-lm-lora` integration with native implementations. Your job is to (1) audit the refactor for correctness, and (2) extend the toolkit with state-of-the-art training methods as of February 2026.

---

## Project Overview

MLXSmith is a CLI toolkit (`mlxsmith`) for LLM fine-tuning on Apple Silicon using MLX. It supports:
- **SFT** (supervised fine-tuning)
- **Preference optimization** (DPO, CPO, IPO, ORPO, hinge)
- **Reinforcement fine-tuning** (GRPO, DR-GRPO, DAPO)
- **Online DPO** (generate + judge + train loop)
- **Self-verify** (generate + self-verify + train)
- **Distillation** (offline and online policy distillation)
- **RLM** (recursive language model improvement loop)
- **Synthetic data generation** (prompts, SFT pairs, DPO pairs) — *newly added*
- **Judge model training** — *newly added*
- **Loss function registry** — *newly exposed via CLI*
- **Environment plugins** for task/verifier packaging
- **OpenAI-compatible serving** via FastAPI

### Key Architecture

```
src/mlxsmith/
├── cli.py                    # Typer CLI entry point
├── config.py                 # Pydantic config with env/file/CLI precedence
├── synthetic.py              # NEW: synthetic data generation
├── llm/
│   ├── backend.py            # LLMBackend protocol + Generation/DecodingConfig dataclasses
│   ├── registry.py           # get_llm_backend("mlx-lm" | "mock")
│   ├── interface.py          # High-level: load_base_model(), generate(), chat(), compute_logprobs()
│   ├── mlx_lm_backend.py     # Concrete MLX backend
│   └── mock_backend.py       # Mock for testing
├── sdk/
│   └── losses.py             # LOSS_REGISTRY with @register_loss decorator
├── train/
│   ├── sft.py                # run_sft()
│   ├── pref.py               # run_pref() — DPO/CPO/IPO/ORPO/hinge
│   ├── rft.py                # run_rft() — GRPO/DR-GRPO/DAPO
│   ├── online_dpo.py         # run_online_dpo() — generate + judge + preference train
│   ├── self_verify.py        # run_self_verify()
│   ├── distill.py            # run_distill()
│   └── lora.py               # LoRAConfig dataclass
├── verifiers/
│   ├── llm_judge.py          # verify() with judge/thinkprm modes
│   ├── regex.py, pytest_verifier.py, jsonschema.py, prime.py
│   └── types.py              # VerifyResult(reward, passed, info)
├── rlm/                      # Recursive language model loop
├── models.py                 # resolve_model_spec(), hf_pull()
├── adapters.py               # merge_adapters()
├── envs/                     # Environment plugin system
└── integrations/
    └── __init__.py            # Now empty (mlx_lm_lora removed)
```

---

## What Was Just Changed (Refactor Summary)

### Removed
- `src/mlxsmith/integrations/mlx_lm_lora.py` — subprocess wrapper that shelled out to `mlx-lm-lora` pip package
- `tests/test_lora_integration.py` — tests for the removed integration
- `pyproject.toml` — removed `lora = ["mlx-lm-lora>=1.0.0"]` optional dep and from `all` extras
- `cli.py` — removed `lora_app` Typer group with 4 passthrough commands (train, synthetic, judge, reward-functions)

### Added
- **`src/mlxsmith/synthetic.py`** — Native synthetic data generation:
  - `generate_prompts(model, cfg, out, *, num, seed_prompts, ...)` → JSONL with `{"prompt": "..."}`
  - `generate_sft(model, cfg, prompts_path, out, ...)` → JSONL with `{"prompt": ..., "response": ...}`
  - `generate_dpo(model, cfg, prompts_path, out, *, candidates_per_prompt, judge_model, ...)` → JSONL with `{"prompt": ..., "chosen": ..., "rejected": ...}`
  - Uses `get_llm_backend()`, `resolve_model_spec()`, `llm.generate()`, `judge_verify()` — same patterns as `train/online_dpo.py`

- **CLI commands** in `cli.py`:
  - `mlxsmith synthetic prompts --model M --out FILE`
  - `mlxsmith synthetic sft --model M --prompts FILE --out FILE`
  - `mlxsmith synthetic dpo --model M --prompts FILE --out FILE`
  - `mlxsmith judge --model M --data DIR` (thin wrapper around `run_sft()`)
  - `mlxsmith losses` (prints LOSS_REGISTRY table)

- **`tests/test_synthetic.py`** — 3 mock-based tests covering all generation functions

### Current State
- 127/127 tests pass
- `ruff check src/` clean
- CLI shows synthetic, judge, losses commands; no lora

---

## Task 1: Audit the Refactor

Review these files for correctness, edge cases, and consistency with existing patterns:

1. **`src/mlxsmith/synthetic.py`** — Check:
   - Does `_load_llm()` handle adapter resolution correctly? Compare with `train/online_dpo.py` lines 64-93 which also handles `BackendNotAvailable` and LoRA config fallback — the synthetic module skips both.
   - Is `_read_prompts()` consistent with `_iter_prompts()` in `train/online_dpo.py`? The online_dpo version also checks `"messages"` key.
   - Does `generate_dpo()` handle the case where all candidates get the same score (all tied)?
   - Should `generate_sft()` / `generate_dpo()` support batching or streaming for large prompt files?
   - Is the prompt construction in `generate_prompts()` robust enough, or could the LLM echo back the prefix?

2. **`src/mlxsmith/cli.py`** — Check:
   - Are all CLI options consistent with existing command patterns (types, defaults, help text)?
   - Does the `judge` command correctly delegate to `run_sft()`? Should it set any special config overrides for judge training?
   - Does `losses` correctly trigger registration of all loss functions?

3. **`tests/test_synthetic.py`** — Check:
   - Are the mocks realistic? Do they test failure paths?
   - Should there be tests for empty input, malformed JSONL, zero candidates, etc.?

4. **`pyproject.toml`** — Verify no other files reference `mlx-lm-lora` or the removed integration.

---

## Task 2: Extend with 2026 Training Methods

Research and implement the latest training methods as of February 2026. Here are the areas to investigate and potential additions:

### A. Loss Functions (add to `sdk/losses.py`)

Add to `LOSS_REGISTRY` with the `@register_loss` decorator. The existing pattern:

```python
@register_loss("name")
def name_loss(backend, token_ids_or_chosen_rejected, *, prompt_len, ...):
    mx = _require_mx(backend)
    # compute and return scalar loss
```

Research and implement if they exist by Feb 2026:
- **SimPO** (Simple Preference Optimization) — reference-free, length-normalized
- **KTO** (Kahneman-Tversky Optimization) — works with binary feedback (no paired preferences)
- **SPPO** (Self-Play Preference Optimization)
- **ORPO v2** or improved ORPO variants
- **Token-level DPO / TDPO** — per-token preference optimization
- **RLOO** (REINFORCE Leave-One-Out) — unbiased policy gradient
- **REBEL** (Reward-regularized) — if formalized by 2026
- **Online iterative DPO** improvements (e.g., self-play, iterative RPO)
- **DAPO** improvements or successors (Dynamic Advantage Policy Optimization)
- **Any new RL-from-human-feedback loss that became standard post-May 2025**

### B. Training Pipelines (add to `train/`)

Each training pipeline follows this pattern (see `train/online_dpo.py`):
1. Create run directory via `new_run(project_root, "name")`
2. Load LLM backend, resolve model, apply adapter or LoRA
3. Set up optimizer via `llm.optimizer_and_params()`
4. Training loop with gradient accumulation, logging, checkpointing
5. Return `RunPaths`

Research and implement:
- **RLHF with PPO** — if not already fully implemented beyond the loss function (check if there's a `train/ppo.py`)
- **Online iterative DPO** — improvements to the existing `online_dpo.py` (e.g., replay buffer, EMA reference model)
- **Self-play fine-tuning (SPIN)** — generate + self-discriminate
- **Constitutional AI / RLAIF** pipeline — LLM-as-judge with constitutional principles
- **Rejection sampling fine-tuning (RSF / Best-of-N)** — sample N, keep best, SFT on it
- **SteerLM** or attribute-conditioned generation
- **Any new training paradigm that became standard post-May 2025**

### C. Synthetic Data Generation (extend `synthetic.py`)

- **Evol-Instruct / WizardLM-style** prompt evolution (complexify, deepen, broaden)
- **Self-Instruct** pipeline with better deduplication
- **Magpie-style** synthetic data from pre-training distributions
- **Persona-driven** generation (diverse synthetic users)
- **Multi-turn** synthetic conversation generation
- **Rejection sampling** in `generate_sft()` — generate N, keep highest-scoring via judge
- **Constitutional filtering** — filter synthetic data through safety/quality rubrics
- **DPO pair generation improvements** — margin-aware selection, diversity filtering

### D. Verifiers (extend `verifiers/`)

Existing: regex, pytest, jsonschema, llm_judge (with judge and thinkprm modes), prime

- **Process Reward Models (PRM)** — step-level reward beyond thinkprm
- **Outcome Reward Models (ORM)** — binary outcome verification
- **Code execution sandbox** verifier — safer than raw pytest
- **Math verification** — symbolic checking for math tasks
- **Factuality verification** — cross-reference with retrieval

### E. Infrastructure Improvements

- **LoRA+ / rsLoRA** — if newer adapter methods exist, add to `train/lora.py`
- **GaLore** or memory-efficient training methods
- **Curriculum learning** — progressive difficulty scheduling
- **Data mixing** — multi-dataset training with configurable ratios
- **Distributed training** — multi-GPU MLX support if available by 2026
- **Quantization-aware fine-tuning** — QLoRA improvements
- **Speculative decoding** for faster synthetic generation

---

## Feb 2026 “Catch‑Up” Checklist (Remaining Gaps)

Prioritize additions below if aiming for “fully up to date” as of Feb 2026.
Already covered in codebase: **SimPO**, **TDPO**, **KTO** (losses + KTO training).

### Preference / Alignment Losses
- **SimPER** (hyperparameter‑free preference optimization) — add loss + tests. Ref: arXiv:2502.00883.
- **Temporal‑decay DPO** (earlier tokens weighted more) — add loss + tests. Ref: arXiv:2502.14340.
- **Token‑importance DPO variants** — add optional loss forms and data hooks:
  - **TIS‑DPO** (token‑level importance sampling) — arXiv:2410.04350.
  - **TI‑DPO** (token‑importance guided) — arXiv:2505.19653.

### RLHF / RFT Pipelines
- **RLOO** (REINFORCE Leave‑One‑Out baseline) — add RFT mode + trainer (Back to Basics, arXiv:2402.14740).
- **REINFORCE++** (global advantage normalization; critic‑free) — add RFT mode + trainer (arXiv:2501.03262).

### Optional (Niche / Emerging)
- **RPG / KL‑regularized policy gradients** for reasoning stability (arXiv:2505.17508).

---

## RLM vs. “Recursive Language Models” (arXiv:2512.24601)

**What the paper proposes**
- **Inference‑time recursion**: treat long prompts as an external environment; model programmatically examines, decomposes, and recursively calls itself over prompt snippets.
- **Goal**: scale to contexts far beyond window size while improving long‑context task quality at comparable cost.

**How MLXSmith RLM differs today**
- **Training loop, not inference recursion**: our RLM is a self‑play training pipeline (generate → rollout → verify → train → eval → gate) rather than an inference strategy.
- **No REPL / snippet environment**: current loop does not provide a programmable environment over the prompt (no recursive sub‑calls or prompt‑slicing API).
- **No long‑context benchmark focus**: the loop optimizes task/verifier reward, but does not directly target long‑context tasks or “context‑length scaling” claims.

**Implications / gap to close**
- Add a **recursive inference wrapper** (REPL or tool API) that can inspect/segment long prompts and call the model on sub‑pieces.
- Add **recursion depth controls** and **cost guards** (stop conditions, budgets).
- Add **long‑context eval suites** for RLM‑style claims.

**Update (v0.1.6)**
- Added a **recursive compaction wrapper** for rollouts and internal /internal/rollout requests (config‑gated).
- Still missing: REPL‑style recursion over prompt snippets, long‑context evaluation suites, and tool‑level prompt inspection APIs.

---

## Code Conventions

When writing new code, follow these conventions observed in the codebase:

1. **Imports**: `from __future__ import annotations` at top, relative imports within package
2. **Config**: Use `ProjectConfig` from `.config`, add new sections as Pydantic models
3. **CLI**: Typer commands with `typer.Option(...)`, `console.print()` for output
4. **Logging**: `console = Console()` from rich, structured JSONL metrics via `write_jsonl()`
5. **Testing**: Mock `_load_llm` and external dependencies, use `tempfile.TemporaryDirectory`
6. **Loss functions**: Use `@register_loss("name")` decorator, operate on `backend` with `_require_mx(backend)` for MLX ops
7. **Type hints**: Use `from __future__ import annotations` for modern syntax (`list[str]` not `List[str]`)
8. **Error handling**: `BackendNotAvailable` for graceful degradation, `RuntimeError` for data issues

---

## Files to Read First

Before making changes, read these files to understand the patterns:

```
src/mlxsmith/cli.py                    # CLI structure and command patterns
src/mlxsmith/synthetic.py              # Newly added, review target
src/mlxsmith/sdk/losses.py             # Loss registry pattern
src/mlxsmith/train/online_dpo.py       # Reference training loop
src/mlxsmith/train/rft.py              # GRPO training loop
src/mlxsmith/train/sft.py              # SFT training loop
src/mlxsmith/llm/backend.py            # LLMBackend protocol
src/mlxsmith/llm/interface.py          # High-level LLM utilities
src/mlxsmith/verifiers/llm_judge.py    # Judge verifier
src/mlxsmith/config.py                 # Config system
tests/test_synthetic.py                # New test patterns
tests/test_online_dpo_self_verify.py   # Existing training tests
pyproject.toml                         # Dependencies
```

---

## Deliverables

1. **Audit report**: List any bugs, edge cases, or inconsistencies found in the refactor
2. **Fixes**: Implement any fixes from the audit
3. **New loss functions**: Add 3-5 new losses to `sdk/losses.py` with tests
4. **New training pipeline**: Add at least 1 new training method to `train/`
5. **Synthetic improvements**: Enhance `synthetic.py` with at least 2 new capabilities
6. **CLI integration**: Wire everything into `cli.py`
7. **Tests**: Full mock-based test coverage for all additions
8. **Verification**: All tests pass, `ruff check src/` clean
