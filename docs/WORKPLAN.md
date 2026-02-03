# MLXSMITH Workplan (Parity + Apps)

Date: 2026-02-02

This document is a **handoff guide** for any AI or engineer to complete
PrimeIntellect/Tinker parity and to ship both a web app + a SwiftUI macOS app.

> Note: the codebase uses the `mlxsmith` module naming.
> Paths and references have been updated to **mlxsmith**.

---

## 1) PrimeIntellect / Tinker Parity — What’s Done vs Missing

### Implemented (core parity)
- RLM loop: generate → rollout → verify → train → eval → gate (`src/mlxsmith/rlm/*`).
- Task mutation + quality filtering (similarity, length, blocklist) (`rlm/mutate.py`, `rlm/generate.py`).
- Verifier backends: regex/jsonschema/pytest + Docker sandbox (`verifiers/*`).
- Verifier composition + latency logging (`verifiers/compose.py`).
- Loss registry: cross_entropy, DPO/ORPO + RL losses (IS/PPO/CISPO/DRO).
- Internal rollout endpoint returning tokens + logprobs (`server.py`).
- Adapter hot‑reload endpoint (`server.py`).
- Distillation: offline + OPD (teacher logprobs + reverse‑KL advantages) (`train/distill.py`).
- Bench modes: inference, trainer, end‑to‑end (`bench.py`).
- Serve UI + RLM monitor endpoints (`server.py`).
- Environment plugin system + local registry (`envs/system.py`).
- Orchestrated mode: queue‑driven inference + trainer workers (`rlm/loop.py`, `orchestrator/*`).
- Weight pointers for inference/trainer staleness (`rlm/weights.py`).
- Config precedence (CLI > config > env) via `pydantic-settings` + `MLXSMITH__` env prefix.
- HF auth helpers: `mlxsmith auth login/status/logout`.

### Remaining parity gaps (highest priority)

#### A) Orchestrator / Inference / Trainer **multi‑process split**
**Status:** ✅ Implemented (queue‑driven rollouts + trainer worker + hot reload).

#### C) Environment Hub parity
**Gap:** environments are local; PRIME uses installable packages + hub.

**Do:**
- `mlxsmith env init` should scaffold a Python package with `pyproject.toml`.
- `mlxsmith env install/publish` should use a registry index (local or remote).
- Support `env list`, `env info`, `env pull`, and version pinning.

**Targets:**
- `src/mlxsmith/envs/system.py` (expand)
- Optional `src/mlxsmith/envs/hub.py`
- Docs + CLI

**Status:** `env init` scaffolds a Python package; local registry supports `env list/info/pull` and version pinning (registry-only).

#### D) Tinker API parity (futures + logprobs)
**Gap:** threadpool futures exist, but no TrainingClient‑style API or prompt
logprobs top‑k for distillation.

**Do:**
- Implement a `TrainingClient` abstraction with `forward_backward`, `optim_step`,
  `save_state`, `load_state` returning an APIFuture‑like object.
- Add **top‑k prompt logprobs** to sampling client, as per Tinker docs.

**Targets:**
- `src/mlxsmith/sdk/*` (new class)
- `src/mlxsmith/llm/*` (support top‑k)

#### E) OPD (on‑policy distillation) — faithful version
**Status:** ✅ Implemented (teacher logprobs + reverse‑KL advantages + IS loss).

#### F) Token‑level RL environments
**Status:** ✅ Implemented (token_env interface + tasks shim in RFT).

**Do:**
- Add an optional token‑level env interface for RL tasks.
- Provide a shim that wraps string‑based tasks.

**Targets:**
- `src/mlxsmith/envs/*` (new token‑env interface)
- `src/mlxsmith/rlm/*` (optional usage)

---

## 2) SwiftUI + Web App — How to Build Both in Parallel

### Core idea
Make **one shared backend API** (mlxsmith server). Both apps are just clients.

- Web app: React/Next.js (or Svelte) talking to `http://localhost:PORT`.
- SwiftUI app: native UI + Keychain, talking to the same API.
- Keep the API stable with an OpenAPI spec to generate clients for both.

### Shared API contract (proposed)
Add a simple OpenAPI spec in `docs/api/openapi.yaml` for:
- `/v1/chat/completions` (OpenAI compatible)
- `/internal/rollout` (tokens + logprobs)
- `/internal/adapter/reload`
- `/internal/rlm/state` + `/internal/rlm/history`
- `/internal/models/list` (cached MLX models)
- `/internal/models/pull` (proxy `mlxsmith pull`)
- `/internal/hf/token` (store token securely on host)

### Web app scope
- Model list + pull (progress/logs)
- Serve control + chat UI (streaming)
- RLM monitor (charts + history)
- Environment registry view + run

### SwiftUI macOS scope
- Keychain‑backed HF token management
- Model pull manager (uses CLI or API)
- Serve control + streaming chat
- RLM monitor (native charts or embedded web)
- Settings for project root + cache path

### Practical way to build both simultaneously
1) **Define API contract first** (OpenAPI spec).
2) Build web UI against that spec.
3) Use codegen or lightweight client in SwiftUI to hit same endpoints.
4) For features missing in the API, add **internal endpoints** instead of
   duplicating logic in Swift.

### Suggested structure
```text
apps/
  web/          # Next.js or Svelte
  macos/        # SwiftUI
server/
  api/          # OpenAPI + handlers (mlxsmith)
```

### Definition of done (apps)
- Both apps can start/stop serve.
- Both can pull models.
- Both can show RLM history/state.
- HF token is stored securely (Keychain for macOS; backend storage for web).

---

## 3) “Done” checklist for parity

- [ ] Multi‑process orchestrator/inference/trainer running concurrently.
- [ ] Weight updates applied without inference restart.
- [ ] Config precedence + env var support (pydantic‑settings).
- [ ] Environment hub workflows parity.
- [ ] TrainingClient futures parity with Tinker API.
- [ ] OPD with teacher logprobs + reverse‑KL advantage.
- [ ] Token‑level RL env support.
- [ ] OpenAPI spec published; web + SwiftUI apps consuming it.

---

## 4) Quick references
- Prime‑RL entrypoints: https://docs.primeintellect.ai/prime-rl/entrypoints
- Prime environments: https://docs.primeintellect.ai/tutorials-environments
- Tinker API: https://tinker-docs.thinkingmachines.ai/
- OPD blog: https://thinkingmachines.ai/blog/on-policy-distillation/

---

## 6) Branding & Legal (Apple‑like UI)

You can build a native, Apple‑style SwiftUI app as long as you avoid Apple
trademarks and proprietary assets.

**Allowed**
- Use SwiftUI/AppKit components and standard macOS patterns.
- Use SF Pro and SF Symbols (on Apple platforms) per Apple’s license.
- Follow Apple’s Human Interface Guidelines (HIG).

**Avoid**
- Apple logos, official product names, or implied endorsement.
- Copying Apple screenshots or proprietary artwork.
- Using Apple assets outside permitted license terms.

Add a short attribution note in the app (e.g., “Not affiliated with Apple”) if
you want extra clarity.

---

## 7) Full Vision (End‑to‑End)

**One product, three surfaces**
- **CLI (mlxsmith):** automation + reproducible runs.
- **API server:** shared contract for all clients.
- **Clients:** SwiftUI macOS app (primary) + web app (secondary/portable).

**Golden path**
1) Install MLX + mlxsmith CLI.
2) Authenticate HF token.
3) Pull/convert model → run SFT or RLM.
4) Monitor training + gating outcomes.
5) Serve model locally with streaming chat.

**Client strategy**
- The SwiftUI app is the “Apple‑like” primary UX.
- The web app mirrors the same API for portability and demos.
- Both rely on the same OpenAPI contract to avoid logic drift.

**Reliability + observability**
- Consistent run artifacts and config snapshots.
- Benchmarks and trend history are first‑class.
- Verifier composition and latency diagnostics are visible.

**Scaling path**
- Local single‑machine first.
- Add multi‑process orchestrator + async inference/trainer workers.
- Optional remote inference workers later via the same API.

## 6) Branding & Legal (Apple‑like UI)

You can build a native, Apple‑style SwiftUI app as long as you avoid Apple
trademarks and proprietary assets.

**Allowed**
- Use SwiftUI/AppKit components and standard macOS patterns.
- Use SF Pro and SF Symbols (on Apple platforms) per Apple’s license.
- Follow Apple’s Human Interface Guidelines (HIG).

**Avoid**
- Apple logos, official product names, or implied endorsement.
- Copying Apple screenshots or proprietary artwork.
- Using Apple assets outside permitted license terms.

Add a short attribution note in the app (e.g., “Not affiliated with Apple”) if
you want extra clarity.

---

## 7) Full Vision (End‑to‑End)

**One product, three surfaces**
- **CLI (mlxsmith):** automation + reproducible runs.
- **API server:** shared contract for all clients.
- **Clients:** SwiftUI macOS app (primary) + web app (secondary/portable).

**Golden path**
1) Install MLX + mlxsmith CLI.
2) Authenticate HF token.
3) Pull/convert model → run SFT or RLM.
4) Monitor training + gating outcomes.
5) Serve model locally with streaming chat.

**Client strategy**
- The SwiftUI app is the “Apple‑like” primary UX.
- The web app mirrors the same API for portability and demos.
- Both rely on the same OpenAPI contract to avoid logic drift.

**Reliability + observability**
- Consistent run artifacts and config snapshots.
- Benchmarks and trend history are first‑class.
- Verifier composition and latency diagnostics are visible.

**Scaling path**
- Local single‑machine first.
- Add multi‑process orchestrator + async inference/trainer workers.
- Optional remote inference workers later via the same API.


## 5) Suggested next actions
1) Create OpenAPI spec + minimal handlers for models/token management.
2) Split RLM into daemon orchestrator + inference server + trainer worker.
3) Implement TrainingClient + top‑k logprob support.
4) Scaffold web app + SwiftUI app against same API.
