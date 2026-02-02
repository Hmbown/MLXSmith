# Prime Intellect Design Notes

Date: 2026-02-02

These notes summarize how MLXSMITH mirrors the Prime Intellect / Tinker
architecture and where the remaining gaps still live. For source links, see
`docs/UPSTREAM_NOTES.md`.

## Alignment (current)
- Orchestrator/inference/trainer split (in‑process today, multi‑process planned).
- Verifier‑driven RL with Docker sandbox and composition.
- Environment‑driven tasks with a local registry workflow.
- Loss registry including cross‑entropy + RL losses (IS/PPO/CISPO/DRO).
- Internal rollout API returns tokens + logprobs.
- Weight pointers for inference/trainer staleness.

## Known gaps vs Prime/Tinker
- Multi‑process orchestration and async worker queues.
- PRIME‑style config precedence (CLI > config > env vars).
- Environment hub parity (package publishing + discovery).
- Tinker‑style TrainingClient with APIFutures + top‑k prompt logprobs.
- Faithful OPD (teacher logprobs + reverse‑KL advantages).
- Token‑level RL environments.

## Design decisions
- Keep MLX core in Python; use SwiftUI + web as first‑class clients.
- Prefer stable, explicit internal endpoints over UI‑side logic.
- Use OpenAPI as a shared contract for web + macOS apps.

## Next steps
- Implement multi‑process orchestrator and inference reload endpoints.
- Finalize OpenAPI spec and use it to build both clients.
- Expand environment tooling to mirror Prime Hub workflows.

