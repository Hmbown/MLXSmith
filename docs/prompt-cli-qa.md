# Prompt: CLI QA (mlxsmith)

Use this prompt to ask another AI to run a full CLI QA pass.

```text
You are a QA lead with many sub‑agents. Validate the mlxsmith CLI end‑to‑end.
Spin up multiple agents for: install/venv, unit tests, CLI smoke, serve,
bench, rlm loop, env plugins, and docs consistency.

Repo context:
- Python package in `src/`
- CLI entrypoint: `mlxsmith.cli:app`
- Tests: `tests/`
- Docs: `docs/WORKPLAN.md`, `docs/ROADMAP.md`

Required checks (run or simulate):
1) Install and import
   - `python3 -m venv .venv && source .venv/bin/activate`
   - `pip install -e ".[dev,serve]"`
   - `python -c "import mlxsmith"`
2) Unit tests
   - `PYTHONPATH=src python3 -m pytest -q`
3) CLI smoke
   - `mlxsmith init /tmp/mlxsmith_demo`
   - `cd /tmp/mlxsmith_demo`
   - `mlxsmith doctor`
   - `mlxsmith data split --in <small.jsonl> --out-dir data/sft`
4) Serve + UI
   - `mlxsmith serve --model dummy/model --port 8080`
   - curl `/health` and `/v1/chat/completions`
5) RLM loop smoke (mock backend)
   - run a 1‑iteration loop and check `runs/rlm_state.json`
6) Env plugins
   - `mlxsmith env init myenv`, `mlxsmith env package myenv`, `mlxsmith env publish ...`

Report:
- A table of PASS/FAIL with logs.
- Any fixes needed, with file paths.
- Confirm if CLI is “done” or still missing parity items.
```
```
