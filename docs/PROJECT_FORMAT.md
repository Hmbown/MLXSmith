# Project Layout & Run Artifacts

`mlxsmith init <project>` creates a self-contained workspace:

```
myproj/
  mlxsmith.yaml
  data/
    sft/
    prefs/
  models/
  envs/
    <env-name>/
      env.yaml
    registry.json
    registry/
  verifiers/
  eval/
    suites/
  runs/
  cache/
  bench/
```

## Run directory structure

Each training command writes a run folder under `runs/`:

```
runs/
  sft_0001/
    adapter/
      adapters.safetensors
      adapter_config.json
      adapter_metadata.json
    artifacts/
    checkpoints/
    logs/
    metrics.jsonl
    config.snapshot.yaml

  pref_0001/
    ...

  rft_0001/
    adapter/
    artifacts/
    accepted.jsonl
    metrics.jsonl
    config.snapshot.yaml

  distill_0001/
    artifacts/
      distill_data/
        train.jsonl
    metrics.jsonl
    config.snapshot.yaml

  rlm_0001/
    adapter/
    artifacts/
    metrics.jsonl
    config.snapshot.yaml

  rlm_weights/
    infer.json
    train.json
```

### Key artifacts

- `adapter/` — LoRA adapter weights + config (MLX-LM compatible).
- `metrics.jsonl` — structured logs for loss/reward/throughput.
- `config.snapshot.yaml` — exact run config captured at start.
- `accepted.jsonl` (RFT only) — verifier-passed trajectories.
- `rlm_weights/` — weight pointers for inference/trainer staleness control.

## Bench reports

`mlxsmith bench` writes reports to `bench/bench_<timestamp>.json` with per-rep
tokens/sec (inference/end-to-end) or steps/sec (trainer) stats.
