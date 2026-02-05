# Troubleshooting

Common issues and quick fixes when running MLXSmith.

## MLX not detected

If `mlxsmith doctor` shows `mlx: False` or `metal: False`:

- Install MLX dependencies: `pip install "mlxsmith[mlx,llm]"`.
- Ensure you are on Apple Silicon and running a native Python build.
- Re-run `mlxsmith doctor` after reinstalling.

## Model conversion fails

If `mlxsmith pull` fails during conversion:

- Try `mlxsmith pull <model> --no-convert` to confirm the HF snapshot downloads.
- Upgrade `mlx-lm` to the version in `docs/COMPATIBILITY.md`.
- Retry with `--trust-remote-code` for models that require custom code.
- Check disk space under `cache/`.

## Qwen `<think>` or `<|im_end|>` in outputs

Some Qwen chat models can emit reasoning blocks (`<think>...</think>`) or special markers like `<|im_end|>`.

Enable sanitization:

- Config: set `infer.strip_think: true`
- Env var: `export MLXSMITH__INFER__STRIP_THINK=1`

## Out of memory (unified memory pressure)

- Reduce `train.batch_size` and increase `train.grad_accum`.
- Lower `model.max_seq_len` or `infer.max_new_tokens`.
- Use a quantized model (`mlxsmith pull --quantize --q-bits 4`).
- Reduce RFT rollouts or max tokens per rollout.

## Slow throughput

- Use a quantized model to increase tokens/sec.
- Reduce `max_new_tokens` and temperature where possible.
- Close other GPU/Metal-heavy applications.
- Check `mlxsmith accel status` for backend availability.

## Verifier failures

- Run the verifier script directly to confirm it passes on expected outputs.
- Confirm the environment YAML points to the correct verifier path.
- Increase `rlm.verifier_timeout_s` for slow tests.
- For containerized tests, switch to the Docker verifier backend.

## Judge model requirements

Online DPO, self-verify, and synthetic filtering require a judge model:

- Pass `--judge-model` (or `--verifier-model` for self-verify), or
- Set `MLXSMITH_JUDGE_MODEL` in the environment.

## Where outputs live

Training outputs are under `runs/<kind>_0001/` and include:

- `adapter/` — LoRA adapter weights
- `metrics.jsonl` — training metrics
- `config.snapshot.yaml` — exact config used
