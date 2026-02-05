# Compatibility

This repo targets the **current MLX + mlx-lm ecosystem** and aligns with the
HF → MLX conversion flow used by `mlx_lm.convert`.

## Recommended minimum versions

These match the package constraints in `pyproject.toml`:

- Python: 3.10+
- MLX: `>=0.30.4`
- mlx-lm: `>=0.30.5`
- transformers: `>=5.0.0`
- huggingface_hub: `>=1.3.4`

## HF → MLX conversion

Use the official conversion tool:

```bash
mlx_lm.convert --hf-path <hf_repo_or_local> --mlx-path <mlx_output_dir> [--quantize]
```

This repo’s `mlxsmith pull` wraps that flow and can pass quantization options
(e.g., `--q-bits`, `--q-group-size`, `--q-mode`).

## Known-good model families (targeted)

mlxsmith is designed to work with common decoder-only LMs that
convert via `mlx_lm.convert`, including:

- Qwen3 (verified with mlx-community/Qwen3-4B-Instruct-2507-4bit)
- Qwen3 (verified with Qwen/Qwen3-1.7B-MLX-4bit)
- Llama 3.x
- Mistral

## Known limitations

- Full GPU/Metal acceleration depends on MLX availability on Apple Silicon.
- Speculative decoding and draft models are not wired in mlxsmith yet.
- Strong sandboxing for code execution is out of scope (see `docs/VERIFIERS.md`).

## Verify your install

Use the built-in checker:

```bash
mlxsmith doctor
```

To inspect Python package versions:

```bash
python -m pip show mlx mlx-lm transformers huggingface-hub
```

mlxsmith is tested against modern Python 3.10–3.12 on macOS (Apple Silicon). Linux
is supported for non-MLX workflows (e.g., mock backend, docs, and tooling).

## References

- [MLX install docs](https://ml-explore.github.io/mlx/build/html/install.html)
- [MLX examples (LLMs)](https://github.com/ml-explore/mlx-examples/tree/main/llms)
- [mlx-lm README](https://github.com/ml-explore/mlx-lm/blob/main/README.md)
- [MLX releases](https://github.com/ml-explore/mlx/releases)
- [PyPI: mlx](https://pypi.org/project/mlx/)
- [PyPI: mlx-lm](https://pypi.org/project/mlx-lm/)
- [PyPI: transformers](https://pypi.org/project/transformers/)
- [PyPI: huggingface-hub](https://pypi.org/project/huggingface-hub/)
- [Qwen3 4B model card](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit)
