# Model Management

Model management commands handle model download/conversion, quantization stubs, and adapter merging.

## Pull a model

Download a Hugging Face model and convert it to MLX format (stored under `cache/mlx/` by default):

```bash
mlxsmith pull mlx-community/Qwen3-4B-Instruct-2507-4bit
```

Common options:

| Option | Default | Description |
|--------|---------|-------------|
| `--out` | None | Output path (defaults to `cache/mlx/<model>`) |
| `--no-convert` | `false` | Only download HF snapshot, skip MLX conversion |
| `--quantize` | `false` | Quantize during conversion |
| `--q-bits` | None | Quantization bits (e.g., 4 or 8) |
| `--q-group-size` | None | Quantization group size |
| `--q-mode` | None | Quantization mode override |
| `--quant-predicate` | None | Predicate for selective quantization |
| `--trust-remote-code` | `false` | Allow custom model code |

## Quantization stub

`mlxsmith quantize` creates a placeholder marker for legacy workflows. Use `mlxsmith pull --quantize` for real quantization.

```bash
mlxsmith quantize cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit --to q4 --out models/quantized
```

## Merge adapters

Merge multiple LoRA adapters into a single adapter directory:

```bash
mlxsmith adapters merge \
  --base cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --adapters runs/sft_0001/adapter,runs/pref_0001/adapter \
  --out models/merged_adapter
```

You can optionally pass weights to control blending:

```bash
mlxsmith adapters merge \
  --base cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --adapters runs/sft_0001/adapter,runs/pref_0001/adapter \
  --weights 0.7,0.3 \
  --out models/merged_adapter
```
