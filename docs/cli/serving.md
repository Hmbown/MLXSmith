# Serving

The `serve` command starts an OpenAI-compatible API server for a trained model or adapter. It supports streaming, logprobs, stop sequences, and an optional web UI.

## When to use

- You want to interact with a trained model via HTTP API.
- You need an OpenAI-compatible endpoint for integration with other tools.

## Minimal example

```bash
mlxsmith serve --model runs/sft_0001/adapter --port 8080
```

## API

The server exposes a `/v1/chat/completions` endpoint compatible with the OpenAI API:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64
  }'
```

### Streaming

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64,
    "stream": true
  }'
```

## Qwen formatting cleanup

Some Qwen chat models can emit `<think>...</think>` blocks or special markers like `<|im_end|>` in the raw text output.

Enable output sanitization via config:

```yaml
infer:
  strip_think: true
```

Or via environment variable:

```bash
export MLXSMITH__INFER__STRIP_THINK=1
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | Required | Model path, adapter path, or HF id |
| `--host` | From config | Host to bind (`0.0.0.0` for all interfaces) |
| `--port` | From config | Port number |
| `--ui` | From config | Enable web UI dashboard |
| `--mhc` | From config | Enable experimental mHC adapters (not a speedup) |
| `--mhc-n` | From config | mHC stream expansion rate (n) |
| `--mhc-tmax` | From config | mHC Sinkhorn iterations (tmax) |
| `--config`, `-c` | `mlxsmith.yaml` | Config file path |

Config options under the `serve` section:

| Config Key | Default | Description |
|------------|---------|-------------|
| `serve.host` | `0.0.0.0` | Server host |
| `serve.port` | `8080` | Server port |
| `serve.ui` | `false` | Enable web UI |
| `serve.stream` | `true` | Default streaming behavior |

## Typical workflows

### Serve a base model

```bash
mlxsmith serve --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit --port 8080
```

### Serve a fine-tuned adapter

```bash
mlxsmith serve --model runs/sft_0001/adapter --port 8080
```

### Serve with web UI

```bash
mlxsmith serve --model runs/sft_0001/adapter --port 8080 --ui
```
