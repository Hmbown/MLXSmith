# Configuration

MLXSmith uses a layered configuration system. Settings can come from a config file, environment variables, or CLI flags. The layers are applied in priority order: CLI flags override environment variables, which override the config file, which overrides built-in defaults.

## Config file

The default config file is `mlxsmith.yaml` in the project root. Create one with defaults:

```bash
mlxsmith config init
```

Supported formats: YAML (default), JSON, TOML.

```bash
mlxsmith config init mlxsmith.json --format json
mlxsmith config init mlxsmith.toml --format toml
```

## Viewing configuration

Show the merged configuration (all sources combined):

```bash
mlxsmith config show
```

Show where each value comes from:

```bash
mlxsmith config show --sources
```

Output as JSON or TOML:

```bash
mlxsmith config show --format json
```

## Validating configuration

```bash
mlxsmith config validate mlxsmith.yaml
```

## Environment variables

Any config option can be set via environment variables using the `MLXSMITH__` prefix with double-underscore separators:

```bash
MLXSMITH__MODEL__ID=mlx-community/Qwen3-4B-Instruct-2507-4bit mlxsmith sft --data data/sft
```

To see all available environment variables:

```bash
mlxsmith config env
```

## Config sections

### model

| Key | Default | Description |
|-----|---------|-------------|
| `id` | `mlx-community/Llama-3.2-3B-Instruct-4bit` | Model path or HF id |
| `backend` | `mlx-lm` | LLM backend |
| `dtype` | `bf16` | Data type override |
| `max_seq_len` | `8192` | Maximum sequence length |
| `quantization` | `none` | Quantization mode (`none`, `q4`, `q6`, `q8`) |
| `trust_remote_code` | `false` | Allow remote code for HF models |
| `use_chat_template` | `true` | Apply model chat template when available |

### accel

| Key | Default | Description |
|-----|---------|-------------|
| `backend` | `none` | Acceleration backend |
| `compile_cache` | `cache/compiled_kernels` | MLX compile cache directory |
| `mhc` | `false` | Enable experimental mHC adapters (not a speedup) |
| `mhc_n` | `4` | mHC stream expansion rate (n) |
| `mhc_tmax` | `20` | mHC Sinkhorn iterations (tmax) |

### train

| Key | Default | Description |
|-----|---------|-------------|
| `seed` | `1337` | Random seed |
| `batch_size` | `1` | Training batch size |
| `grad_accum` | `8` | Gradient accumulation steps |
| `lr` | `2e-4` | Learning rate |
| `weight_decay` | `0.0` | Weight decay |
| `optimizer` | `adamw` | Optimizer (`adamw`, `adam`, `qhadam`, `muon`) |
| `optimizer_kwargs` | `{}` | Extra optimizer arguments |
| `iters` | `1000` | Training iterations |
| `max_grad_norm` | `1.0` | Gradient clipping |
| `save_every` | `100` | Checkpoint save interval |
| `eval_every` | `100` | Evaluation interval |
| `log_every` | `10` | Metrics logging interval |
| `train_on_prompt` | `false` | Train on prompt tokens as well as response |

### lora

| Key | Default | Description |
|-----|---------|-------------|
| `r` | `16` | LoRA rank |
| `alpha` | `32` | LoRA alpha |
| `dropout` | `0.05` | LoRA dropout |
| `target_modules` | `["q_proj", "v_proj", "o_proj"]` | Modules to apply LoRA to |
| `num_layers` | `0` | Number of layers (0 = all) |
| `scale` | None | Optional LoRA scale (overrides alpha/r) |
| `fine_tune_type` | `lora` | Fine-tune type (`lora`, `dora`, `full`) |

### pref

| Key | Default | Description |
|-----|---------|-------------|
| `algo` | `dpo` | Preference algorithm |
| `loss_type` | `dpo` | Loss function |
| `beta` | `0.1` | Preference loss scale / inverse temperature |
| `kl_coeff` | `0.0` | KL penalty coefficient |
| `delta` | `0.0` | Margin |
| `reference_model` | None | Optional reference model id/path |

### kto

| Key | Default | Description |
|-----|---------|-------------|
| `beta` | `0.1` | Loss scale / inverse temperature (KTO) |
| `gain_power` | `1.0` | Gain exponent |
| `loss_power` | `1.0` | Loss exponent |
| `loss_aversion` | `1.0` | Loss aversion coefficient |
| `reference_point` | `0.0` | Prospect theory reference point |
| `reference_model` | None | Optional reference model id/path |

### rft

| Key | Default | Description |
|-----|---------|-------------|
| `algo` | `grpo` | RFT algorithm |
| `loss_type` | `grpo` | Loss variant (`grpo`, `dr_grpo`, `dapo`) |
| `rollouts` | `8` | Rollouts per task |
| `kl_coeff` | `0.02` | KL penalty coefficient |
| `max_steps_per_task` | `1` | Max steps per task |
| `temperature` | `0.8` | Sampling temperature |
| `max_new_tokens` | `256` | Max tokens per rollout |
| `normalize_advantage` | `true` | Normalize advantages across rollouts |
| `epsilon_low` | `0.2` | Lower PPO clipping bound |
| `epsilon_high` | `0.2` | Upper PPO clipping bound |
| `token_level_loss` | `false` | Token-level vs. sequence-level loss |
| `reference_model` | None | Optional reference model id/path |

### infer

| Key | Default | Description |
|-----|---------|-------------|
| `max_new_tokens` | `256` | Max tokens for generation |
| `temperature` | `0.7` | Sampling temperature |
| `top_p` | `1.0` | Top-p sampling |
| `top_k` | None | Top-k sampling |
| `strip_think` | `false` | Strip Qwen-style `<think>...</think>` blocks and chat template markers (e.g. `<|im_end|>`) from outputs |

### serve

| Key | Default | Description |
|-----|---------|-------------|
| `host` | `0.0.0.0` | Server host |
| `port` | `8080` | Server port |
| `ui` | `false` | Enable web UI |
| `stream` | `true` | Default streaming |

### rlm

| Key | Default | Description |
|-----|---------|-------------|
| `iterations` | `50` | Number of RLM iterations |
| `rollouts_per_task` | `8` | Solutions per task |
| `corpus_max` | `8000` | Max corpus size |
| `mix_old_ratio` | `0.4` | Old data mixing fraction |
| `hard_ratio` | `0.6` | Hard sample weight |
| `gating` | `strict` | Gating strategy |

### logging

| Key | Default | Description |
|-----|---------|-------------|
| `level` | `INFO` | Log level |
| `file` | None | Log file path |
| `format` | `%(asctime)s - %(name)s - %(levelname)s - %(message)s` | Log format |

## Priority order

1. **CLI flags** (highest)
2. **Environment variables** (`MLXSMITH__SECTION__KEY`)
3. **Config file** (`mlxsmith.yaml`)
4. **Built-in defaults** (lowest)
