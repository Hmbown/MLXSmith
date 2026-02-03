# Reinforcement Fine-Tuning (GRPO)

GRPO (Generalized Reward Policy Optimization) trains a model using reward signals from a verifier. For each task, the model generates multiple candidate solutions. A verifier scores each one, and the model is updated with policy gradients that reinforce successful outputs.

## When to use

- You can programmatically verify whether a model's output is correct (tests, regex, schema, LLM judge).
- You want to improve reasoning, code generation, or structured output quality.
- You have an environment defining tasks with verifiable solutions.

## Minimal example

```bash
mlxsmith rft \
  --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --env envs/coding.yaml \
  --verifier verifiers/regex.py
```

## Environment format

A YAML file defining tasks with prompts and optional verifier configuration:

```yaml
name: coding-sample
tasks:
  - id: add
    prompt: |
      Write a Python function `add(a, b)` that returns the sum.
    verifier_kwargs:
      pattern: "def\\s+add\\("
  - id: mul
    prompt: |
      Implement `mul(a, b)` in main.py.
    tests: |
      from main import mul
      def test_mul():
          assert mul(2, 3) == 6
```

## Verifiers

The `--verifier` flag points to a Python file implementing the `verify(prompt, completion, workdir, **kwargs)` function. Built-in verifiers:

- `verifiers/regex.py` — pattern matching
- `verifiers/pytest.py` — sandboxed test execution
- `verifiers/jsonschema.py` — JSON structure validation
- `verifiers/llm_judge.py` — LLM-based scoring

See [Verifiers reference](../VERIFIERS.md) for the full API.

## Loss variants

| Variant | Flag | Description |
|---------|------|-------------|
| GRPO | `--loss-type grpo` | Standard generalized reward policy optimization |
| DR-GRPO | `--loss-type dr_grpo` | Doubly robust GRPO with variance reduction |
| DAPO | `--loss-type dapo` | Decaying advantage policy optimization |

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | Required | Model path or id |
| `--env` | `envs/coding.yaml` | Environment YAML file |
| `--verifier` | `verifiers/regex.py` | Verifier script path |
| `--config`, `-c` | `mlxsmith.yaml` | Config file path |
| `--accel` | From config | Acceleration backend |
| `--rollouts` | From config | Number of rollouts per task |
| `--loss-type` | `grpo` | Loss variant (`grpo`, `dr_grpo`, `dapo`) |
| `--epsilon-low` | From config | Lower clipping bound |
| `--epsilon-high` | From config | Upper clipping bound |
| `--token-level-loss` / `--sequence-level-loss` | From config | Loss granularity |

Key config options under the `rft` section:

| Config Key | Default | Description |
|------------|---------|-------------|
| `rft.rollouts` | `8` | Rollouts per task |
| `rft.loss_type` | `grpo` | Loss variant |
| `rft.kl_coeff` | `0.02` | KL penalty from reference model |
| `rft.temperature` | `0.8` | Sampling temperature for rollouts |
| `rft.max_new_tokens` | `256` | Max tokens per rollout |
| `rft.normalize_advantage` | `true` | Normalize advantages across rollouts |
| `rft.epsilon_low` | `0.2` | Lower PPO clipping bound |
| `rft.epsilon_high` | `0.2` | Upper PPO clipping bound |
| `rft.token_level_loss` | `false` | Use token-level vs. sequence-level loss |

## Output

Training writes to `runs/rft_0001/`:

- `adapter/` — LoRA adapter weights
- `metrics.jsonl` — reward mean/std, pass@1, pass@k, acceptance rate, tokens/sec
- `accepted.jsonl` — verifier-passed trajectories (prompt + completion pairs)
- `config.snapshot.yaml` — exact configuration used

## Typical workflows

### RFT with regex verifier

```bash
mlxsmith rft \
  --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --env envs/coding.yaml \
  --verifier verifiers/regex.py \
  --rollouts 8
```

### RFT with pytest verifier

```bash
mlxsmith rft \
  --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit \
  --env envs/coding.yaml \
  --verifier verifiers/pytest.py
```

### RFT after SFT and preference tuning

```bash
mlxsmith sft --model cache/mlx/mlx-community__Qwen3-4B-Instruct-2507-4bit --data data/sft
mlxsmith pref --model runs/sft_0001/adapter --data data/prefs
mlxsmith rft --model runs/pref_0001/adapter --env envs/coding.yaml --verifier verifiers/pytest.py
```

### Using environment plugins

Environments package tasks and verifiers together:

```bash
mlxsmith env run coding-sample --model runs/sft_0001/adapter
```

See [Environments reference](../ENVIRONMENTS.md).
