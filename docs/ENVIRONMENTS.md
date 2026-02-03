# Environments

mlxsmith environments package tasks, verifiers, and metadata into a reusable unit.
They live under `envs/` and can be packaged/published to a local registry.

Legacy single-file envs (e.g., `envs/coding.yaml`) still work with `mlxsmith rft`.

## Layout

```text
envs/
  <name>/
    env.yaml
    ... optional assets ...
```

`env.yaml` is the manifest:

```yaml
name: coding-sample
version: 0.1.0
description: Sample environment
verifier: verifiers/regex.py
tasks:
  - id: add
    prompt: |
      Write a Python function add(a, b) that returns the sum.
    tests: |
      from main import add
      def test_add():
          assert add(2, 3) == 5
```

### Token-level RL environments (optional)

You can provide a token-level env that implements `initial_observation()` and
`step(token_id)` for RL-style tasks:

```yaml
token_env:
  path: envs/myenv/token_env.py
  class: MyTokenEnv
  kwargs:
    max_steps: 128
```

`token_env` also supports a tasks shim if you want to run token-level rollouts
against string-based tasks:

```yaml
token_env: tasks
```

## CLI workflow

Initialize an environment (scaffolds `pyproject.toml` + package stub):

```bash
mlxsmith env init myenv
```

List registry entries:

```bash
mlxsmith env list
mlxsmith env list myenv --all
```

Inspect a registry entry:

```bash
mlxsmith env info myenv
mlxsmith env info myenv --version 0.1.0
```

Install from a directory or package:

```bash
mlxsmith env install path/to/envs/myenv
mlxsmith env install path/to/myenv-0.1.0.tar.gz
```

Install from registry (latest or pinned):

```bash
mlxsmith env install myenv
mlxsmith env install myenv --version 0.1.0
mlxsmith env install myenv@0.1.0
```

Package and publish to the local registry:

```bash
mlxsmith env package myenv
mlxsmith env publish envs/packages/myenv-0.1.0.tar.gz
```

Pull env source from the registry:

```bash
mlxsmith env pull myenv
mlxsmith env pull myenv --version 0.1.0 --out ./myenv
```

Run a packaged env (invokes `mlxsmith rft` under the hood):

```bash
mlxsmith env run myenv --model runs/sft_0001/adapter
```

Inspect the registry index:

```bash
mlxsmith env registry
```

## Local registry

The local registry index is stored at:

```text
envs/registry.json
```

Published packages are copied into `envs/registry/` and indexed by name/version.

## Notes

- Environments are local-first; publishing only updates the local registry index.
- For stronger isolation during verifier execution, use the Docker verifier backend.
