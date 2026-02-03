# Environment Plugins

Environment plugins package tasks and verifiers into reusable bundles for RFT. This page covers the `mlxsmith env` commands. For the full manifest format and registry structure, see [Environments](../ENVIRONMENTS.md).

## Initialize an environment

```bash
mlxsmith env init myenv
```

This creates `envs/myenv/` with `env.yaml`, a package stub, and metadata.

## List registry entries

```bash
mlxsmith env list
mlxsmith env list myenv --all
```

## Inspect an environment

```bash
mlxsmith env info myenv
mlxsmith env info myenv --version 0.1.0
```

## Install environments

From a directory or package:

```bash
mlxsmith env install path/to/envs/myenv
mlxsmith env install path/to/myenv-0.1.0.tar.gz
```

From the local registry:

```bash
mlxsmith env install myenv
mlxsmith env install myenv --version 0.1.0
mlxsmith env install myenv@0.1.0
```

## Package and publish

```bash
mlxsmith env package myenv
mlxsmith env publish envs/packages/myenv-0.1.0.tar.gz
```

## Pull and run

```bash
mlxsmith env pull myenv --version 0.1.0 --out ./myenv
mlxsmith env run myenv --model runs/sft_0001/adapter
```

## Registry index

```bash
mlxsmith env registry
```
