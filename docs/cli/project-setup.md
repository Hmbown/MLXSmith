# Project Setup

Project setup covers `mlxsmith init` and `mlxsmith doctor`.

## Initialize a workspace

`mlxsmith init` scaffolds the default project layout, sample verifiers, and a starter config:

```bash
mlxsmith init myproj
cd myproj
```

The command creates `mlxsmith.yaml`, `data/`, `envs/`, `verifiers/`, `runs/`, and other directories used by training and serving.

## Check your environment

`mlxsmith doctor` inspects your system and reports MLX/Metal availability:

```bash
mlxsmith doctor
```

If MLX or Metal are missing, install the Apple Silicon dependencies:

```bash
pip install "mlxsmith[mlx,llm]"
```

## Typical workflow

```bash
mlxsmith init myproj
cd myproj
mlxsmith doctor
mlxsmith pull mlx-community/Qwen3-4B-Instruct-2507-4bit
```
