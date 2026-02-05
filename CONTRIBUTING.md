# Contributing to MLXSmith

Thank you for your interest in improving MLXSmith. This document covers the development workflow and expectations for contributions.

## Development Setup

```bash
git clone https://github.com/Hmbown/MLXSmith.git
cd MLXSmith
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run the full test suite
pytest

# Run a specific test file
pytest tests/test_config.py

# Run with verbose output
pytest -v
```

## Linting

MLXSmith uses [Ruff](https://docs.astral.sh/ruff/) for linting with a 100-character line width.

```bash
ruff check src tests
```

## Code Style

- Use `from __future__ import annotations` for modern type hint syntax.
- Use relative imports within the `mlxsmith` package.
- Add new configuration sections as Pydantic models in `config_models.py`.
- Register new loss functions with the `@register_loss` decorator in `sdk/losses.py`.
- Use `console.print()` (Rich) for CLI output. Write structured metrics as JSONL.

## Project Structure

- `src/mlxsmith/` — Main package source.
- `tests/` — Test suite (pytest). Use mocks for MLX and external backends.
- `docs/` — Documentation. CLI guides live under `docs/cli/`.
- `examples/` — Example configuration files.
- `scripts/` — Experiment and data-generation scripts.
- `apps/` — Web dashboard (Next.js) and native macOS app (SwiftUI).

## Pull Requests

1. Create a feature branch from `main`.
2. Keep changes focused — one feature or fix per PR.
3. Update documentation in `docs/cli/` when CLI behavior changes.
4. Add or update tests for any behavior change.
5. Ensure `pytest` and `ruff check src tests` pass before submitting.
6. Write a clear description with motivation, and note any breaking changes.
