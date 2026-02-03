# Contributing to MLXSmith

Thanks for helping improve MLXSmith.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

## Running tests and lint

```bash
pytest
ruff check .
```

## Guidelines

- Keep CLI behavior and docs in sync. Update `docs/cli/` when you change commands or defaults.
- Add or update tests where behavior changes.
- Keep examples consistent with the standard model id used in docs.

## Pull requests

- Describe the change and include the motivation.
- Include screenshots or logs when behavior changes in the UI or CLI output.
- Note any breaking changes and migration steps.
