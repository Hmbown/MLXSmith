# Verifiers

mlxsmith verifiers implement a **strict, deterministic interface** used by preference tuning, RFT/GRPO, and eval.

## Verifier API

```python
from mlxsmith.verifiers.types import VerifyResult

def verify(prompt: str, completion: str, workdir: str, **kwargs) -> VerifyResult:
    ...
```

Return fields:

- `reward` (float): reward assigned to the completion.
- `passed` (bool): whether the completion passed the check.
- `info` (dict): diagnostic info (e.g., error messages, match data).
- `artifacts_dir` (str | None): directory containing artifacts for this run.

## Built-in verifiers

- `regex.py`: regex match on completion text.
- `jsonschema.py`: parse JSON + validate against a JSON Schema.
- `pytest_verifier.py`: run pytest in a per-rollout sandbox directory.
- `docker_verifier.py`: run pytest inside a locked-down Docker container.
- `compose.py`: compose multiple verifiers (AND/OR/weighted).
- `llm_judge.py`: LLM-based verifier (self-verification / ThinkPRM-style).

## Sandbox behavior

`pytest_verifier.py` runs tests with:

- **Per-rollout workdir** (unique directory for each rollout).
- **Hard timeout** (default 30s).
- **Captured stdout/stderr** (truncated in output).
- **Sanitized env** (`HOME`, `TMPDIR`, `PYTHONPATH` set to workdir).

This is intentionally lightweight and deterministic, **not a hardened security sandbox**. Avoid untrusted code; use containerized isolation if you need stronger guarantees.

`docker_verifier.py` executes tests in a container with:

- `--network none`
- read-only root filesystem
- `/tmp` mounted as `tmpfs` with `noexec`
- memory/CPU/PID limits
- optional fallback to local pytest sandbox

## Composing verifiers

`compose.py` lets you combine multiple verifiers:

```python
from mlxsmith.verifiers.compose import verify as _verify

def verify(prompt: str, completion: str, workdir: str, **kwargs):
    return _verify(prompt, completion, workdir, **kwargs)
```

Example `verifier_kwargs`:

```yaml
verifier_kwargs:
  mode: all
  verifiers:
    - path: verifiers/regex.py
      kwargs:
        pattern: "def\\s+solve\\("
    - path: verifiers/pytest.py
```

### LLM judge verifier

Use `llm_judge.py` to score completions with a judge model. Example task:

```yaml
verifier_kwargs:
  model: mlx-community/Qwen2.5-3B-Instruct-4bit
  mode: thinkprm
  rubric: "@verifiers/rubrics/coding.txt"
  min_score: 0.6
```

You can also set `MLXSMITH_JUDGE_MODEL` to provide the default judge model id.

The composed verifier reports per-verifier latency in `info.verifier_latencies_ms`.

## Writing a custom verifier

```python
from mlxsmith.verifiers.types import VerifyResult


def verify(prompt: str, completion: str, workdir: str, **kwargs):
    # ... run checks ...
    return VerifyResult(
        reward=1.0,
        passed=True,
        info={"detail": "ok"},
        artifacts_dir=workdir,
    )
```
