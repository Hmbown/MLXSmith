from __future__ import annotations

import importlib
import importlib.util
import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol


@dataclass
class TokenEnvStep:
    observation: list[int]
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class TokenEnv(Protocol):
    def initial_observation(self) -> list[int] | TokenEnvStep:
        ...

    def step(self, action: int) -> TokenEnvStep:
        ...


@dataclass
class TokenEnvSpec:
    factory: Callable[..., TokenEnv]
    kwargs: dict[str, Any] = field(default_factory=dict)
    kind: str = "custom"


def _filter_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs
    for param in sig.parameters.values():
        if param.kind == param.VAR_KEYWORD:
            return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _load_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load token env module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def _resolve_factory(module, class_name: Optional[str]) -> Callable[..., TokenEnv]:
    if class_name:
        factory = getattr(module, class_name, None)
        if factory is None:
            raise RuntimeError(f"Token env factory not found: {class_name}")
        if not callable(factory):
            raise RuntimeError(f"Token env factory not callable: {class_name}")
        return factory

    for fallback in ("Env", "TokenEnv", "make_env", "load_env"):
        factory = getattr(module, fallback, None)
        if callable(factory):
            return factory

    raise RuntimeError("Token env factory not found (expected class or make_env/load_env).")


def _parse_token_env_spec(project_root: Path, token_env: Any) -> TokenEnvSpec:
    if isinstance(token_env, str):
        if token_env in {"tasks", "task_shim"}:
            return TokenEnvSpec(factory=StringTaskTokenEnv, kind="tasks")
        path_part, _, class_part = token_env.partition(":")
        class_name = class_part or None
        if Path(path_part).suffix == ".py" or Path(path_part).exists():
            path = Path(path_part)
            if not path.is_absolute():
                path = project_root / path
            module = _load_from_path(path)
        else:
            module = importlib.import_module(path_part)
        factory = _resolve_factory(module, class_name)
        return TokenEnvSpec(factory=factory, kind="custom")

    if isinstance(token_env, dict):
        if token_env.get("type") in {"tasks", "task_shim"}:
            return TokenEnvSpec(factory=StringTaskTokenEnv, kind="tasks")
        class_name = token_env.get("class") or token_env.get("cls")
        kwargs = token_env.get("kwargs") or {}
        if "path" in token_env:
            path = Path(token_env["path"])
            if not path.is_absolute():
                path = project_root / path
            module = _load_from_path(path)
        elif "module" in token_env:
            module = importlib.import_module(str(token_env["module"]))
        else:
            raise RuntimeError("token_env requires 'path' or 'module'")
        factory = _resolve_factory(module, class_name)
        return TokenEnvSpec(factory=factory, kwargs=dict(kwargs), kind="custom")

    raise RuntimeError("token_env must be a string or mapping")


def load_token_env_spec(project_root: Path, env_data: dict) -> Optional[TokenEnvSpec]:
    token_env = env_data.get("token_env")
    if not token_env:
        return None
    return _parse_token_env_spec(project_root, token_env)


def create_token_env(spec: TokenEnvSpec, **kwargs) -> TokenEnv:
    params = dict(spec.kwargs)
    params.update(kwargs)
    params = _filter_kwargs(spec.factory, params)
    return spec.factory(**params)


class StringTaskTokenEnv:
    def __init__(
        self,
        *,
        prompt: str,
        tests: str,
        verifier_fn: Callable[..., Any],
        workdir: Path,
        max_steps: int,
        encode: Callable[[str], list[int]],
        decode: Callable[[list[int]], str],
        verifier_kwargs: Optional[dict[str, Any]] = None,
        eos_token_id: Optional[int] = None,
    ):
        self.prompt = prompt
        self.tests = tests
        self.verifier_fn = verifier_fn
        self.workdir = Path(workdir)
        self.max_steps = max_steps
        self.encode = encode
        self.decode = decode
        self.verifier_kwargs = verifier_kwargs or {}
        self.eos_token_id = eos_token_id
        self._prompt_ids: list[int] = []
        self._generated: list[int] = []
        self._steps = 0

    def initial_observation(self) -> list[int]:
        self._prompt_ids = list(self.encode(self.prompt))
        self._generated = []
        self._steps = 0
        tests_dir = self.workdir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        (tests_dir / "test_task.py").write_text(self.tests or "", encoding="utf-8")
        return list(self._prompt_ids)

    def step(self, action: int) -> TokenEnvStep:
        if self.eos_token_id is not None and action == self.eos_token_id:
            done = True
        else:
            done = False
        self._generated.append(int(action))
        self._steps += 1

        if self._steps >= self.max_steps:
            done = True

        reward = 0.0
        info: dict[str, Any] = {}
        if done:
            completion_ids = list(self._generated)
            if self.eos_token_id is not None and completion_ids and completion_ids[-1] == self.eos_token_id:
                completion_ids = completion_ids[:-1]
            completion = self.decode(completion_ids)
            (self.workdir / "main.py").write_text(completion, encoding="utf-8")
            t0 = time.time()
            res = self.verifier_fn(self.prompt, completion, str(self.workdir), **self.verifier_kwargs)
            latency_ms = (time.time() - t0) * 1000.0
            reward = float(getattr(res, "reward", 0.0))
            info = dict(getattr(res, "info", {}) or {})
            info["passed"] = bool(getattr(res, "passed", False))
            info["verifier_latency_ms"] = latency_ms

        observation = list(self._prompt_ids) + list(self._generated)
        return TokenEnvStep(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )
