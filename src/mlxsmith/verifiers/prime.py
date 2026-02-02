from __future__ import annotations

import importlib.util
import re
from typing import Any, Dict, List, Optional

from .types import VerifyResult

_STATE: Dict[str, Dict[str, float]] = {
    "values": {},
    "counts": {},
}


def _load_verifier(path: str):
    import sys
    from pathlib import Path as _Path

    verifier_path = _Path(path).resolve()

    # If the file lives inside a Python package, set __package__ so that
    # relative imports (e.g. ``from .types import ...``) work correctly.
    pkg_name: Optional[str] = None
    if (verifier_path.parent / "__init__.py").exists():
        parts: list[str] = []
        p = verifier_path.parent
        while (p / "__init__.py").exists():
            parts.insert(0, p.name)
            p = p.parent
        pkg_name = ".".join(parts)
        root = str(p)
        if root not in sys.path:
            sys.path.insert(0, root)

    mod_name = f"{pkg_name}._prime_loaded" if pkg_name else "prime_verifier"
    spec = importlib.util.spec_from_file_location(mod_name, str(verifier_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load verifier: {verifier_path}")
    module = importlib.util.module_from_spec(spec)
    if pkg_name is not None:
        module.__package__ = pkg_name
    spec.loader.exec_module(module)  # type: ignore
    verify_fn = getattr(module, "verify", None)
    if not callable(verify_fn):
        raise RuntimeError(f"Verifier must define verify(...): {verifier_path}")
    return verify_fn


def _extract_steps(text: str, *, max_steps: int = 12) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    steps = []
    for ln in lines:
        if re.match(r"^(\d+\.|\-|\*|\+)\s+", ln):
            steps.append(re.sub(r"^(\d+\.|\-|\*|\+)\s+", "", ln).strip())
    if not steps:
        steps = lines[:max_steps]
    return steps[:max_steps]


def _aggregate(values: List[float], mode: str) -> float:
    if not values:
        return 0.0
    mode = (mode or "mean").lower()
    if mode == "min":
        return min(values)
    if mode == "product":
        out = 1.0
        for v in values:
            out *= v
        return out
    return sum(values) / float(len(values))


def verify(
    prompt: str,
    completion: str,
    workdir: str,
    *,
    verifier: str,
    verifier_kwargs: Optional[Dict[str, Any]] = None,
    ema_alpha: float = 0.2,
    max_steps: int = 12,
    agg: str = "mean",
    reward_mode: str = "process",
    min_score: float = 0.0,
    **kwargs,
) -> VerifyResult:
    """PRIME-style implicit process rewards.

    Uses outcome reward from a base verifier to update per-step values.
    """
    verify_fn = _load_verifier(verifier)
    base = verify_fn(prompt, completion, workdir, **(verifier_kwargs or {}), **kwargs)
    outcome_reward = float(getattr(base, "reward", 0.0))
    steps = _extract_steps(completion, max_steps=max_steps)

    step_values: List[float] = []
    for step in steps:
        prev = _STATE["values"].get(step, outcome_reward)
        new_val = (1.0 - ema_alpha) * prev + ema_alpha * outcome_reward
        _STATE["values"][step] = new_val
        _STATE["counts"][step] = _STATE["counts"].get(step, 0.0) + 1.0
        step_values.append(new_val)

    process_reward = _aggregate(step_values, agg)
    if reward_mode == "combined":
        reward = (process_reward + outcome_reward) / 2.0
    else:
        reward = process_reward

    passed = bool(getattr(base, "passed", False)) and reward >= min_score

    return VerifyResult(
        reward=reward,
        passed=passed,
        info={
            "mode": "prime",
            "base_reward": outcome_reward,
            "process_reward": process_reward,
            "steps": steps,
            "step_values": step_values,
            "agg": agg,
            "ema_alpha": ema_alpha,
            "base_info": getattr(base, "info", {}),
        },
        artifacts_dir=workdir,
    )
