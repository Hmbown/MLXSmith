from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import VerifyResult


def _load_verifier(path: str):
    verifier_path = Path(path)
    spec = importlib.util.spec_from_file_location(verifier_path.stem, verifier_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load verifier: {verifier_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    verify_fn = getattr(module, "verify", None)
    if not callable(verify_fn):
        raise RuntimeError(f"Verifier must define verify(...): {verifier_path}")
    return verify_fn


def verify(
    prompt: str,
    completion: str,
    workdir: str,
    *,
    verifiers: List[Any],
    mode: str = "all",
    weights: Optional[List[float]] = None,
    per_verifier_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    **kwargs,
) -> VerifyResult:
    """Compose multiple verifiers with AND/OR/weighted reward aggregation.

    verifiers: list of paths or dicts {path, kwargs}.
    mode: all | any | weighted
    weights: optional weights for weighted reward aggregation.
    per_verifier_kwargs: optional mapping of path -> kwargs.
    """
    results = []
    latencies: Dict[str, float] = {}
    per_kwargs = per_verifier_kwargs or {}

    for idx, entry in enumerate(verifiers):
        if isinstance(entry, dict):
            path = entry.get("path")
            extra = entry.get("kwargs") or {}
        else:
            path = entry
            extra = {}

        if not path:
            continue
        verify_fn = _load_verifier(str(path))
        merged = dict(kwargs)
        merged.update(per_kwargs.get(str(path), {}))
        merged.update(extra)

        t0 = time.time()
        res = verify_fn(prompt, completion, workdir, **merged)
        latencies[str(path)] = (time.time() - t0) * 1000.0
        results.append((str(path), res))

    if not results:
        return VerifyResult(
            reward=0.0,
            passed=False,
            info={"error": "no_verifiers"},
            artifacts_dir=workdir,
        )

    mode = (mode or "all").lower()
    passes = [bool(getattr(r, "passed", False)) for _p, r in results]
    rewards = [float(getattr(r, "reward", 0.0)) for _p, r in results]

    if mode == "any":
        passed = any(passes)
        reward = max(rewards) if rewards else 0.0
    elif mode == "weighted":
        if weights and len(weights) == len(rewards):
            reward = sum(w * r for w, r in zip(weights, rewards))
        else:
            reward = sum(rewards) / max(1, len(rewards))
        passed = reward > 0.0 and any(passes)
    else:
        passed = all(passes)
        reward = sum(rewards) / max(1, len(rewards))

    return VerifyResult(
        reward=reward,
        passed=passed,
        info={
            "mode": mode,
            "verifiers": [
                {
                    "path": path,
                    "passed": bool(getattr(res, "passed", False)),
                    "reward": float(getattr(res, "reward", 0.0)),
                    "info": getattr(res, "info", {}),
                }
                for path, res in results
            ],
            "verifier_latencies_ms": latencies,
        },
        artifacts_dir=workdir,
    )

