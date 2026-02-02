from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_jsonl(path: Path, rows):
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def run_cmd(cmd: list[str], cwd: Optional[Path] = None, timeout_s: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, timeout=timeout_s, capture_output=True, text=True)

@dataclass
class SystemInfo:
    python: str
    python_arch: str
    platform: str
    macos_version: Optional[str]
    machine: str
    cpu_count: int
    has_metal: Optional[bool]
    has_mlx: bool
    mlx_version: Optional[str]

def detect_system() -> SystemInfo:
    has_mlx = False
    mlx_version = None
    try:
        import mlx  # type: ignore
        has_mlx = True
        mlx_version = getattr(mlx, "__version__", None)
    except Exception:
        pass

    # Metal detection (best-effort): on macOS we assume Metal is present; for CI, this is not reliable.
    has_metal = None
    if sys.platform == "darwin":
        has_metal = True

    macos_version = None
    if sys.platform == "darwin":
        macos_version = platform.mac_ver()[0] or None

    py_arch = platform.architecture()[0] or "unknown"

    return SystemInfo(
        python=sys.version.split()[0],
        python_arch=py_arch,
        platform=platform.platform(),
        macos_version=macos_version,
        machine=platform.machine(),
        cpu_count=os.cpu_count() or 0,
        has_metal=has_metal,
        has_mlx=has_mlx,
        mlx_version=mlx_version,
    )

def require(cond: bool, msg: str):
    if not cond:
        raise RuntimeError(msg)

def copytree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def tree_map(fn, tree):
    if tree is None:
        return None
    if isinstance(tree, dict):
        return {k: tree_map(fn, v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(tree_map(fn, v) for v in tree)
    return fn(tree)


def tree_add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: tree_add(a.get(k), b.get(k)) for k in a.keys() | b.keys()}
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a)(tree_add(x, y) for x, y in zip(a, b))
    return a + b


def tree_scale(tree, scale: float):
    return tree_map(lambda x: x * scale, tree)


def tree_leaves(tree) -> list:
    leaves = []
    if tree is None:
        return leaves
    if isinstance(tree, dict):
        for v in tree.values():
            leaves.extend(tree_leaves(v))
    elif isinstance(tree, (list, tuple)):
        for v in tree:
            leaves.extend(tree_leaves(v))
    else:
        leaves.append(tree)
    return leaves


def clip_grad_norm(grads, max_norm: float):
    """Clip gradients by global L2 norm. Returns clipped grads."""
    import mlx.core as mx

    leaves = tree_leaves(grads)
    if not leaves:
        return grads
    total_norm_sq = mx.array(0.0)
    for g in leaves:
        total_norm_sq = total_norm_sq + (g * g).sum()
    total_norm = mx.sqrt(total_norm_sq)
    clip_coef = mx.minimum(mx.array(max_norm) / mx.maximum(total_norm, mx.array(1e-8)), mx.array(1.0))
    return tree_map(lambda g: g * clip_coef, grads)


def latency_summary_ms(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {}
    items = sorted(samples)
    n = len(items)
    mean = sum(items) / n

    def _pct(p: float) -> float:
        if n == 1:
            return items[0]
        idx = int((p / 100.0) * (n - 1))
        return items[max(0, min(idx, n - 1))]

    return {
        "mean": mean,
        "p50": _pct(50),
        "p90": _pct(90),
        "p99": _pct(99),
        "max": items[-1],
    }
