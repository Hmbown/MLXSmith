"""Passthrough helpers for mlx-lm-lora CLI integration."""

from __future__ import annotations

import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

from rich.console import Console

console = Console()


def ensure_available() -> None:
    if importlib.util.find_spec("mlx_lm_lora") is None:
        raise RuntimeError(
            "mlx-lm-lora is not installed. Install with: pip install 'mlxsmith[lora]' or 'mlx-lm-lora'"
        )


def _flag_present(args: Sequence[str], *flags: str) -> bool:
    for flag in flags:
        if flag in args:
            return True
        if flag.startswith("--"):
            prefix = flag + "="
            if any(a.startswith(prefix) for a in args):
                return True
    return False


def _append_flag(cmd: list[str], args: Sequence[str], flag: str, value: Optional[str]) -> None:
    if value is None:
        return
    if _flag_present(args, flag):
        return
    cmd.extend([flag, value])


def _base_python_cmd(module: str) -> list[str]:
    return [sys.executable, "-m", module]


def build_train_command(
    *,
    config: Optional[str] = None,
    model: Optional[str] = None,
    data: Optional[str] = None,
    train_mode: Optional[str] = None,
    train_type: Optional[str] = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    args = list(extra_args)
    cmd = _base_python_cmd("mlx_lm_lora.train")
    _append_flag(cmd, args, "--config", config)
    _append_flag(cmd, args, "--model", model)
    _append_flag(cmd, args, "--data", data)
    _append_flag(cmd, args, "--train-mode", train_mode)
    _append_flag(cmd, args, "--train-type", train_type)
    cmd.extend(args)
    return cmd


def build_synthetic_command(
    kind: str,
    *,
    extra_args: Sequence[str] = (),
) -> list[str]:
    kind = kind.strip().lower()
    module = {
        "prompts": "mlx_lm_lora.synthetic_prompts",
        "sft": "mlx_lm_lora.synthetic_sft",
        "dpo": "mlx_lm_lora.synthetic_dpo",
    }.get(kind)
    if module is None:
        raise ValueError(f"Unknown synthetic kind: {kind}")
    cmd = _base_python_cmd(module)
    cmd.extend(list(extra_args))
    return cmd


def build_judge_command(*, extra_args: Sequence[str] = ()) -> list[str]:
    cmd = _base_python_cmd("mlx_lm_lora.train_judge")
    cmd.extend(list(extra_args))
    return cmd


def build_reward_functions_command(*, extra_args: Sequence[str] = ()) -> list[str]:
    cmd = _base_python_cmd("mlx_lm_lora.train")
    cmd.append("--list-reward-functions")
    cmd.extend(list(extra_args))
    return cmd


def run_command(
    cmd: Sequence[str],
    *,
    dry_run: bool = False,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
) -> int:
    if dry_run:
        console.print("[cyan]mlx-lm-lora cmd[/cyan]", shlex.join(list(cmd)))
        return 0
    ensure_available()
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    console.print("[cyan]mlx-lm-lora cmd[/cyan]", shlex.join(list(cmd)))
    result = subprocess.run(list(cmd), cwd=str(cwd) if cwd else None, env=run_env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"mlx-lm-lora failed with exit code {result.returncode}")
    return result.returncode
