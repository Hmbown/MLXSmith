from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .util import ensure_dir

@dataclass
class RunPaths:
    run_dir: Path
    logs_dir: Path
    checkpoints_dir: Path
    adapter_dir: Path
    artifacts_dir: Path
    metrics_path: Path
    config_snapshot_path: Path

def new_run(root: Path, kind: str) -> RunPaths:
    runs_root = ensure_dir(root / "runs")
    # monotonically increasing id by counting existing runs of same kind
    existing = sorted([p for p in runs_root.glob(f"{kind}_*") if p.is_dir()])
    next_idx = len(existing) + 1
    run_name = f"{kind}_{next_idx:04d}"
    run_dir = ensure_dir(runs_root / run_name)
    logs_dir = ensure_dir(run_dir / "logs")
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    adapter_dir = ensure_dir(run_dir / "adapter")
    artifacts_dir = ensure_dir(run_dir / "artifacts")
    metrics_path = run_dir / "metrics.jsonl"
    config_snapshot_path = run_dir / "config.snapshot.yaml"
    return RunPaths(
        run_dir=run_dir,
        logs_dir=logs_dir,
        checkpoints_dir=checkpoints_dir,
        adapter_dir=adapter_dir,
        artifacts_dir=artifacts_dir,
        metrics_path=metrics_path,
        config_snapshot_path=config_snapshot_path,
    )

def snapshot_config(cfg_dict: dict, path: Path):
    path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")
