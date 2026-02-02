"""Orchestrator Daemon for MLXSmith Multi-Process RLM.

Thin wrapper around the orchestrated RLM loop to keep the legacy daemon API
usable while delegating implementation to the maintained orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console

from ..config import ProjectConfig

console = Console()


@dataclass
class DaemonConfig:
    """Configuration for orchestrator daemon."""

    project_root: Path
    model_spec: str

    # Process management (reserved for future extensions)
    inference_port: int = 8080
    inference_host: str = "0.0.0.0"
    max_restarts: int = 3
    restart_delay: float = 5.0
    health_check_interval: float = 10.0

    # Training config
    iterations: int = 50
    tasks_per_iter: int = 80
    rollouts_per_task: int = 8
    batch_size: int = 32

    # Paths (currently derived from project_root in the orchestrator)
    weights_dir: Optional[Path] = None
    checkpoint_dir: Optional[Path] = None

    # Gating
    gating_mode: str = "strict"
    gating_threshold: float = 0.0
    gating_ema_alpha: float = 0.2

    # Verifier
    verifier_backend: str = "pytest"
    verifier_timeout_s: int = 30


class OrchestratorDaemon:
    """Orchestrator daemon wrapper for multi-process RLM."""

    def __init__(self, config: DaemonConfig, project_cfg: ProjectConfig):
        self.config = config
        # Work on a copy so daemon overrides don't mutate caller state.
        self.project_cfg = project_cfg.model_copy(deep=True)
        self._apply_overrides()
        from ..rlm.loop import RLMOrchestrator
        self._orchestrator = RLMOrchestrator(
            project_root=self.config.project_root,
            cfg=self.project_cfg,
            model_spec=self.config.model_spec,
            iterations=self.config.iterations,
            resume=False,
        )

    def _apply_overrides(self) -> None:
        """Apply daemon config overrides onto the project config."""
        self.project_cfg.serve.host = self.config.inference_host
        self.project_cfg.serve.port = self.config.inference_port

        self.project_cfg.rlm.iterations = self.config.iterations
        self.project_cfg.rlm.tasks_per_iter = self.config.tasks_per_iter
        self.project_cfg.rlm.rollouts_per_task = self.config.rollouts_per_task
        self.project_cfg.rlm.gating = self.config.gating_mode
        self.project_cfg.rlm.gating_threshold = self.config.gating_threshold
        self.project_cfg.rlm.gating_ema_alpha = self.config.gating_ema_alpha
        self.project_cfg.rlm.verifier_backend = self.config.verifier_backend
        self.project_cfg.rlm.verifier_timeout_s = self.config.verifier_timeout_s

    def run_iteration(self, iteration: int) -> bool:
        """Run a single orchestrated iteration."""
        return self._orchestrator.run_iteration(iteration)

    def run(self) -> None:
        """Run the orchestrated RLM loop."""
        console.print("[bold green]Starting MLXSmith Orchestrator[/bold green]")
        self._orchestrator.run()


def run_daemon(
    project_root: Path,
    project_cfg: ProjectConfig,
    model_spec: Optional[str] = None,
    iterations: Optional[int] = None,
) -> None:
    """Run the orchestrator daemon."""
    config = DaemonConfig(
        project_root=project_root,
        model_spec=model_spec or project_cfg.model.id,
        iterations=iterations or project_cfg.rlm.iterations,
        tasks_per_iter=project_cfg.rlm.tasks_per_iter,
        rollouts_per_task=project_cfg.rlm.rollouts_per_task,
        gating_mode=project_cfg.rlm.gating,
        gating_threshold=project_cfg.rlm.gating_threshold,
        gating_ema_alpha=project_cfg.rlm.gating_ema_alpha,
        verifier_backend=project_cfg.rlm.verifier_backend,
        verifier_timeout_s=project_cfg.rlm.verifier_timeout_s,
    )

    daemon = OrchestratorDaemon(config, project_cfg)
    daemon.run()
