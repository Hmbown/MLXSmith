"""Orchestrator Daemon for MLXSmith Multi-Process RLM.

Central queue-based job scheduler that coordinates between:
- Inference server (generates rollouts)
- Trainer worker (consumes batches and updates weights)

Manages rollout requests, training batches, and weight updates.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from rich.console import Console

from ..config import ProjectConfig
from ..rlm.corpus import append_corpus, load_corpus, sample_corpus
from ..rlm.gating import load_state, save_state, should_accept, update_state
from ..rlm.history import append_history
from ..rlm.inference import Rollout, build_tasks
from ..rlm.weights import WeightPointerStore, WeightPointerIPC
from ..runs import new_run, snapshot_config
from ..util import ensure_dir, now_ts, write_jsonl
from .queue import MessageQueue, MessageType, Message
from .inference_worker import InferenceConfig, run_inference_worker
from .trainer_worker import TrainerConfig, run_trainer_worker


console = Console()


@dataclass
class DaemonConfig:
    """Configuration for orchestrator daemon."""
    project_root: Path
    model_spec: str
    
    # Process management
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
    
    # Paths
    weights_dir: Optional[Path] = None
    checkpoint_dir: Optional[Path] = None
    
    # Gating
    gating_mode: str = "strict"
    gating_threshold: float = 0.0
    gating_ema_alpha: float = 0.2
    
    # Verifier
    verifier_backend: str = "pytest"
    verifier_timeout_s: int = 30


@dataclass
class ProcessHandle:
    """Handle for a managed process."""
    name: str
    process: mp.Process
    config: Any
    restart_count: int = 0
    last_restart: float = 0.0
    healthy: bool = True
    start_time: float = field(default_factory=time.time)


class OrchestratorDaemon:
    """Orchestrator daemon for multi-process RLM.
    
    Responsibilities:
    - Spawn and manage inference and trainer processes
    - Coordinate rollout requests and training batches
    - Manage weight pointer updates
    - Handle process lifecycle, monitoring, and restarts
    - Graceful shutdown handling
    """
    
    def __init__(self, config: DaemonConfig, project_cfg: ProjectConfig):
        self.config = config
        self.project_cfg = project_cfg
        self.queue = MessageQueue(maxsize=10000)
        self._processes: Dict[str, ProcessHandle] = {}
        self._shutdown = False
        self._pointer_store: Optional[WeightPointerStore] = None
        self._current_iteration = 0
        self._metrics: List[Dict] = []
        
        # Setup paths
        self._weights_dir = config.weights_dir or (config.project_root / "runs" / "rlm_weights")
        self._checkpoint_dir = config.checkpoint_dir or (config.project_root / "runs" / "rlm_checkpoints")
        self._state_path = config.project_root / "runs" / "rlm_state.json"
        self._history_path = config.project_root / "runs" / "rlm_history.jsonl"
        self._corpus_path = config.project_root / "runs" / "rlm_corpus.jsonl"
        
        ensure_dir(self._weights_dir)
        ensure_dir(self._checkpoint_dir)
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(sig, frame):
            console.print("[yellow]Orchestrator received shutdown signal[/yellow]")
            self._shutdown = True
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def _spawn_inference_worker(self) -> ProcessHandle:
        """Spawn the inference worker process."""
        inf_config = InferenceConfig(
            model_spec=self.config.model_spec,
            host=self.config.inference_host,
            port=self.config.inference_port,
            max_seq_len=self.project_cfg.model.max_seq_len,
            dtype=self.project_cfg.model.dtype,
            trust_remote_code=self.project_cfg.model.trust_remote_code,
            use_chat_template=self.project_cfg.model.use_chat_template,
            weights_dir=self._weights_dir,
            hot_reload=True,
        )
        
        # Create process
        process = mp.Process(
            target=run_inference_worker,
            args=(inf_config, None),  # Queue passed separately
            name="inference_worker",
            daemon=False,
        )
        
        handle = ProcessHandle(
            name="inference",
            process=process,
            config=inf_config,
        )
        
        process.start()
        console.print(f"[green]Spawned inference worker (PID: {process.pid})[/green]")
        
        return handle
    
    def _spawn_trainer_worker(self) -> ProcessHandle:
        """Spawn the trainer worker process."""
        # Resolve base model
        from ..models import resolve_model_spec
        base_model, adapter_path, _ = resolve_model_spec(
            self.config.project_root, self.config.model_spec, self.project_cfg
        )
        
        trainer_config = TrainerConfig(
            model_spec=self.config.model_spec,
            base_model=base_model,
            max_seq_len=self.project_cfg.model.max_seq_len,
            dtype=self.project_cfg.model.dtype,
            trust_remote_code=self.project_cfg.model.trust_remote_code,
            lr=self.project_cfg.train.lr,
            weight_decay=self.project_cfg.train.weight_decay,
            kl_coeff=self.project_cfg.rft.kl_coeff,
            normalize_advantage=self.project_cfg.rft.normalize_advantage,
            lora_r=self.project_cfg.lora.r,
            lora_alpha=self.project_cfg.lora.alpha,
            lora_dropout=self.project_cfg.lora.dropout,
            lora_target_modules=list(self.project_cfg.lora.target_modules or []),
            lora_num_layers=self.project_cfg.lora.num_layers,
            weights_dir=self._weights_dir,
            checkpoint_dir=self._checkpoint_dir,
            reference_model=self.project_cfg.rft.reference_model,
        )
        
        # Create process
        process = mp.Process(
            target=run_trainer_worker,
            args=(trainer_config, None),  # Queue passed separately
            name="trainer_worker",
            daemon=False,
        )
        
        handle = ProcessHandle(
            name="trainer",
            process=process,
            config=trainer_config,
        )
        
        process.start()
        console.print(f"[green]Spawned trainer worker (PID: {process.pid})[/green]")
        
        return handle
    
    def _monitor_processes(self) -> None:
        """Monitor processes and restart if needed."""
        current_time = time.time()
        
        for name, handle in list(self._processes.items()):
            # Check if process is alive
            if not handle.process.is_alive():
                if self._shutdown:
                    continue
                
                console.print(f"[red]Process {name} (PID: {handle.process.pid}) died[/red]")
                handle.healthy = False
                
                # Check restart limit
                if handle.restart_count >= self.config.max_restarts:
                    console.print(f"[red]Process {name} exceeded max restarts[/red]")
                    continue
                
                # Check restart delay
                if current_time - handle.last_restart < self.config.restart_delay:
                    time.sleep(self.config.restart_delay)
                
                # Restart process
                console.print(f"[yellow]Restarting {name}...[/yellow]")
                
                if name == "inference":
                    new_handle = self._spawn_inference_worker()
                elif name == "trainer":
                    new_handle = self._spawn_trainer_worker()
                else:
                    continue
                
                new_handle.restart_count = handle.restart_count + 1
                new_handle.last_restart = current_time
                self._processes[name] = new_handle
    
    def _health_check(self) -> Dict[str, Any]:
        """Perform health checks on all processes via queues."""
        results = {}
        
        # Check inference via queue
        if "inference" in self._processes:
            self.queue.send(
                "control",
                MessageType.HEALTH_CHECK,
                {},
                source="daemon",
            )
            # Response will be processed in main loop
        
        # Check trainer via queue
        if "trainer" in self._processes:
            self.queue.send(
                "train_batches",  # Trainer reads from train_batches
                MessageType.HEALTH_CHECK,
                {},
                source="daemon",
            )
        
        return results
    
    def _forward_weight_updates(self) -> None:
        """Forward weight updates from trainer to inference."""
        # Check for weight updates from trainer
        msg = self.queue.receive("weight_updates", timeout=0)
        if msg and msg.msg_type == MessageType.WEIGHT_UPDATE:
            # Forward to inference worker
            self.queue.send(
                "weight_forward",
                MessageType.WEIGHT_UPDATE,
                msg.payload,
                source="daemon",
            )
            
            # Also update inference pointer
            if self._pointer_store:
                pointer = WeightPointerIPC(
                    base_model=msg.payload.get("base_model", ""),
                    adapter_path=msg.payload.get("adapter_path"),
                    iteration=msg.payload.get("version", 0),
                    updated_at=now_ts(),
                    version=msg.payload.get("version", 0),
                    name="inference",
                )
                self._pointer_store.save(pointer)
                console.print(f"[blue]Forwarded weight update: {pointer.adapter_path}[/blue]")
    
    def _shutdown_all(self) -> None:
        """Shutdown all processes gracefully."""
        console.print("[yellow]Shutting down all processes...[/yellow]")
        
        # Send shutdown messages
        for name in self._processes:
            self.queue.send(
                "control",
                MessageType.SHUTDOWN,
                {},
                source="daemon",
            )
        
        # Wait for processes to terminate
        for name, handle in self._processes.items():
            console.print(f"  Waiting for {name}...")
            handle.process.join(timeout=10.0)
            
            if handle.process.is_alive():
                console.print(f"  Force terminating {name}")
                handle.process.terminate()
                handle.process.join(timeout=5.0)
                
                if handle.process.is_alive():
                    handle.process.kill()
        
        # Stop queue manager
        self.queue.stop()
        
        console.print("[green]All processes shutdown[/green]")
    
    def run_iteration(self, iteration: int) -> bool:
        """Run a single RLM iteration.
        
        Returns True if iteration completed successfully.
        """
        console.print(f"\n[bold blue]=== RLM Iteration {iteration} ===[/bold blue]")
        
        run = new_run(self.config.project_root, "rlm")
        snapshot_config(self.project_cfg.model_dump(), run.config_snapshot_path)
        
        # Phase 1: Generate tasks (via inference worker API)
        console.print("  [dim]Generating tasks...[/dim]")
        # Tasks are generated by querying inference worker
        
        # Phase 2: Collect rollouts (via inference worker)
        console.print("  [dim]Collecting rollouts...[/dim]")
        # Rollouts are generated via /internal/rollout endpoint
        
        # Phase 3: Send training batch to trainer
        console.print("  [dim]Sending training batch...[/dim]")
        
        # Phase 4: Wait for training completion
        console.print("  [dim]Waiting for training...[/dim]")
        
        # This is a placeholder - actual implementation would
        # coordinate via queues and the inference worker API
        
        return True
    
    def run(self) -> None:
        """Run the orchestrator daemon."""
        self._setup_signal_handlers()
        
        console.print("[bold green]Starting MLXSmith Orchestrator[/bold green]")
        
        # Start queue manager
        self.queue.start()
        console.print("[dim]Queue manager started[/dim]")
        
        # Initialize weight pointer store
        self._pointer_store = WeightPointerStore(self._weights_dir)
        console.print(f"[dim]Weight store: {self._weights_dir}[/dim]")
        
        # Spawn worker processes
        console.print("[dim]Spawning worker processes...[/dim]")
        self._processes["inference"] = self._spawn_inference_worker()
        self._processes["trainer"] = self._spawn_trainer_worker()
        
        # Wait for processes to initialize
        console.print("[dim]Waiting for workers to initialize...[/dim]")
        time.sleep(5.0)
        
        # Load state
        state = load_state(self._state_path)
        self._current_iteration = state.last_iteration + 1
        
        last_health_check = time.time()
        
        try:
            # Main orchestrator loop
            while not self._shutdown:
                # Monitor processes
                self._monitor_processes()
                
                # Health checks
                current_time = time.time()
                if current_time - last_health_check > self.config.health_check_interval:
                    self._health_check()
                    last_health_check = current_time
                
                # Forward weight updates
                self._forward_weight_updates()
                
                # Process queue messages
                self._process_queue_messages()
                
                # Small sleep to prevent busy waiting
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            console.print("[yellow]Interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"[red]Orchestrator error: {e}[/red]")
            traceback.print_exc()
        finally:
            self._shutdown_all()
    
    def _process_queue_messages(self) -> None:
        """Process pending queue messages."""
        # Process responses from inference
        msg = self.queue.receive("rollout_responses", timeout=0)
        if msg:
            # Handle rollout response
            pass
        
        # Process training completion
        msg = self.queue.receive("train_complete", timeout=0)
        if msg:
            # Handle training completion
            pass
        
        # Process checkpoints
        msg = self.queue.receive("checkpoints", timeout=0)
        if msg:
            # Handle checkpoint notification
            pass


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
