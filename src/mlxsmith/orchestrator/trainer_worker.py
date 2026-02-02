"""Trainer Worker Process for MLXSmith Orchestrator.

Runs as a separate process for training.
Consumes training batches from queue.
Publishes adapter checkpoints and signals weight updates.
"""

from __future__ import annotations

import json
import signal
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import ProjectConfig
from ..llm.registry import get_llm_backend
from ..rlm.inference import Rollout
from ..rlm.weights import WeightPointerStore, WeightPointerIPC
from ..train.lora import LoRAConfig
from ..util import ensure_dir, now_ts
from .queue import MessageQueue, MessageType, Message


@dataclass
class TrainerConfig:
    """Configuration for trainer worker."""
    model_spec: str
    base_model: str
    backend: str = "mlx-lm"
    max_seq_len: int = 8192
    dtype: str = "bf16"
    trust_remote_code: bool = False
    
    # Training config
    lr: float = 2e-4
    weight_decay: float = 0.0
    kl_coeff: float = 0.02
    normalize_advantage: bool = True
    
    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "o_proj"])
    lora_num_layers: int = 0
    
    # Paths
    weights_dir: Optional[Path] = None
    checkpoint_dir: Optional[Path] = None
    
    # Reference model for KL penalty
    reference_model: Optional[str] = None


@dataclass
class TrainingBatch:
    """A batch of rollouts for training."""
    iteration: int
    run_id: str
    rollouts: List[Rollout]
    task_metadata: List[Dict[str, Any]] = field(default_factory=list)


class TrainerWorker:
    """Trainer worker process for RLM training.
    
    Runs in a separate process, handles:
    - Consuming training batches from queue
    - Running forward/backward passes
    - Publishing adapter checkpoints
    - Signaling weight updates to orchestrator
    """
    
    def __init__(
        self,
        config: TrainerConfig,
        queue: Optional[MessageQueue] = None,
    ):
        self.config = config
        self.queue = queue
        self._llm = None
        self._ref_llm = None
        self._optimizer = None
        self._pointer_store: Optional[WeightPointerStore] = None
        self._current_iteration = 0
        self._current_adapter: Optional[str] = None
        self._shutdown = False
        self._metrics: List[Dict] = []
    
    def _load_model(self) -> None:
        """Load the model and optimizer."""
        print("[TrainerWorker] Loading model...")
        self._llm = get_llm_backend(self.config.backend)
        
        # Load base model
        self._llm.load(
            self.config.base_model,
            max_seq_len=self.config.max_seq_len,
            dtype=self.config.dtype,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        # Setup weight pointer store
        if self.config.weights_dir:
            self._pointer_store = WeightPointerStore(self.config.weights_dir)
            pointer = self._pointer_store.load("trainer", self.config.base_model)
            
            if pointer.adapter_path:
                print(f"[TrainerWorker] Loading adapter: {pointer.adapter_path}")
                self._llm.apply_adapter(pointer.adapter_path)
                self._current_adapter = pointer.adapter_path
                self._current_iteration = pointer.iteration
            else:
                # Initialize new LoRA adapter
                print("[TrainerWorker] Initializing LoRA adapter...")
                lora_cfg = LoRAConfig(
                    r=self.config.lora_r,
                    alpha=self.config.lora_alpha,
                    dropout=self.config.lora_dropout,
                    target_modules=list(self.config.lora_target_modules),
                    num_layers=self.config.lora_num_layers,
                )
                self._llm.apply_lora_from_config(lora_cfg)
        
        # Setup optimizer
        self._optimizer, _ = self._llm.optimizer_and_params(
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        
        # Load reference model if needed for KL
        if self.config.reference_model and self.config.kl_coeff > 0:
            print("[TrainerWorker] Loading reference model...")
            self._ref_llm = get_llm_backend(self.config.backend)
            self._ref_llm.load(
                self.config.reference_model,
                max_seq_len=self.config.max_seq_len,
                dtype=self.config.dtype,
                trust_remote_code=self.config.trust_remote_code,
            )
        
        print("[TrainerWorker] Model loaded successfully")
    
    def _train_on_batch(self, batch: TrainingBatch) -> Dict[str, Any]:
        """Train on a batch of rollouts."""
        rollouts = batch.rollouts
        if not rollouts:
            return {"status": "empty", "loss": 0.0}
        
        # Group rollouts by task
        grouped = defaultdict(list)
        for r in rollouts:
            grouped[r.task_id].append(r)
        
        total_loss = 0.0
        num_tasks = 0
        
        for task_id, task_rollouts in grouped.items():
            if not task_rollouts:
                continue
            
            # Compute advantages
            mean_r = sum(r.reward for r in task_rollouts) / len(task_rollouts)
            std_r = (sum((r.reward - mean_r) ** 2 for r in task_rollouts) / len(task_rollouts)) ** 0.5
            advs = [r.reward - mean_r for r in task_rollouts]
            
            if self.config.normalize_advantage and std_r > 1e-6:
                advs = [a / std_r for a in advs]
            
            # Define loss function
            def loss_fn(_model):
                loss = self._llm.mx.array(0.0)
                for rollout, adv in zip(task_rollouts, advs):
                    logp = self._llm.sequence_logprob(
                        rollout.token_ids,
                        prompt_len=rollout.prompt_len,
                    )
                    
                    # Importance sampling if rollout was from different policy
                    if rollout.logprobs and rollout.weight_adapter and rollout.weight_adapter != self._current_adapter:
                        behavior_logp = self._llm.mx.array(sum(rollout.logprobs))
                        ratio = self._llm.mx.exp(logp - behavior_logp)
                        pg = -ratio * self._llm.mx.array(float(adv))
                    else:
                        pg = -self._llm.mx.array(float(adv)) * logp
                    
                    # KL penalty from reference model
                    if self._ref_llm is not None and self.config.kl_coeff > 0:
                        ref_logp = self._ref_llm.sequence_logprob(
                            rollout.token_ids,
                            prompt_len=rollout.prompt_len,
                        )
                        pg = pg + self._llm.mx.array(self.config.kl_coeff) * (logp - ref_logp)
                    
                    loss = loss + pg
                
                return loss / self._llm.mx.array(float(len(task_rollouts)))
            
            # Compute gradients and update
            lval, grads = self._llm.value_and_grad(loss_fn)
            if grads is not None:
                self._llm.apply_grads(self._optimizer, grads)
            
            loss_val = float(lval.item()) if hasattr(lval, "item") else float(lval)
            total_loss += loss_val
            num_tasks += 1
            
            # Record metrics
            self._metrics.append({
                "ts": now_ts(),
                "iteration": batch.iteration,
                "task_id": task_id,
                "mean_reward": mean_r,
                "std_reward": std_r,
                "loss": loss_val,
                "num_rollouts": len(task_rollouts),
            })
        
        avg_loss = total_loss / max(1, num_tasks)
        return {
            "status": "success",
            "loss": avg_loss,
            "num_tasks": num_tasks,
            "num_rollouts": len(rollouts),
        }
    
    def _save_checkpoint(self, iteration: int) -> Optional[str]:
        """Save adapter checkpoint and return path."""
        if not self.config.checkpoint_dir:
            return None
        
        checkpoint_path = ensure_dir(self.config.checkpoint_dir / f"iter_{iteration:04d}")
        
        self._llm.save_adapter(
            str(checkpoint_path),
            metadata={
                "base_model": self.config.base_model,
                "source_adapter": self._current_adapter,
                "iteration": iteration,
                "kind": "rlm",
            },
        )
        
        # Save metrics
        metrics_path = checkpoint_path / "training_metrics.jsonl"
        with open(metrics_path, "w") as f:
            for m in self._metrics:
                f.write(json.dumps(m) + "\n")
        
        self._current_adapter = str(checkpoint_path)
        self._current_iteration = iteration
        
        return str(checkpoint_path)
    
    def _update_weight_pointer(self, adapter_path: str, iteration: int) -> None:
        """Update the weight pointer for inference to pick up."""
        if not self._pointer_store:
            return
        
        pointer = WeightPointerIPC(
            base_model=self.config.base_model,
            adapter_path=adapter_path,
            iteration=iteration,
            updated_at=now_ts(),
            version=iteration,  # Use iteration as version
            name="trainer",
        )
        
        self._pointer_store.save(pointer)
        print(f"[TrainerWorker] Updated weight pointer: {adapter_path} (iter {iteration})")
    
    def _handle_train_batch(self, msg: Message) -> Message:
        """Handle a training batch message."""
        payload = msg.payload
        
        # Deserialize rollouts
        rollout_data = payload.get("rollouts", [])
        rollouts = []
        for r in rollout_data:
            rollouts.append(Rollout(
                task_id=r["task_id"],
                prompt=r["prompt"],
                completion=r["completion"],
                token_ids=r["token_ids"],
                prompt_len=r["prompt_len"],
                logprobs=r.get("logprobs"),
                passed=r["passed"],
                reward=r["reward"],
                verifier_latency_ms=r["verifier_latency_ms"],
                weight_adapter=r.get("weight_adapter"),
            ))
        
        batch = TrainingBatch(
            iteration=payload.get("iteration", 0),
            run_id=payload.get("run_id", ""),
            rollouts=rollouts,
            task_metadata=payload.get("task_metadata", []),
        )
        
        # Train
        result = self._train_on_batch(batch)
        
        # Save checkpoint
        checkpoint_path = None
        if payload.get("save_checkpoint", False):
            checkpoint_path = self._save_checkpoint(batch.iteration)
            
            # Update weight pointer for hot-reload
            if checkpoint_path:
                self._update_weight_pointer(checkpoint_path, batch.iteration)
        
        return Message(
            msg_type=MessageType.TRAIN_COMPLETE,
            payload={
                "request_id": msg.msg_id,
                "iteration": batch.iteration,
                "run_id": batch.run_id,
                "result": result,
                "checkpoint_path": checkpoint_path,
            },
            source="trainer",
        )
    
    def _handle_health_check(self, msg: Message) -> Message:
        """Handle a health check request."""
        return Message(
            msg_type=MessageType.HEALTH_RESPONSE,
            payload={
                "status": "healthy",
                "base_model": self.config.base_model,
                "current_adapter": self._current_adapter,
                "current_iteration": self._current_iteration,
                "metrics_count": len(self._metrics),
            },
            source="trainer",
        )
    
    def _process_message(self, msg: Message) -> Optional[Message]:
        """Process a single message."""
        try:
            if msg.msg_type == MessageType.TRAIN_BATCH:
                return self._handle_train_batch(msg)
            elif msg.msg_type == MessageType.HEALTH_CHECK:
                return self._handle_health_check(msg)
            elif msg.msg_type == MessageType.SHUTDOWN:
                self._shutdown = True
                return None
            else:
                print(f"[TrainerWorker] Unknown message type: {msg.msg_type}")
                return None
        except Exception as e:
            print(f"[TrainerWorker] Error processing message: {e}")
            traceback.print_exc()
            return Message(
                msg_type=MessageType.TRAIN_COMPLETE,
                payload={
                    "request_id": msg.msg_id,
                    "status": "error",
                    "error": str(e),
                },
                source="trainer",
            )
    
    def run(self) -> None:
        """Run the trainer worker loop."""
        # Setup signal handlers
        def signal_handler(sig, frame):
            print("[TrainerWorker] Shutting down...")
            self._shutdown = True
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Load model
        self._load_model()
        
        if not self.queue:
            print("[TrainerWorker] No queue provided, running in standalone mode")
            # Standalone mode - just wait for shutdown
            while not self._shutdown:
                time.sleep(0.1)
            return
        
        print("[TrainerWorker] Starting training loop...")
        
        # Main training loop
        while not self._shutdown:
            try:
                # Check for training batches
                msg = self.queue.receive("train_batches", timeout=1.0)
                
                if msg:
                    print(f"[TrainerWorker] Received training batch: {msg.msg_id}")
                    response = self._process_message(msg)
                    
                    if response:
                        self.queue.get_queue("train_complete").put(response.to_dict())
                        
                        # Signal weight update if checkpoint was saved
                        if response.payload.get("checkpoint_path"):
                            weight_update = Message(
                                msg_type=MessageType.WEIGHT_UPDATE,
                                payload={
                                    "adapter_path": response.payload["checkpoint_path"],
                                    "version": response.payload.get("iteration", 0),
                                    "base_model": self.config.base_model,
                                },
                                source="trainer",
                            )
                            self.queue.get_queue("weight_updates").put(weight_update.to_dict())
                            self.queue.get_queue("checkpoints").put(weight_update.to_dict())
                
                # Check control queue
                control_msg = self.queue.receive("control", timeout=0)
                if control_msg:
                    self._process_message(control_msg)
                    
            except Exception as e:
                print(f"[TrainerWorker] Error in main loop: {e}")
                traceback.print_exc()
                time.sleep(1.0)
        
        print("[TrainerWorker] Shutdown complete")


def run_trainer_worker(
    config: TrainerConfig,
    queue: Optional[MessageQueue] = None,
) -> None:
    """Entry point for trainer worker process."""
    worker = TrainerWorker(config, queue)
    worker.run()
