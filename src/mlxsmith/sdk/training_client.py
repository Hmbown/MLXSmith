"""TrainingClient SDK for MLXSmith.

Async client for training operations with futures-based API.
Provides methods for forward/backward passes, optimizer steps,
checkpoint management, and weight manipulation.

Example:
    >>> from mlxsmith.sdk import TrainingClient
    >>> client = TrainingClient(backend, pool)
    >>> 
    >>> # Run training step
    >>> future = client.forward_backward(batch)
    >>> loss, grads = future.result()
    >>> 
    >>> # Optimizer step
    >>> client.optim_step(grads).result()
    >>> 
    >>> # Save checkpoint
    >>> client.save_state("checkpoint.pt").result()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .future import APIFuture, SdkFuturePool


@dataclass
class ForwardBackwardResult:
    """Result from a forward/backward pass."""
    loss: float
    grads: Any  # Backend-specific gradient type
    metrics: Dict[str, float]
    batch_size: int = 1
    has_grads: bool = False


@dataclass
class OptimizerStepResult:
    """Result from an optimizer step."""
    step: int
    learning_rate: float
    grad_norm: Optional[float]


@dataclass
class CheckpointResult:
    """Result from a checkpoint operation."""
    path: str
    success: bool
    message: str


@dataclass
class WeightsResult:
    """Result for weight operations."""
    weights: Dict[str, Any]
    success: bool
    message: str
    num_tensors: int = 0


class TrainingBatch:
    """A batch of training data.
    
    Supports SFT, preference, and custom loss training.
    """
    
    def __init__(
        self,
        prompts: List[str],
        responses: Optional[List[str]] = None,
        rejected_responses: Optional[List[str]] = None,
        advantages: Optional[List[float]] = None,
        loss_type: str = "sft",
        train_on_prompt: bool = False,
        max_seq_len: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a training batch.
        
        Args:
            prompts: List of prompt strings
            responses: List of response strings (for SFT/positive in preference)
            rejected_responses: List of rejected responses (for preference training)
            advantages: List of advantage values (for RL training)
            loss_type: Type of loss - "sft", "dpo", "orpo", "ppo", "custom"
            train_on_prompt: Whether to compute loss on prompt tokens
            max_seq_len: Maximum sequence length
            extra: Additional batch metadata
        """
        self.prompts = prompts
        self.responses = responses
        self.rejected_responses = rejected_responses
        self.advantages = advantages
        self.loss_type = loss_type
        self.train_on_prompt = train_on_prompt
        self.max_seq_len = max_seq_len
        self.extra = extra or {}
        self._size = len(prompts)
    
    def __len__(self) -> int:
        return self._size
    
    @property
    def is_preference(self) -> bool:
        """Check if this is a preference batch."""
        return self.loss_type in ("dpo", "orpo", "ipo", "preference")
    
    @property
    def is_rl(self) -> bool:
        """Check if this is an RL batch."""
        return self.loss_type in ("ppo", "grpo", "reinforce")


class TrainingClient:
    """Async client for training operations.
    
    Provides a futures-based API for all training operations, enabling
    concurrent execution and flexible callback handling.
    
    Example:
        >>> client = TrainingClient(backend, pool)
        >>> 
        >>> # Async training loop
        >>> for batch in dataloader:
        ...     fb_future = client.forward_backward(batch)
        ...     
        ...     # Chain operations with callbacks
        ...     fb_future.then(lambda r: print(f"Loss: {r.loss}"))
        ...     
        ...     # Get result and continue
        ...     loss, grads = fb_future.result()
        ...     if grads is not None:
        ...         client.optim_step(grads).result()
        >>> 
        >>> # Save checkpoint
        >>> client.save_state("checkpoint.pt").result()
    """
    
    def __init__(
        self,
        backend: Any,
        pool: Optional[SdkFuturePool] = None,
        optimizer: Optional[Any] = None,
        step: int = 0,
    ):
        """Initialize TrainingClient.
        
        Args:
            backend: The LLM backend instance
            pool: Optional SdkFuturePool for async execution (creates default if None)
            optimizer: Optional pre-created optimizer
            step: Initial training step counter
        """
        self.backend = backend
        self.pool = pool or SdkFuturePool(max_workers=1)
        self.optimizer = optimizer
        self._step = step
        self._training_state: Dict[str, Any] = {}
        self._checkpoint_handlers: Dict[str, Callable] = {}
    
    # ========================================================================
    # Core Training Operations
    # ========================================================================
    
    def forward_backward(self, batch: TrainingBatch) -> APIFuture[ForwardBackwardResult]:
        """Run forward and backward pass on a batch.
        
        Args:
            batch: TrainingBatch with prompts and responses
            
        Returns:
            APIFuture resolving to ForwardBackwardResult
            
        Example:
            >>> batch = TrainingBatch(
            ...     prompts=["What is 2+2?"],
            ...     responses=["The answer is 4."],
            ...     loss_type="sft"
            ... )
            >>> future = client.forward_backward(batch)
            >>> result = future.result()
            >>> print(f"Loss: {result.loss}")
        """
        def _run_forward_backward() -> ForwardBackwardResult:
            from . import sft_forward_backward, preference_forward_backward
            
            losses = []
            all_grads = []
            
            if batch.is_preference:
                # Preference training (DPO, ORPO, etc.)
                if batch.rejected_responses is None:
                    raise ValueError("Preference batch requires rejected_responses")
                
                for prompt, chosen, rejected in zip(
                    batch.prompts, 
                    batch.responses or [],
                    batch.rejected_responses
                ):
                    loss, grads = preference_forward_backward(
                        self.backend,
                        prompt,
                        chosen,
                        rejected,
                        algo=batch.loss_type,
                        beta=batch.extra.get("beta", 0.1),
                        reference_backend=batch.extra.get("reference_backend"),
                        kl_coeff=batch.extra.get("kl_coeff", 0.0),
                        delta=batch.extra.get("delta", 0.0),
                        train_on_prompt=batch.train_on_prompt,
                        max_seq_len=batch.max_seq_len,
                    )
                    losses.append(float(loss) if loss is not None else 0.0)
                    if grads is not None:
                        all_grads.append(grads)
            
            elif batch.is_rl:
                # RL training (PPO, etc.)
                # For now, fall back to SFT-style with advantages
                for prompt, response, advantage in zip(
                    batch.prompts,
                    batch.responses or [],
                    batch.advantages or [0.0] * len(batch.prompts)
                ):
                    # Use SFT forward/backward with modified loss
                    loss, grads = sft_forward_backward(
                        self.backend,
                        prompt,
                        response,
                        train_on_prompt=batch.train_on_prompt,
                        max_seq_len=batch.max_seq_len,
                    )
                    # Scale by advantage
                    if loss is not None and advantage != 0.0:
                        loss = loss * advantage
                    losses.append(float(loss) if loss is not None else 0.0)
                    if grads is not None:
                        all_grads.append(grads)
            
            else:
                # Standard SFT
                for prompt, response in zip(batch.prompts, batch.responses or []):
                    loss, grads = sft_forward_backward(
                        self.backend,
                        prompt,
                        response,
                        train_on_prompt=batch.train_on_prompt,
                        max_seq_len=batch.max_seq_len,
                    )
                    losses.append(float(loss) if loss is not None else 0.0)
                    if grads is not None:
                        all_grads.append(grads)
            
            # Average gradients if multiple
            grads = self._aggregate_gradients(all_grads) if all_grads else None
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            
            return ForwardBackwardResult(
                loss=avg_loss,
                grads=grads,
                batch_size=len(batch),
                has_grads=grads is not None,
                metrics={
                    "avg_loss": avg_loss,
                    "num_samples": len(losses),
                }
            )
        
        return self.pool.submit(_run_forward_backward)
    
    def optim_step(self, grads: Optional[Any] = None) -> APIFuture[OptimizerStepResult]:
        """Execute optimizer step with gradients.
        
        Args:
            grads: Gradients from forward/backward (uses stored if None)
            
        Returns:
            APIFuture resolving to OptimizerStepResult
            
        Example:
            >>> # After forward_backward
            >>> grads = fb_result.grads
            >>> step_future = client.optim_step(grads)
            >>> step_info = step_future.result()
            >>> print(f"Step {step_info.step} completed")
        """
        def _run_optim_step() -> OptimizerStepResult:
            from . import optim_step as _optim_step
            
            if self.optimizer is None:
                raise RuntimeError("Optimizer not initialized. Call create_optimizer() first.")
            
            if grads is None:
                raise ValueError("No gradients provided for optimizer step")
            
            _optim_step(self.backend, self.optimizer, grads)
            self._step += 1
            
            # Compute gradient norm if possible
            grad_norm = None
            if hasattr(grads, '__iter__'):
                try:
                    import math
                    grad_norm = math.sqrt(sum(float(g**2) for g in grads if g is not None))
                except Exception:
                    pass
            
            # Get current learning rate
            lr = 0.0
            if hasattr(self.optimizer, 'learning_rate'):
                lr = self.optimizer.learning_rate
            elif isinstance(self.optimizer, dict):
                lr = self.optimizer.get('learning_rate', 0.0)
            
            return OptimizerStepResult(
                step=self._step,
                learning_rate=lr,
                grad_norm=grad_norm,
            )
        
        return self.pool.submit(_run_optim_step)
    
    # ========================================================================
    # Checkpoint Management
    # ========================================================================
    
    def save_state(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> APIFuture[CheckpointResult]:
        """Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
            metadata: Optional metadata to save with checkpoint
            
        Returns:
            APIFuture resolving to CheckpointResult
            
        Example:
            >>> client.save_state("checkpoints/step_1000.pt").result()
            >>> # With metadata
            >>> client.save_state("checkpoint.pt", {"epoch": 5, "score": 0.95}).result()
        """
        def _run_save() -> CheckpointResult:
            try:
                from pathlib import Path
                
                save_path = Path(path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save adapter weights
                full_metadata = {
                    "step": self._step,
                    "training_state": self._training_state,
                    **(metadata or {}),
                }
                
                # Use backend's save_adapter
                self.backend.save_adapter(str(save_path), metadata=full_metadata)
                
                return CheckpointResult(
                    path=str(save_path),
                    success=True,
                    message=f"Checkpoint saved to {save_path}",
                )
            except Exception as e:
                return CheckpointResult(
                    path=path,
                    success=False,
                    message=f"Failed to save checkpoint: {e}",
                )
        
        return self.pool.submit(_run_save)
    
    def load_state(self, path: str) -> APIFuture[CheckpointResult]:
        """Load training checkpoint.
        
        Args:
            path: Path to checkpoint to load
            
        Returns:
            APIFuture resolving to CheckpointResult
            
        Example:
            >>> result = client.load_state("checkpoints/step_1000.pt").result()
            >>> if result.success:
            ...     print(f"Loaded from {result.path}")
        """
        def _run_load() -> CheckpointResult:
            try:
                from pathlib import Path
                import json
                
                load_path = Path(path)
                if not load_path.exists():
                    return CheckpointResult(
                        path=path,
                        success=False,
                        message=f"Checkpoint not found: {path}",
                    )
                
                # Load adapter weights
                self.backend.apply_adapter(str(load_path))
                
                # Try to load metadata
                metadata_path = load_path / "adapter_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    self._step = metadata.get("step", self._step)
                    self._training_state = metadata.get("training_state", {})
                
                return CheckpointResult(
                    path=str(load_path),
                    success=True,
                    message=f"Checkpoint loaded from {load_path}",
                )
            except Exception as e:
                return CheckpointResult(
                    path=path,
                    success=False,
                    message=f"Failed to load checkpoint: {e}",
                )
        
        return self.pool.submit(_run_load)
    
    # ========================================================================
    # Weight Management
    # ========================================================================
    
    def get_weights(self) -> APIFuture[WeightsResult]:
        """Get current model weights.
        
        Returns:
            APIFuture resolving to WeightsResult with weights dictionary
            
        Example:
            >>> weights_future = client.get_weights()
            >>> result = weights_future.result()
            >>> print(f"Loaded {len(result.weights)} weight tensors")
        """
        def _run_get_weights() -> WeightsResult:
            try:
                weights = {}
                
                # Try to get model parameters
                if hasattr(self.backend, 'model'):
                    model = self.backend.model
                    if hasattr(model, 'parameters'):
                        params = model.parameters()
                        if isinstance(params, dict):
                            weights = params
                        else:
                            weights = {"params": params}
                    elif hasattr(model, 'trainable_parameters'):
                        weights = model.trainable_parameters()
                
                return WeightsResult(
                    weights=weights,
                    success=True,
                    message=f"Retrieved {len(weights)} weight tensors",
                    num_tensors=len(weights),
                )
            except Exception as e:
                return WeightsResult(
                    weights={},
                    success=False,
                    message=f"Failed to get weights: {e}",
                    num_tensors=0,
                )
        
        return self.pool.submit(_run_get_weights)
    
    def set_weights(self, weights: Dict[str, Any]) -> APIFuture[WeightsResult]:
        """Set model weights.
        
        Args:
            weights: Dictionary of weight tensors
            
        Returns:
            APIFuture resolving to WeightsResult
            
        Example:
            >>> client.set_weights(new_weights).result()
        """
        def _run_set_weights() -> WeightsResult:
            try:
                # This is backend-specific - for MLX we need to update arrays
                if hasattr(self.backend, 'model'):
                    model = self.backend.model
                    if hasattr(model, 'update'):
                        model.update(weights)
                    elif hasattr(model, 'load_weights'):
                        model.load_weights(weights)
                
                return WeightsResult(
                    weights=weights,
                    success=True,
                    message=f"Set {len(weights)} weight tensors",
                    num_tensors=len(weights),
                )
            except Exception as e:
                return WeightsResult(
                    weights={},
                    success=False,
                    message=f"Failed to set weights: {e}",
                    num_tensors=0,
                )
        
        return self.pool.submit(_run_set_weights)
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def create_optimizer(
        self,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        optimizer: Optional[str] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> APIFuture[Any]:
        """Create optimizer for training.
        
        Args:
            lr: Learning rate
            weight_decay: Weight decay coefficient
            optimizer: Optimizer name (e.g., adamw, adam, qhadam, muon)
            optimizer_kwargs: Extra optimizer kwargs
            
        Returns:
            APIFuture resolving to optimizer instance
            
        Example:
            >>> opt_future = client.create_optimizer(lr=1e-4, weight_decay=0.01)
            >>> client.optimizer = opt_future.result()
        """
        def _run_create_optimizer() -> Any:
            from . import create_optimizer as _create_optimizer
            
            self.optimizer, _ = _create_optimizer(
                self.backend,
                lr=lr,
                weight_decay=weight_decay,
                optimizer=optimizer,
                optimizer_kwargs=optimizer_kwargs,
            )
            return self.optimizer
        
        return self.pool.submit(_run_create_optimizer)
    
    def zero_grad(self) -> APIFuture[None]:
        """Zero out gradients.
        
        Returns:
            APIFuture that completes when gradients are zeroed
        """
        def _run_zero_grad() -> None:
            if self.optimizer is not None and hasattr(self.optimizer, 'zero_grad'):
                self.optimizer.zero_grad()
        
        return self.pool.submit(_run_zero_grad)
    
    # ========================================================================
    # Properties
    # ========================================================================
    
    @property
    def step(self) -> int:
        """Current training step."""
        return self._step
    
    @property
    def training_state(self) -> Dict[str, Any]:
        """Get training state dictionary."""
        return self._training_state.copy()
    
    def update_training_state(self, updates: Dict[str, Any]) -> None:
        """Update training state."""
        self._training_state.update(updates)
    
    # ========================================================================
    # Internal Helpers
    # ========================================================================
    
    def _aggregate_gradients(self, grads_list: List[Any]) -> Any:
        """Aggregate gradients from multiple samples.
        
        Args:
            grads_list: List of gradient dictionaries/arrays
            
        Returns:
            Aggregated gradients
        """
        if not grads_list:
            return None
        if len(grads_list) == 1:
            return grads_list[0]
        
        from ..util import tree_add, tree_scale

        agg = None
        count = 0
        for grads in grads_list:
            if grads is None:
                continue
            agg = tree_add(agg, grads)
            count += 1
        if agg is None or count == 0:
            return None
        return tree_scale(agg, 1.0 / float(count))
    
    def shutdown(self) -> None:
        """Shutdown the client and its thread pool."""
        self.pool.shutdown(wait=True)


class DistillationTrainingClient(TrainingClient):
    """Extended TrainingClient for knowledge distillation.
    
    Adds support for teacher model logprobs during training,
    enabling distillation from a larger teacher model.
    
    Example:
        >>> student = TrainingClient(student_backend, pool)
        >>> teacher = SamplingClient(teacher_backend_endpoint)
        >>> 
        >>> # Get teacher logprobs for student samples
        >>> samples = student.sample_batch(prompts)
        >>> teacher_logprobs = teacher.get_logprobs_for_texts(prompts, samples.texts)
        >>> 
        >>> # Distillation loss
        >>> batch = TrainingBatch(
        ...     prompts=prompts,
        ...     responses=samples.texts,
        ...     loss_type="distillation",
        ...     extra={"teacher_logprobs": teacher_logprobs}
        ... )
        >>> client.forward_backward(batch)
    """
    
    def __init__(
        self,
        backend: Any,
        pool: Optional[SdkFuturePool] = None,
        optimizer: Optional[Any] = None,
        step: int = 0,
        teacher_sampling_client: Optional['SamplingClient'] = None,  # type: ignore
    ):
        """Initialize DistillationTrainingClient.
        
        Args:
            backend: The LLM backend instance
            pool: Optional SdkFuturePool
            optimizer: Optional pre-created optimizer
            step: Initial training step
            teacher_sampling_client: Optional SamplingClient for teacher model
        """
        super().__init__(backend, pool, optimizer, step)
        self.teacher_client = teacher_sampling_client
    
    def compute_distillation_loss(
        self,
        student_batch: TrainingBatch,
        teacher_logprobs: List[List[Dict[str, float]]],
        temperature: float = 2.0,
    ) -> APIFuture[ForwardBackwardResult]:
        """Compute distillation loss with teacher logprobs.
        
        Args:
            student_batch: Training batch for student
            teacher_logprobs: Top-k logprobs from teacher for each token
            temperature: Distillation temperature
            
        Returns:
            APIFuture resolving to ForwardBackwardResult
        """
        def _run_distillation() -> ForwardBackwardResult:
            # Store teacher logprobs in batch for use during loss computation
            student_batch.extra["teacher_logprobs"] = teacher_logprobs
            student_batch.extra["distillation_temperature"] = temperature
            
            # Fall back to standard forward/backward
            # In practice, the loss function would use teacher_logprobs
            result_future = self.forward_backward(student_batch)
            return result_future.result()
        
        return self.pool.submit(_run_distillation)


# Import at end to avoid circular dependency
from .sampling_client import SamplingClient  # noqa: E402
