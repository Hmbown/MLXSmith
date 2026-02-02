"""MLXSmith SDK for training and inference.

This module provides high-level interfaces for:
- Model loading and sampling
- Training operations (SFT, preference, RL)
- Async futures-based API
- Checkpoint management

Example:
    >>> from mlxsmith.sdk import load_model, TrainingClient, SamplingClient
    >>> 
    >>> # Load model
    >>> loaded = load_model("mlx-community/Llama-3.2-1B-Instruct-4bit", cfg)
    >>> 
    >>> # Create training client
    >>> trainer = TrainingClient(loaded.backend)
    >>> trainer.create_optimizer(lr=1e-4).result()
    >>> 
    >>> # Training loop
    >>> batch = TrainingBatch(prompts=[...], responses=[...])
    >>> result = trainer.forward_backward(batch).result()
    >>> trainer.optim_step(result.grads).result()
    >>> 
    >>> # Sampling
    >>> sampler = SamplingClient(backend=loaded.backend)
    >>> result = sampler.sample("Hello", logprobs_k=5)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from ..config import ProjectConfig
from ..llm.backend import DecodingConfig, Generation
from ..llm.registry import get_llm_backend
from ..models import resolve_model_spec
from ..train.lora import LoRAConfig
from .losses import (
    LOSS_REGISTRY,
    get_loss,
    cross_entropy_loss,
    dpo_loss,
    orpo_loss,
    simpo_loss,
    tdpo_loss,
    kto_loss,
    preference_loss,
    importance_sampling_loss,
    ppo_loss,
    cispo_loss,
    dro_loss,
)
from .future import (
    APIFuture,
    APIFutureState,
    SdkFuturePool,
    completed_future,
    failed_future,
    cancelled_future,
)
from .training_client import (
    TrainingClient,
    TrainingBatch,
    ForwardBackwardResult,
    OptimizerStepResult,
    CheckpointResult,
    WeightsResult,
    DistillationTrainingClient,
)
from .sampling_client import (
    SamplingClient,
    SampleResult,
    SampleBatchResult,
    DistillationSampler,
)


@dataclass
class LoadedModel:
    backend: Any
    base_model: str
    adapter_path: Optional[str]
    adapter_meta: Optional[dict]


def _truncate_ids(ids: Sequence[int], prompt_len: int, max_seq_len: Optional[int]) -> Tuple[List[int], int]:
    if max_seq_len is None:
        return list(ids), prompt_len
    max_len = int(max_seq_len)
    if max_len <= 0:
        return list(ids), prompt_len
    ids_list = list(ids)
    if len(ids_list) <= max_len:
        return ids_list, prompt_len
    overflow = len(ids_list) - max_len
    ids_list = ids_list[overflow:]
    prompt_len = max(0, prompt_len - overflow)
    return ids_list, prompt_len


def load_model(
    model_id_or_path: str,
    cfg: ProjectConfig,
    *,
    apply_lora_if_missing: bool = False,
    lora_config: Optional[LoRAConfig] = None,
) -> LoadedModel:
    """Load a model with the configured backend.
    
    Args:
        model_id_or_path: Model identifier or path
        cfg: Project configuration
        apply_lora_if_missing: Whether to apply LoRA if no adapter found
        lora_config: Optional LoRA configuration
        
    Returns:
        LoadedModel with backend and metadata
    """
    backend = get_llm_backend(cfg.model.backend)
    base_model, adapter_path, adapter_meta = resolve_model_spec(Path.cwd(), model_id_or_path, cfg)
    backend.load(
        base_model,
        max_seq_len=cfg.model.max_seq_len,
        dtype=cfg.model.dtype,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if adapter_path:
        backend.apply_adapter(str(adapter_path))
    elif apply_lora_if_missing:
        if lora_config is None:
            lora_config = LoRAConfig(
                r=cfg.lora.r,
                alpha=cfg.lora.alpha,
                dropout=cfg.lora.dropout,
                target_modules=list(cfg.lora.target_modules or []),
                num_layers=cfg.lora.num_layers,
                scale=cfg.lora.scale,
                fine_tune_type=cfg.lora.fine_tune_type,
            )
        backend.apply_lora_from_config(lora_config)

    return LoadedModel(
        backend=backend,
        base_model=base_model,
        adapter_path=str(adapter_path) if adapter_path else None,
        adapter_meta=adapter_meta,
    )


def sample(backend: Any, prompts: Iterable[str], decoding: DecodingConfig) -> List[Generation]:
    """Sample completions from the backend.
    
    Args:
        backend: LLM backend instance
        prompts: Iterable of prompt strings
        decoding: Decoding configuration
        
    Returns:
        List of Generation results
    """
    results: List[Generation] = []
    for prompt in prompts:
        results.append(
            backend.generate(
                prompt,
                max_new_tokens=decoding.max_new_tokens,
                temperature=decoding.temperature,
                top_p=decoding.top_p,
                top_k=decoding.top_k,
                seed=decoding.seed,
            )
        )
    return results


def logprobs(
    backend: Any,
    prompts: Iterable[str],
    completions: Iterable[str],
    *,
    max_seq_len: Optional[int] = None,
) -> List[Any]:
    """Compute logprobs for prompt-completion pairs.
    
    Args:
        backend: LLM backend instance
        prompts: Iterable of prompt strings
        completions: Iterable of completion strings
        max_seq_len: Maximum sequence length
        
    Returns:
        List of logprob values
    """
    results: List[Any] = []
    for prompt, completion in zip(prompts, completions):
        prompt_ids = backend.encode(prompt)
        ids = backend.encode(prompt + completion)
        ids, prompt_len = _truncate_ids(ids, len(prompt_ids), max_seq_len)
        results.append(backend.sequence_logprob(ids, prompt_len=prompt_len))
    return results


def forward_backward(backend: Any, loss_fn) -> Tuple[Any, Any | None]:
    """Execute forward and backward pass.
    
    Args:
        backend: LLM backend instance
        loss_fn: Loss function
        
    Returns:
        Tuple of (loss, gradients)
    """
    return backend.value_and_grad(loss_fn)


def forward_backward_custom(backend: Any, loss_fn) -> Tuple[Any, Any | None]:
    """Execute custom forward and backward pass."""
    return backend.value_and_grad(loss_fn)


def sft_forward_backward(
    backend: Any,
    prompt: str,
    response: str,
    *,
    train_on_prompt: bool = False,
    max_seq_len: Optional[int] = None,
) -> Tuple[Any, Any | None]:
    """Execute SFT forward/backward pass.
    
    Args:
        backend: LLM backend instance
        prompt: Prompt string
        response: Response string
        train_on_prompt: Whether to compute loss on prompt tokens
        max_seq_len: Maximum sequence length
        
    Returns:
        Tuple of (loss, gradients)
    """
    prompt_ids = backend.encode(prompt)
    ids = backend.encode(prompt + response)
    ids, prompt_len = _truncate_ids(ids, len(prompt_ids), max_seq_len)

    def loss_fn(_model):
        return backend.sft_loss(ids, train_on_prompt=train_on_prompt, prompt_len=prompt_len)

    return backend.value_and_grad(loss_fn)


def preference_forward_backward(
    backend: Any,
    prompt: str,
    chosen: str,
    rejected: str,
    *,
    algo: str = "dpo",
    beta: float = 0.1,
    reference_backend: Optional[Any] = None,
    kl_coeff: float = 0.0,
    train_on_prompt: bool = False,
    max_seq_len: Optional[int] = None,
    delta: float = 0.0,
) -> Tuple[Any, Any | None]:
    """Execute preference-based forward/backward pass.
    
    Args:
        backend: LLM backend instance
        prompt: Prompt string
        chosen: Chosen (preferred) response
        rejected: Rejected response
        algo: Algorithm - "dpo", "orpo", etc.
        beta: Temperature parameter for preference loss
        reference_backend: Optional reference model backend
        kl_coeff: KL divergence coefficient
        train_on_prompt: Whether to compute loss on prompt tokens
        max_seq_len: Maximum sequence length
        
    Returns:
        Tuple of (loss, gradients)
    """
    prompt_ids = backend.encode(prompt)
    chosen_ids = backend.encode(prompt + chosen)
    rejected_ids = backend.encode(prompt + rejected)

    chosen_ids, prompt_len_c = _truncate_ids(chosen_ids, len(prompt_ids), max_seq_len)
    rejected_ids, prompt_len_r = _truncate_ids(rejected_ids, len(prompt_ids), max_seq_len)

    def loss_fn(_model):
        return preference_loss(
            backend,
            chosen_ids,
            rejected_ids,
            prompt_len_chosen=prompt_len_c,
            prompt_len_rejected=prompt_len_r,
            algo=algo,
            beta=beta,
            reference_backend=reference_backend,
            kl_coeff=kl_coeff,
            train_on_prompt=train_on_prompt,
            delta=delta,
        )

    return backend.value_and_grad(loss_fn)


def create_optimizer(
    backend: Any,
    *,
    lr: float,
    weight_decay: float = 0.0,
    optimizer: Optional[str] = None,
    optimizer_kwargs: Optional[dict] = None,
) -> Tuple[Any, Any]:
    """Create optimizer for training.
    
    Args:
        backend: LLM backend instance
        lr: Learning rate
        weight_decay: Weight decay coefficient
        optimizer: Optimizer name
        optimizer_kwargs: Extra optimizer kwargs
        
    Returns:
        Tuple of (optimizer, parameters)
    """
    return backend.optimizer_and_params(
        lr=lr,
        weight_decay=weight_decay,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
    )


def optim_step(backend: Any, optimizer: Any, grads: Any) -> None:
    """Execute optimizer step.
    
    Args:
        backend: LLM backend instance
        optimizer: Optimizer instance
        grads: Gradients
    """
    if grads is None:
        return
    backend.apply_grads(optimizer, grads)


def save_adapter(backend: Any, out_dir: str, *, metadata: Optional[dict] = None) -> None:
    """Save adapter weights.
    
    Args:
        backend: LLM backend instance
        out_dir: Output directory
        metadata: Optional metadata to save
    """
    backend.save_adapter(out_dir, metadata=metadata)


__all__ = [
    # Core classes
    "LoadedModel",
    "load_model",
    
    # Operations
    "sample",
    "logprobs",
    "forward_backward",
    "forward_backward_custom",
    "sft_forward_backward",
    "preference_forward_backward",
    "create_optimizer",
    "optim_step",
    "save_adapter",
    
    # Futures
    "APIFuture",
    "APIFutureState",
    "SdkFuturePool",
    "completed_future",
    "failed_future",
    "cancelled_future",
    
    # Training Client
    "TrainingClient",
    "TrainingBatch",
    "ForwardBackwardResult",
    "OptimizerStepResult",
    "CheckpointResult",
    "WeightsResult",
    "DistillationTrainingClient",
    
    # Sampling Client
    "SamplingClient",
    "SampleResult",
    "SampleBatchResult",
    "DistillationSampler",
    
    # Losses
    "LOSS_REGISTRY",
    "get_loss",
    "cross_entropy_loss",
    "dpo_loss",
    "orpo_loss",
    "simpo_loss",
    "tdpo_loss",
    "kto_loss",
    "preference_loss",
    "importance_sampling_loss",
    "ppo_loss",
    "cispo_loss",
    "dro_loss",
]
