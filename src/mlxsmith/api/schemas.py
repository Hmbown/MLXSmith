"""Pydantic models for API request/response validation.

OpenAPI 3.1 compatible schemas for MLXSmith API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# Common Schemas
# =============================================================================

class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class HealthResponse(BaseModel):
    """Health check response."""
    ok: bool = Field(..., description="Service health status")
    version: Optional[str] = Field(None, description="API version")
    model: Optional[str] = Field(None, description="Currently loaded model")


# =============================================================================
# Chat Completions (OpenAI-compatible)
# =============================================================================

class ChatMessage(BaseModel):
    """A single chat message."""
    role: Literal["system", "user", "assistant", "tool"] = Field(
        ..., description="Role of the message sender"
    )
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Optional name for the sender")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls (if any)")


class ChatRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: Optional[str] = Field(
        None, description="Model identifier (optional, uses default if not provided)"
    )
    messages: List[ChatMessage] = Field(
        ..., description="List of chat messages", min_length=1
    )
    max_tokens: int = Field(
        256, description="Maximum tokens to generate", ge=1, le=8192
    )
    temperature: float = Field(
        0.7, description="Sampling temperature", ge=0.0, le=2.0
    )
    top_p: float = Field(
        1.0, description="Nucleus sampling parameter", ge=0.0, le=1.0
    )
    top_k: Optional[int] = Field(
        None, description="Top-k sampling parameter", ge=1
    )
    stream: Optional[bool] = Field(
        False, description="Enable streaming response via SSE"
    )
    stop: Optional[Union[str, List[str]]] = Field(
        None, description="Stop sequences"
    )
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    presence_penalty: Optional[float] = Field(
        0.0, description="Presence penalty", ge=-2.0, le=2.0
    )
    frequency_penalty: Optional[float] = Field(
        0.0, description="Frequency penalty", ge=-2.0, le=2.0
    )
    logprobs: Optional[bool] = Field(
        False, description="Return logprobs of output tokens"
    )
    top_logprobs: Optional[int] = Field(
        None, description="Number of top logprobs to return per token", ge=0, le=20
    )


class LogprobsContent(BaseModel):
    """Logprob information for a token."""
    token: str = Field(..., description="The token string")
    logprob: float = Field(..., description="The log probability of the token")
    bytes: Optional[List[int]] = Field(None, description="Bytes representation of token")
    top_logprobs: Optional[List[Dict[str, float]]] = Field(
        None, description="Top logprobs for this position"
    )


class ChoiceLogprobs(BaseModel):
    """Logprobs for a completion choice."""
    content: Optional[List[LogprobsContent]] = Field(
        None, description="Logprobs for each token in the completion"
    )


class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens")


class Choice(BaseModel):
    """A single completion choice."""
    index: int = Field(..., description="Index of the choice")
    message: ChatMessage = Field(..., description="The generated message")
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = Field(
        None, description="Reason for completion finish"
    )
    logprobs: Optional[ChoiceLogprobs] = Field(None, description="Logprobs for this choice")


class ChatResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(..., description="Unique identifier for the completion")
    object: Literal["chat.completion"] = Field("chat.completion")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for the completion")
    choices: List[Choice] = Field(..., description="List of completion choices")
    usage: UsageInfo = Field(..., description="Token usage information")


class DeltaMessage(BaseModel):
    """Delta message for streaming responses."""
    role: Optional[Literal["assistant"]] = Field(None)
    content: Optional[str] = Field(None, description="Incremental content")


class StreamChoice(BaseModel):
    """A streaming completion choice."""
    index: int = Field(..., description="Index of the choice")
    delta: DeltaMessage = Field(..., description="Incremental message delta")
    finish_reason: Optional[Literal["stop", "length"]] = Field(None)
    logprobs: Optional[ChoiceLogprobs] = Field(None, description="Logprobs for this chunk")


class ChatCompletionChunk(BaseModel):
    """Streaming chat completion chunk (SSE)."""
    id: str = Field(..., description="Unique identifier")
    object: Literal["chat.completion.chunk"] = Field("chat.completion.chunk")
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used")
    choices: List[StreamChoice] = Field(..., description="List of choices")


# =============================================================================
# Internal Rollout (for RLM training)
# =============================================================================

class RolloutRequest(BaseModel):
    """Internal rollout request with detailed output options."""
    prompt: str = Field(..., description="Input prompt text", min_length=1)
    max_tokens: int = Field(256, description="Maximum tokens to generate", ge=1)
    temperature: float = Field(0.7, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: float = Field(1.0, description="Nucleus sampling parameter", ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, description="Top-k sampling", ge=1)
    seed: Optional[int] = Field(None, description="Random seed")
    include_tokens: bool = Field(True, description="Include token IDs in response")
    include_logprobs: bool = Field(True, description="Include per-token logprobs")
    include_top_k_logprobs: Optional[int] = Field(
        None, description="Number of top logprobs per token to include", ge=0, le=20
    )
    include_prompt_logprobs: bool = Field(
        False, description="Include per-token logprobs for prompt tokens"
    )
    include_prompt_top_k_logprobs: Optional[int] = Field(
        None,
        description="Number of top logprobs per prompt token to include",
        ge=0,
        le=20,
    )
    include_text: bool = Field(True, description="Include generated text")
    recursive: Optional[bool] = Field(
        None, description="Override recursive inference for this rollout"
    )


class RolloutResponse(BaseModel):
    """Internal rollout response with tokens and logprobs."""
    id: str = Field(..., description="Unique rollout identifier")
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used")
    prompt_len: int = Field(..., description="Length of prompt in tokens")
    token_ids: Optional[List[int]] = Field(None, description="Generated token IDs")
    logprobs: Optional[List[float]] = Field(None, description="Per-token log probabilities")
    top_k_logprobs: Optional[List[Dict[str, float]]] = Field(
        None, description="Top-k logprobs per token"
    )
    prompt_logprobs: Optional[List[float]] = Field(
        None,
        description="Per-token log probabilities for prompt tokens (excluding first token)",
    )
    prompt_top_k_logprobs: Optional[List[Dict[str, float]]] = Field(
        None, description="Top-k logprobs per prompt token"
    )
    completion: Optional[str] = Field(None, description="Generated text (if requested)")
    prompt_used: Optional[str] = Field(None, description="Prompt actually used for generation")
    recursion_depth: Optional[int] = Field(None, description="Recursive compaction depth")
    recursion_chunks: Optional[int] = Field(None, description="Number of chunks summarized")
    recursion_truncated: Optional[bool] = Field(None, description="Whether recursion truncated")


# =============================================================================
# Training Endpoints
# =============================================================================

class ForwardBackwardRequest(BaseModel):
    """Request for forward/backward pass."""
    prompts: List[str] = Field(..., description="List of prompts", min_length=1)
    responses: Optional[List[str]] = Field(None, description="List of responses (for SFT)")
    rejected_responses: Optional[List[str]] = Field(
        None, description="List of rejected responses (for preference training)"
    )
    loss_type: Literal["sft", "dpo", "orpo", "cpo", "ipo", "hinge", "simpo", "tdpo", "ppo", "custom"] = Field(
        "sft", description="Type of loss to compute"
    )
    train_on_prompt: bool = Field(False, description="Compute loss on prompt tokens")
    max_seq_len: Optional[int] = Field(None, description="Maximum sequence length")
    extra: Optional[Dict[str, Any]] = Field(None, description="Additional loss parameters")


class ForwardBackwardResponse(BaseModel):
    """Response from forward/backward pass."""
    loss: float = Field(..., description="Computed loss value")
    has_grads: bool = Field(..., description="Whether gradients were computed")
    batch_size: int = Field(..., description="Batch size processed")
    metrics: Optional[Dict[str, float]] = Field(None, description="Additional metrics")


class OptimStepRequest(BaseModel):
    """Request for optimizer step."""
    learning_rate: Optional[float] = Field(None, description="Override learning rate")
    grad_clip: Optional[float] = Field(None, description="Gradient clipping threshold")


class OptimStepResponse(BaseModel):
    """Response from optimizer step."""
    step: int = Field(..., description="Current training step")
    learning_rate: float = Field(..., description="Learning rate used")
    grad_norm: Optional[float] = Field(None, description="Gradient norm")
    success: bool = Field(True, description="Whether step succeeded")


class SaveStateRequest(BaseModel):
    """Request to save training state."""
    path: str = Field(..., description="Path to save checkpoint")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata to save")


class SaveStateResponse(BaseModel):
    """Response from save state operation."""
    path: str = Field(..., description="Path where checkpoint was saved")
    success: bool = Field(..., description="Whether save succeeded")
    message: str = Field(..., description="Status message")


class LoadStateRequest(BaseModel):
    """Request to load training state."""
    path: str = Field(..., description="Path to checkpoint to load")


class LoadStateResponse(BaseModel):
    """Response from load state operation."""
    path: str = Field(..., description="Path from which checkpoint was loaded")
    success: bool = Field(..., description="Whether load succeeded")
    message: str = Field(..., description="Status message")
    step: Optional[int] = Field(None, description="Training step from checkpoint")


class GetWeightsResponse(BaseModel):
    """Response for get weights operation."""
    weights: Dict[str, Any] = Field(..., description="Model weights (may be partial/shape info)")
    success: bool = Field(..., description="Whether operation succeeded")
    message: str = Field(..., description="Status message")


class SetWeightsRequest(BaseModel):
    """Request to set model weights."""
    weights: Dict[str, Any] = Field(..., description="Model weights to set")


class SetWeightsResponse(BaseModel):
    """Response from set weights operation."""
    success: bool = Field(..., description="Whether operation succeeded")
    message: str = Field(..., description="Status message")
    num_tensors: int = Field(..., description="Number of weight tensors set")


# =============================================================================
# Adapter Management
# =============================================================================

class AdapterReloadRequest(BaseModel):
    """Request to reload adapter weights."""
    adapter_path: Optional[str] = Field(
        None, description="Path to adapter directory (relative or absolute)"
    )
    reload_base: bool = Field(
        False, description="Reload the base model before applying adapter"
    )


class AdapterReloadResponse(BaseModel):
    """Response after adapter reload."""
    ok: bool = Field(..., description="Whether reload was successful")
    base_model: str = Field(..., description="Base model identifier")
    adapter_path: Optional[str] = Field(None, description="Currently loaded adapter path")
    message: Optional[str] = Field(None, description="Status message")


# =============================================================================
# RLM State and History
# =============================================================================

class RLMTrainingMetrics(BaseModel):
    """RLM training metrics."""
    loss: Optional[float] = Field(None, description="Training loss")
    reward_mean: Optional[float] = Field(None, description="Mean reward")
    reward_std: Optional[float] = Field(None, description="Reward standard deviation")
    kl_div: Optional[float] = Field(None, description="KL divergence from reference")
    learning_rate: Optional[float] = Field(None, description="Current learning rate")


class RLMState(BaseModel):
    """Current RLM training state."""
    status: Literal["idle", "running", "paused", "completed", "error"] = Field(
        ..., description="Current training status"
    )
    iteration: Optional[int] = Field(None, description="Current training iteration")
    total_iterations: Optional[int] = Field(None, description="Total planned iterations")
    metrics: Optional[RLMTrainingMetrics] = Field(None, description="Current metrics")
    started_at: Optional[int] = Field(None, description="Training start timestamp")
    updated_at: Optional[int] = Field(None, description="Last update timestamp")
    error_message: Optional[str] = Field(None, description="Error message if status is error")


class RLMHistoryEntry(BaseModel):
    """Single RLM training history entry."""
    iteration: int = Field(..., description="Training iteration number")
    timestamp: int = Field(..., description="Unix timestamp")
    adapter_score: Optional[float] = Field(None, description="Adapter evaluation score")
    base_score: Optional[float] = Field(None, description="Base model score")
    improvement: Optional[float] = Field(None, description="Relative improvement")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Additional metrics")


# =============================================================================
# Model Management
# =============================================================================

class ModelInfo(BaseModel):
    """Information about a cached model."""
    id: str = Field(..., description="Model identifier")
    path: str = Field(..., description="Local path to the model")
    size_bytes: Optional[int] = Field(None, description="Model size in bytes")
    format: Literal["mlx", "hf", "gguf"] = Field(..., description="Model format")
    has_adapter: bool = Field(False, description="Whether model has adapter weights")
    adapter_path: Optional[str] = Field(None, description="Path to adapter if present")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional model metadata")
    downloaded_at: Optional[int] = Field(None, description="Download timestamp")


class ModelsListResponse(BaseModel):
    """Response for listing cached models."""
    models: List[ModelInfo] = Field(..., description="List of cached models")
    total: int = Field(..., description="Total number of models")
    cache_dir: str = Field(..., description="Current cache directory")


class ModelPullRequest(BaseModel):
    """Request to pull a model from HuggingFace."""
    model_id: str = Field(..., description="HuggingFace model identifier", min_length=1)
    convert: bool = Field(True, description="Convert to MLX format")
    quantize: bool = Field(False, description="Quantize during conversion")
    q_bits: Optional[int] = Field(4, description="Quantization bits", ge=1, le=8)
    q_group_size: Optional[int] = Field(64, description="Quantization group size")
    trust_remote_code: bool = Field(False, description="Trust remote code in model")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_id": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                "convert": True,
                "quantize": False,
            }
        }
    )


class ModelPullStatus(BaseModel):
    """Status of model pull operation."""
    status: Literal["pending", "downloading", "converting", "completed", "error"] = Field(
        ..., description="Current pull status"
    )
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)", ge=0, le=100)
    message: Optional[str] = Field(None, description="Status message")
    downloaded_bytes: Optional[int] = Field(None, description="Bytes downloaded so far")
    total_bytes: Optional[int] = Field(None, description="Total bytes to download")


class ModelPullResponse(BaseModel):
    """Response for model pull request."""
    ok: bool = Field(..., description="Whether pull was initiated successfully")
    model_id: str = Field(..., description="Model identifier")
    local_path: Optional[str] = Field(None, description="Local path where model will be stored")
    status: ModelPullStatus = Field(..., description="Current pull status")
    message: Optional[str] = Field(None, description="Status message")


class ModelDeleteResponse(BaseModel):
    """Response for model deletion."""
    ok: bool = Field(..., description="Whether delete succeeded")
    model_id: str = Field(..., description="Model identifier")
    message: Optional[str] = Field(None, description="Status message")


# =============================================================================
# HuggingFace Token Management
# =============================================================================

class HFTokenRequest(BaseModel):
    """Request to store HuggingFace token."""
    token: str = Field(
        ..., 
        description="HuggingFace API token",
        min_length=1,
        json_schema_extra={"format": "password"}
    )
    persist: bool = Field(
        True, description="Persist token to disk (encrypted if possible)"
    )
    validate_token: bool = Field(
        True, description="Validate token before storing"
    )


class HFTokenResponse(BaseModel):
    """Response after storing HF token."""
    ok: bool = Field(..., description="Whether token was stored successfully")
    validated: bool = Field(..., description="Whether token was validated")
    username: Optional[str] = Field(None, description="HF username if validated")
    message: str = Field(..., description="Status message")
    storage_method: Literal["keyring", "file", "memory"] = Field(
        ..., description="How the token is stored"
    )
