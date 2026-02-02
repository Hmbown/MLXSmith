"""MLXSmith API handlers and schemas."""

from .schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatCompletionChunk,
    RolloutRequest,
    RolloutResponse,
    AdapterReloadRequest,
    AdapterReloadResponse,
    RLMState,
    RLMHistoryEntry,
    ModelInfo,
    ModelsListResponse,
    ModelPullRequest,
    ModelPullResponse,
    HFTokenRequest,
    HFTokenResponse,
    HealthResponse,
    ErrorResponse,
)
from .handlers import create_router, InternalAuthMiddleware

__all__ = [
    # Schemas
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ChatCompletionChunk",
    "RolloutRequest",
    "RolloutResponse",
    "AdapterReloadRequest",
    "AdapterReloadResponse",
    "RLMState",
    "RLMHistoryEntry",
    "ModelInfo",
    "ModelsListResponse",
    "ModelPullRequest",
    "ModelPullResponse",
    "HFTokenRequest",
    "HFTokenResponse",
    "HealthResponse",
    "ErrorResponse",
    # Handlers
    "create_router",
    "InternalAuthMiddleware",
]
