"""Tests for MLXSmith API handlers and schemas."""

import pytest
from pathlib import Path

# Schemas
from mlxsmith.api.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatCompletionChunk,
    RolloutRequest,
    RolloutResponse,
    AdapterReloadRequest,
    AdapterReloadResponse,
    RLMState,
    RLMTrainingMetrics,
    RLMHistoryEntry,
    ModelInfo,
    ModelsListResponse,
    ModelPullRequest,
    ModelPullResponse,
    HFTokenRequest,
    HFTokenResponse,
    HealthResponse,
    ErrorResponse,
    UsageInfo,
    Choice,
    DeltaMessage,
    StreamChoice,
)

# Handlers
from mlxsmith.api.handlers import create_router, InternalAuthMiddleware


class TestSchemas:
    """Test Pydantic schema validation."""

    def test_chat_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.name is None

    def test_chat_request_defaults(self):
        msg = ChatMessage(role="user", content="Hello")
        req = ChatRequest(messages=[msg])
        assert req.max_tokens == 256
        assert req.temperature == 0.7
        assert req.top_p == 1.0
        assert req.stream is False

    def test_chat_request_validation(self):
        msg = ChatMessage(role="user", content="Hello")
        # Valid request
        req = ChatRequest(messages=[msg], max_tokens=100, temperature=0.5)
        assert req.max_tokens == 100
        assert req.temperature == 0.5

    def test_chat_response(self):
        msg = ChatMessage(role="assistant", content="Hi there!")
        resp = ChatResponse(
            id="chatcmpl-123",
            created=1704067200,
            model="test-model",
            choices=[Choice(index=0, message=msg, finish_reason="stop")],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        assert resp.object == "chat.completion"
        assert len(resp.choices) == 1

    def test_chat_completion_chunk(self):
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            created=1704067200,
            model="test-model",
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(content="Hello"),
                    finish_reason=None,
                )
            ],
        )
        assert chunk.object == "chat.completion.chunk"

    def test_rollout_request_defaults(self):
        req = RolloutRequest(prompt="Test prompt")
        assert req.max_tokens == 256
        assert req.temperature == 0.7
        assert req.include_tokens is True
        assert req.include_logprobs is True
        assert req.include_text is True

    def test_rollout_response(self):
        resp = RolloutResponse(
            id="rollout-123",
            created=1704067200,
            model="test-model",
            prompt_len=10,
            token_ids=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.3],
            completion="Test completion",
        )
        assert resp.token_ids == [1, 2, 3]
        assert resp.logprobs == [-0.1, -0.2, -0.3]

    def test_adapter_reload_request(self):
        req = AdapterReloadRequest(adapter_path="./adapters/test")
        assert req.adapter_path == "./adapters/test"
        assert req.reload_base is False

        req2 = AdapterReloadRequest(reload_base=True)
        assert req2.adapter_path is None
        assert req2.reload_base is True

    def test_rlm_state(self):
        state = RLMState(status="running", iteration=42)
        assert state.status == "running"
        assert state.iteration == 42
        assert state.total_iterations is None

    def test_rlm_state_with_metrics(self):
        metrics = RLMTrainingMetrics(loss=0.5, reward_mean=0.8)
        state = RLMState(
            status="running",
            iteration=10,
            metrics=metrics,
        )
        assert state.metrics.loss == 0.5
        assert state.metrics.reward_mean == 0.8

    def test_rlm_history_entry(self):
        entry = RLMHistoryEntry(
            iteration=5,
            timestamp=1704067200,
            adapter_score=0.85,
            base_score=0.75,
            improvement=0.10,
        )
        assert entry.iteration == 5
        assert entry.adapter_score == 0.85

    def test_model_info(self):
        model = ModelInfo(
            id="test-model",
            path="/cache/mlx/test-model",
            format="mlx",
            has_adapter=False,
        )
        assert model.id == "test-model"
        assert model.format == "mlx"

    def test_models_list_response(self):
        model1 = ModelInfo(id="model-1", path="/cache/1", format="mlx", has_adapter=False)
        model2 = ModelInfo(id="model-2", path="/cache/2", format="hf", has_adapter=False)
        resp = ModelsListResponse(
            models=[model1, model2],
            total=2,
            cache_dir="/cache",
        )
        assert resp.total == 2
        assert len(resp.models) == 2

    def test_model_pull_request(self):
        req = ModelPullRequest(model_id="org/model-name")
        assert req.model_id == "org/model-name"
        assert req.convert is True
        assert req.quantize is False
        assert req.q_bits == 4

    def test_hf_token_request(self):
        req = HFTokenRequest(token="hf_test_token_123")
        assert req.token == "hf_test_token_123"
        assert req.persist is True
        assert req.validate_token is True

    def test_hf_token_response(self):
        resp = HFTokenResponse(
            ok=True,
            validated=True,
            username="testuser",
            message="Token stored",
            storage_method="file",
        )
        assert resp.ok is True
        assert resp.username == "testuser"

    def test_health_response(self):
        resp = HealthResponse(ok=True, version="0.1.0", model="test-model")
        assert resp.ok is True
        assert resp.version == "0.1.0"

    def test_error_response(self):
        resp = ErrorResponse(error="Something went wrong", code="ERROR_CODE")
        assert resp.error == "Something went wrong"
        assert resp.code == "ERROR_CODE"


class TestHandlers:
    """Test handler functions (requires FastAPI)."""

    def test_create_router_returns_router(self):
        """Test that create_router returns an APIRouter instance."""
        # This would require mocking the LLM backend
        # For now, just verify the function exists and has correct signature
        import inspect
        sig = inspect.signature(create_router)
        params = list(sig.parameters.keys())
        assert "llm_backend" in params
        assert "base_model" in params
        assert "current_adapter" in params
        assert "cfg" in params


class TestMiddleware:
    """Test authentication middleware."""

    def test_internal_auth_middleware_init(self):
        """Test middleware initialization."""
        from fastapi import FastAPI
        
        app = FastAPI()
        middleware = InternalAuthMiddleware(
            app,
            api_token="test-token",
            internal_prefix="/internal",
            public_paths=["/health"],
        )
        assert middleware.api_token == "test-token"
        assert middleware.internal_prefix == "/internal"
        assert "/health" in middleware.public_paths
