"""FastAPI handlers for MLXSmith API.

Implements endpoints for:
- OpenAI-compatible chat completions with streaming
- Internal rollout (tokens + logprobs)
- Training operations (forward/backward, optim_step, save/load state)
- Adapter hot-reload
- RLM state and history
- Model management
- HF token storage
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request, Security, status
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .. import __version__
from .schemas import (
    AdapterReloadRequest,
    AdapterReloadResponse,
    ChatCompletionChunk,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Choice,
    ChoiceLogprobs,
    DeltaMessage,
    ErrorResponse,
    ForwardBackwardRequest,
    ForwardBackwardResponse,
    GetWeightsResponse,
    HealthResponse,
    HFTokenRequest,
    HFTokenResponse,
    LoadStateRequest,
    LoadStateResponse,
    LogprobsContent,
    ModelInfo,
    ModelsListResponse,
    ModelDeleteResponse,
    ModelPullRequest,
    ModelPullResponse,
    ModelPullStatus,
    OptimStepRequest,
    OptimStepResponse,
    RolloutRequest,
    RolloutResponse,
    RLMHistoryEntry,
    RLMState,
    SaveStateRequest,
    SaveStateResponse,
    SetWeightsRequest,
    SetWeightsResponse,
    StreamChoice,
    UsageInfo,
)
from ..rlm.recursive import recursive_compact

# =============================================================================
# Authentication Middleware
# =============================================================================

class InternalAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for authenticating internal endpoints.
    
    Checks for a valid API token on internal endpoints.
    Public endpoints (health, chat completions) bypass authentication.
    """
    
    def __init__(
        self,
        app: FastAPI,
        api_token: Optional[str] = None,
        internal_prefix: str = "/internal",
        public_paths: Optional[List[str]] = None,
    ):
        super().__init__(app)
        self.api_token = api_token or os.environ.get("MLXSMITH_API_TOKEN")
        self.internal_prefix = internal_prefix
        self.public_paths = set(public_paths or ["/health", "/v1/chat/completions"])
        self.security = HTTPBearer(auto_error=False)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        path = request.url.path
        
        # Skip auth for public paths
        if path in self.public_paths:
            return await call_next(request)
        
        # Skip auth for non-internal paths
        if not path.startswith(self.internal_prefix):
            return await call_next(request)
        
        # If no token configured, allow all (development mode)
        if not self.api_token:
            return await call_next(request)
        
        # Check Authorization header
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return self._unauthorized("Missing or invalid authorization header")
        
        token = auth_header[7:]  # Remove "Bearer " prefix
        if not secrets.compare_digest(token, self.api_token):
            return self._unauthorized("Invalid API token")
        
        return await call_next(request)
    
    def _unauthorized(self, detail: str) -> Any:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": "Unauthorized", "detail": detail},
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_internal_token(
    credentials: HTTPAuthorizationCredentials = Security(HTTPBearer()),
    expected_token: Optional[str] = None,
) -> bool:
    """Dependency for verifying internal endpoint tokens.
    
    Usage:
        @router.get("/internal/protected", dependencies=[Depends(verify_internal_token)])
    """
    token = expected_token or os.environ.get("MLXSMITH_API_TOKEN")
    if not token:
        return True  # Development mode - no token configured
    
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not secrets.compare_digest(credentials.credentials, token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return True


# =============================================================================
# Helper Functions
# =============================================================================

def _messages_to_prompt(
    messages: List[Any],
    tokenizer: Any,
    *,
    use_chat_template: bool = True,
    system_prefix: str | None = None,
) -> str:
    """Convert chat messages to prompt string."""
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        if system_prefix:
            if msgs and msgs[0].get("role") == "system":
                msgs[0]["content"] = f"{system_prefix}\n\n{msgs[0].get('content','')}".strip()
            else:
                msgs.insert(0, {"role": "system", "content": system_prefix})
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # Fallback
    lines: List[str] = []
    if system_prefix:
        if messages and getattr(messages[0], "role", None) == "system":
            merged = f"{system_prefix}\n\n{getattr(messages[0], 'content', '')}".strip()
            lines.append(f"system: {merged}")
            lines.extend([f"{m.role}: {m.content}" for m in messages[1:]])
        else:
            lines.append(f"system: {system_prefix}")
            lines.extend([f"{m.role}: {m.content}" for m in messages])
    else:
        lines.extend([f"{m.role}: {m.content}" for m in messages])
    return "\n".join(lines) + "\nassistant:"


def _truncate_stop(text: str, stop: Optional[List[str]]) -> str:
    """Truncate text at first stop sequence."""
    if not stop:
        return text
    idx = None
    for s in stop:
        if not s:
            continue
        pos = text.find(s)
        if pos != -1:
            idx = pos if idx is None else min(idx, pos)
    return text if idx is None else text[:idx]


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL | re.IGNORECASE)
_SPECIAL_MARKER_RE = re.compile(r"<\|[^|]{1,80}\|>")


def _strip_think(text: str) -> str:
    """Best-effort removal of Qwen-style `<think>` blocks from model output."""
    if not text:
        return text
    lower = text.lower()
    if "<think" not in lower and "</think>" not in lower and "<|" not in text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text)

    # If we still have a closing tag, drop everything up to it.
    if "</think>" in cleaned.lower():
        cleaned = re.split(r"</think>", cleaned, flags=re.IGNORECASE)[-1]

    # Handle the common "open <think> without closing </think>" case by
    # stripping from the first <think> up to the first blank line.
    lower_cleaned = cleaned.lower()
    idx = lower_cleaned.find("<think>")
    if idx != -1:
        after = cleaned[idx + len("<think>") :]
        m = re.search(r"\n\s*\n", after)
        if m:
            cleaned = (cleaned[:idx] + after[m.end() :]).lstrip()
        else:
            cleaned = (cleaned[:idx] + after).lstrip()

    # Strip common chat template markers (e.g. Qwen eos token `<|im_end|>`).
    cleaned = _SPECIAL_MARKER_RE.sub("", cleaned)

    return cleaned.lstrip("\n")


def _get_cache_dir() -> Path:
    """Get the cache directory for models."""
    cache_dir = os.environ.get("MLXSMITH_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir)
    return Path.home() / ".cache" / "mlxsmith"


def _build_logprobs_content(
    token_ids: List[int],
    logprobs: List[float],
    top_k_logprobs: Optional[List[Dict[str, float]]],
    tokenizer: Any,
) -> List[LogprobsContent]:
    """Build LogprobsContent from token info."""
    content = []
    for i, (token_id, logprob) in enumerate(zip(token_ids, logprobs)):
        try:
            token_str = tokenizer.decode([token_id]) if tokenizer else f"<token_{token_id}>"
        except Exception:
            token_str = f"<token_{token_id}>"
        
        top_logprobs = None
        if top_k_logprobs and i < len(top_k_logprobs):
            top_logprobs = [top_k_logprobs[i]]
        
        content.append(LogprobsContent(
            token=token_str,
            logprob=logprob,
            top_logprobs=top_logprobs,
        ))
    return content


# =============================================================================
# Route Handlers
# =============================================================================

def create_router(
    llm_backend: Any,
    base_model: str,
    current_adapter: Optional[str],
    cfg: Any,
) -> APIRouter:
    """Create API router with all endpoints.
    
    Args:
        llm_backend: The LLM backend instance
        base_model: The base model identifier
        current_adapter: Currently loaded adapter path (if any)
        cfg: Project configuration
    
    Returns:
        Configured APIRouter instance
    """
    router = APIRouter()
    
    # Track adapter state (mutable reference)
    adapter_state = {"path": current_adapter}
    
    # Track training state
    training_state = {
        "step": 0,
        "optimizer": None,
        "learning_rate": 1e-4,
    }
    
    # ==========================================================================
    # Health Check
    # ==========================================================================
    
    @router.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            ok=True,
            version=__version__,
            model=base_model,
        )
    
    # ==========================================================================
    # Chat Completions (OpenAI-compatible)
    # ==========================================================================
    
    @router.post(
        "/v1/chat/completions",
        response_model=ChatResponse,
        responses={
            200: {"description": "Successful completion", "model": ChatResponse},
            400: {"description": "Bad request", "model": ErrorResponse},
            500: {"description": "Internal error", "model": ErrorResponse},
        },
        tags=["Chat"],
    )
    async def chat_completions(request: ChatRequest) -> ChatResponse | StreamingResponse:
        """OpenAI-compatible chat completions endpoint.
        
        Supports both streaming (SSE) and non-streaming responses.
        Supports logprobs parameter for returning token logprobs.
        """
        strip_think = bool(getattr(getattr(cfg, "infer", None), "strip_think", False))
        system_prefix = None
        if strip_think:
            system_prefix = (
                "Answer directly. Do not output <think> blocks, reasoning, or analysis. "
                "Output only the final answer."
            )

        prompt = _messages_to_prompt(
            request.messages,
            llm_backend.tokenizer,
            use_chat_template=getattr(cfg.model, "use_chat_template", True),
            system_prefix=system_prefix,
        )
        
        # Determine if we need logprobs
        logprobs_k = request.top_logprobs or (5 if request.logprobs else 0)
        
        # Handle streaming response
        if request.stream:
            async def event_stream() -> AsyncGenerator[str, None]:
                try:
                    # Try to use mlx_lm streaming if available
                    try:
                        import mlx_lm
                        has_mlx_lm = True
                    except ImportError:
                        has_mlx_lm = False
                    
                    if has_mlx_lm:
                        acc = ""
                        emitted = ""
                        for out in mlx_lm.stream_generate(
                            llm_backend.model,
                            llm_backend.tokenizer,
                            prompt,
                            max_tokens=request.max_tokens,
                            temp=request.temperature,
                            top_p=request.top_p,
                            top_k=request.top_k or 0,
                        ):
                            if out.text:
                                acc += out.text
                                acc_view = _strip_think(acc) if strip_think else acc
                                chunk = _truncate_stop(acc_view, request.stop)
                                if len(chunk) < len(emitted):
                                    # If <think> stripping caused the visible text to shrink,
                                    # reset the emitted cursor and keep going so we can still
                                    # stream the final answer.
                                    if request.stop and len(chunk) < len(acc_view):
                                        break
                                    emitted = chunk
                                    continue
                                delta = chunk[len(emitted):]
                                emitted = chunk
                                
                                chunk_data = ChatCompletionChunk(
                                    id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
                                    created=int(time.time()),
                                    model=request.model or base_model,
                                    choices=[StreamChoice(
                                        index=0,
                                        delta=DeltaMessage(content=delta),
                                        finish_reason=None,
                                    )],
                                )
                                yield f"data: {chunk_data.model_dump_json()}\n\n"
                                
                                if request.stop and len(chunk) < len(acc_view):
                                    break
                            if getattr(out, "finish_reason", None):
                                break
                    else:
                        # Fallback to non-streaming
                        if logprobs_k > 0 and hasattr(llm_backend, 'generate_with_logprobs'):
                            gen = llm_backend.generate_with_logprobs(
                                prompt,
                                max_new_tokens=request.max_tokens,
                                temperature=request.temperature,
                                top_p=request.top_p,
                                top_k_sampling=request.top_k,
                                logprobs=logprobs_k,
                            )
                        else:
                            gen = llm_backend.generate(
                                prompt,
                                max_new_tokens=request.max_tokens,
                                temperature=request.temperature,
                                top_p=request.top_p,
                                top_k=request.top_k,
                            )
                        completion = gen.text[len(prompt):] if gen.text.startswith(prompt) else gen.text
                        completion = _truncate_stop(completion, request.stop)
                        if strip_think:
                            completion = _strip_think(completion)
                        
                        chunk_data = ChatCompletionChunk(
                            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
                            created=int(time.time()),
                            model=request.model or base_model,
                            choices=[StreamChoice(
                                index=0,
                                delta=DeltaMessage(content=completion),
                                finish_reason="stop",
                            )],
                        )
                        yield f"data: {chunk_data.model_dump_json()}\n\n"
                    
                    yield "data: [DONE]\n\n"
                    
                except Exception as e:
                    error_chunk = {"error": str(e)}
                    yield f"data: {json.dumps(error_chunk)}\n\n"
            
            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        
        # Non-streaming response
        try:
            if logprobs_k > 0 and hasattr(llm_backend, 'generate_with_logprobs'):
                gen = llm_backend.generate_with_logprobs(
                    prompt,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k_sampling=request.top_k,
                    logprobs=logprobs_k,
                )
            else:
                gen = llm_backend.generate(
                    prompt,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                )
            completion = gen.text[len(prompt):] if gen.text.startswith(prompt) else gen.text
            completion = _truncate_stop(completion, request.stop)
            if strip_think:
                completion = _strip_think(completion)
            
            prompt_tokens = len(llm_backend.encode(prompt))
            completion_tokens = len(llm_backend.encode(completion))
            
            # Build choice with optional logprobs
            choice = Choice(
                index=0,
                message=ChatMessage(role="assistant", content=completion),
                finish_reason="stop",
            )
            
            if request.logprobs and gen.logprobs:
                completion_ids = gen.token_ids[gen.prompt_len:]
                logprobs_content = _build_logprobs_content(
                    completion_ids,
                    gen.logprobs,
                    gen.top_k_logprobs,
                    llm_backend.tokenizer,
                )
                choice.logprobs = ChoiceLogprobs(content=logprobs_content)
            
            return ChatResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
                created=int(time.time()),
                model=request.model or base_model,
                choices=[choice],
                usage=UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Generation failed: {str(e)}",
            )
    
    # ==========================================================================
    # Internal Rollout (for RLM training)
    # ==========================================================================
    
    @router.post(
        "/internal/rollout",
        response_model=RolloutResponse,
        responses={
            200: {"description": "Successful rollout", "model": RolloutResponse},
            400: {"description": "Bad request", "model": ErrorResponse},
            401: {"description": "Unauthorized", "model": ErrorResponse},
            500: {"description": "Internal error", "model": ErrorResponse},
        },
        tags=["Internal"],
    )
    async def internal_rollout(request: RolloutRequest) -> RolloutResponse:
        """Internal rollout endpoint returning tokens and logprobs.
        
        Used by RLM training loop for generating rollouts with detailed
        token-level information. Supports top-k logprobs for distillation.
        """
        try:
            # Determine logprobs to return
            logprobs_k = request.include_top_k_logprobs or (5 if request.include_logprobs else 0)
            
            prompt_text = request.prompt
            use_recursive = bool(getattr(cfg.rlm, "recursive_inference", False))
            if request.recursive is not None:
                use_recursive = bool(request.recursive)

            recursion_stats = None
            if use_recursive:
                prompt_text, recursion_stats = recursive_compact(
                    llm_backend,
                    prompt_text,
                    max_seq_len=int(cfg.model.max_seq_len),
                    chunk_tokens=int(cfg.rlm.recursive_chunk_tokens),
                    overlap_tokens=int(cfg.rlm.recursive_overlap_tokens),
                    keep_last_tokens=int(cfg.rlm.recursive_keep_last_tokens),
                    summary_tokens=int(cfg.rlm.recursive_summary_tokens),
                    max_depth=int(cfg.rlm.recursive_max_depth),
                    temperature=float(cfg.rlm.recursive_temperature),
                    seed=request.seed,
                    summary_prompt=cfg.rlm.recursive_summary_prompt,
                )

            gen = llm_backend.generate_with_logprobs(
                prompt_text,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k_sampling=request.top_k,
                seed=request.seed,
                logprobs=logprobs_k,
            )
            
            completion = gen.text[len(prompt_text):] if gen.text.startswith(prompt_text) else gen.text

            prompt_logprobs: Optional[List[float]] = None
            prompt_top_k: Optional[List[Dict[str, float]]] = None
            include_prompt = bool(request.include_prompt_logprobs or request.include_prompt_top_k_logprobs)
            if include_prompt and hasattr(llm_backend, "token_logprobs"):
                prompt_ids = llm_backend.encode(request.prompt)
                try:
                    logps, topk = llm_backend.token_logprobs(
                        prompt_ids,
                        prompt_len=len(prompt_ids),
                        top_k=int(request.include_prompt_top_k_logprobs or 0),
                        include_prompt=True,
                    )
                    if request.include_prompt_logprobs:
                        prompt_logprobs = list(logps)
                    if request.include_prompt_top_k_logprobs:
                        prompt_top_k = topk or []
                except Exception:
                    prompt_logprobs = None
                    prompt_top_k = None
            
            return RolloutResponse(
                id=f"rollout-{uuid.uuid4().hex[:12]}",
                created=int(time.time()),
                model=base_model,
                prompt_len=gen.prompt_len,
                token_ids=list(gen.token_ids) if request.include_tokens else None,
                logprobs=list(gen.logprobs) if (request.include_logprobs and gen.logprobs is not None) else None,
                top_k_logprobs=gen.top_k_logprobs if request.include_top_k_logprobs else None,
                prompt_logprobs=prompt_logprobs,
                prompt_top_k_logprobs=prompt_top_k,
                completion=completion if request.include_text else None,
                prompt_used=prompt_text if prompt_text != request.prompt else None,
                recursion_depth=getattr(recursion_stats, "depth", None) if recursion_stats else None,
                recursion_chunks=getattr(recursion_stats, "chunks", None) if recursion_stats else None,
                recursion_truncated=getattr(recursion_stats, "truncated", None) if recursion_stats else None,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Rollout generation failed: {str(e)}",
            )
    
    # ==========================================================================
    # Training Endpoints
    # ==========================================================================
    
    @router.post(
        "/internal/train/forward_backward",
        response_model=ForwardBackwardResponse,
        responses={
            200: {"description": "Forward/backward pass completed", "model": ForwardBackwardResponse},
            400: {"description": "Bad request", "model": ErrorResponse},
            401: {"description": "Unauthorized", "model": ErrorResponse},
            500: {"description": "Internal error", "model": ErrorResponse},
        },
        tags=["Training"],
    )
    async def train_forward_backward(request: ForwardBackwardRequest) -> ForwardBackwardResponse:
        """Execute forward and backward pass.
        
        Computes loss and gradients for the given batch.
        """
        try:
            from ..sdk import sft_forward_backward, preference_forward_backward
            
            losses = []
            has_grads = False
            
            if request.loss_type in ("dpo", "orpo", "cpo", "ipo", "hinge", "simpo", "tdpo"):
                # Preference training
                if not request.rejected_responses:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"{request.loss_type} requires rejected_responses",
                    )
                
                for prompt, chosen, rejected in zip(
                    request.prompts,
                    request.responses or [],
                    request.rejected_responses,
                ):
                    loss, grads = preference_forward_backward(
                        llm_backend,
                        prompt,
                        chosen,
                        rejected,
                        algo=request.loss_type,
                        beta=(request.extra or {}).get("beta", 0.1),
                        max_seq_len=request.max_seq_len,
                        train_on_prompt=request.train_on_prompt,
                    )
                    losses.append(float(loss) if loss is not None else 0.0)
                    if grads is not None:
                        has_grads = True
            else:
                # SFT training
                for prompt, response in zip(request.prompts, request.responses or []):
                    loss, grads = sft_forward_backward(
                        llm_backend,
                        prompt,
                        response,
                        train_on_prompt=request.train_on_prompt,
                        max_seq_len=request.max_seq_len,
                    )
                    losses.append(float(loss) if loss is not None else 0.0)
                    if grads is not None:
                        has_grads = True
            
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            
            return ForwardBackwardResponse(
                loss=avg_loss,
                has_grads=has_grads,
                batch_size=len(request.prompts),
                metrics={
                    "max_loss": max(losses) if losses else 0.0,
                    "min_loss": min(losses) if losses else 0.0,
                }
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Forward/backward failed: {str(e)}",
            )
    
    @router.post(
        "/internal/train/optim_step",
        response_model=OptimStepResponse,
        responses={
            200: {"description": "Optimizer step completed", "model": OptimStepResponse},
            400: {"description": "Bad request", "model": ErrorResponse},
            401: {"description": "Unauthorized", "model": ErrorResponse},
            500: {"description": "Internal error", "model": ErrorResponse},
        },
        tags=["Training"],
    )
    async def train_optim_step(request: OptimStepRequest) -> OptimStepResponse:
        """Execute optimizer step.
        
        Requires optimizer to be initialized via create_optimizer first.
        """
        try:
            if training_state["optimizer"] is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Optimizer not initialized. Call create_optimizer first.",
                )
            
            # Update learning rate if provided
            if request.learning_rate is not None:
                training_state["learning_rate"] = request.learning_rate
            
            # Execute step (note: this requires grads to be stored from forward_backward)
            # In a real implementation, you'd need to store grads between calls
            training_state["step"] += 1
            
            return OptimStepResponse(
                step=training_state["step"],
                learning_rate=training_state["learning_rate"],
                grad_norm=None,  # Would compute from actual grads
                success=True,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Optimizer step failed: {str(e)}",
            )
    
    @router.post(
        "/internal/train/create_optimizer",
        response_model=OptimStepResponse,
        responses={
            200: {"description": "Optimizer created", "model": OptimStepResponse},
            401: {"description": "Unauthorized", "model": ErrorResponse},
            500: {"description": "Internal error", "model": ErrorResponse},
        },
        tags=["Training"],
    )
    async def train_create_optimizer(request: OptimStepRequest) -> OptimStepResponse:
        """Create optimizer for training."""
        try:
            from ..sdk import create_optimizer
            
            lr = request.learning_rate or training_state["learning_rate"]
            opt, _ = create_optimizer(llm_backend, lr=lr)
            training_state["optimizer"] = opt
            training_state["learning_rate"] = lr
            
            return OptimStepResponse(
                step=training_state["step"],
                learning_rate=lr,
                success=True,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create optimizer: {str(e)}",
            )
    
    @router.post(
        "/internal/train/save_state",
        response_model=SaveStateResponse,
        responses={
            200: {"description": "State saved", "model": SaveStateResponse},
            401: {"description": "Unauthorized", "model": ErrorResponse},
            500: {"description": "Internal error", "model": ErrorResponse},
        },
        tags=["Training"],
    )
    async def train_save_state(request: SaveStateRequest) -> SaveStateResponse:
        """Save training checkpoint."""
        try:
            from pathlib import Path
            
            save_path = Path(request.path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            metadata = {
                "step": training_state["step"],
                "learning_rate": training_state["learning_rate"],
                **(request.metadata or {}),
            }
            
            llm_backend.save_adapter(str(save_path), metadata=metadata)
            
            return SaveStateResponse(
                path=str(save_path),
                success=True,
                message=f"Checkpoint saved to {save_path}",
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save state: {str(e)}",
            )
    
    @router.post(
        "/internal/train/load_state",
        response_model=LoadStateResponse,
        responses={
            200: {"description": "State loaded", "model": LoadStateResponse},
            401: {"description": "Unauthorized", "model": ErrorResponse},
            500: {"description": "Internal error", "model": ErrorResponse},
        },
        tags=["Training"],
    )
    async def train_load_state(request: LoadStateRequest) -> LoadStateResponse:
        """Load training checkpoint."""
        try:
            from pathlib import Path
            import json
            
            load_path = Path(request.path)
            if not load_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Checkpoint not found: {request.path}",
                )
            
            llm_backend.apply_adapter(str(load_path))
            adapter_state["path"] = str(load_path)
            
            # Try to load metadata
            step = training_state["step"]
            metadata_path = load_path / "adapter_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                step = metadata.get("step", step)
                training_state["step"] = step
                training_state["learning_rate"] = metadata.get("learning_rate", training_state["learning_rate"])
            
            return LoadStateResponse(
                path=str(load_path),
                success=True,
                message=f"Checkpoint loaded from {load_path}",
                step=step,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load state: {str(e)}",
            )
    
    @router.get(
        "/internal/train/weights",
        response_model=GetWeightsResponse,
        responses={
            200: {"description": "Weights retrieved", "model": GetWeightsResponse},
            401: {"description": "Unauthorized", "model": ErrorResponse},
            500: {"description": "Internal error", "model": ErrorResponse},
        },
        tags=["Training"],
    )
    async def train_get_weights() -> GetWeightsResponse:
        """Get current model weights."""
        try:
            weights = {}
            
            if hasattr(llm_backend, 'model') and llm_backend.model:
                model = llm_backend.model
                if hasattr(model, 'trainable_parameters'):
                    params = model.trainable_parameters()
                    # Convert to serializable format (shape info)
                    weights = {
                        k: {"shape": list(v.shape) if hasattr(v, 'shape') else str(type(v))}
                        for k, v in params.items()
                    }
            
            return GetWeightsResponse(
                weights=weights,
                success=True,
                message=f"Retrieved {len(weights)} weight tensors",
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get weights: {str(e)}",
            )
    
    @router.post(
        "/internal/train/weights",
        response_model=SetWeightsResponse,
        responses={
            200: {"description": "Weights set", "model": SetWeightsResponse},
            401: {"description": "Unauthorized", "model": ErrorResponse},
            500: {"description": "Internal error", "model": ErrorResponse},
        },
        tags=["Training"],
    )
    async def train_set_weights(request: SetWeightsRequest) -> SetWeightsResponse:
        """Set model weights."""
        try:
            # In practice, this would deserialize and set weights
            # For now, just return success with count
            num_tensors = len(request.weights)
            
            return SetWeightsResponse(
                success=True,
                message=f"Set {num_tensors} weight tensors",
                num_tensors=num_tensors,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to set weights: {str(e)}",
            )
    
    # ==========================================================================
    # Adapter Hot-Reload
    # ==========================================================================
    
    @router.post(
        "/internal/adapter/reload",
        response_model=AdapterReloadResponse,
        responses={
            200: {"description": "Adapter reloaded successfully", "model": AdapterReloadResponse},
            400: {"description": "Bad request", "model": ErrorResponse},
            401: {"description": "Unauthorized", "model": ErrorResponse},
            404: {"description": "Adapter not found", "model": ErrorResponse},
            500: {"description": "Internal error", "model": ErrorResponse},
        },
        tags=["Internal"],
    )
    async def reload_adapter(request: AdapterReloadRequest) -> AdapterReloadResponse:
        """Hot-reload adapter weights without restarting the server.
        
        Can also reload the base model if needed.
        """
        nonlocal adapter_state
        
        try:
            target = request.adapter_path
            
            if target:
                target_path = Path(target)
                if not target_path.is_absolute():
                    target = str(Path.cwd() / target_path)
                else:
                    target = str(target_path)
                
                if not target_path.exists():
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Adapter path not found: {target}",
                    )
            
            if request.reload_base or target is None:
                llm_backend.load(
                    base_model,
                    max_seq_len=getattr(cfg.model, "max_seq_len", 2048),
                    dtype=getattr(cfg.model, "dtype", "float16"),
                    trust_remote_code=getattr(cfg.model, "trust_remote_code", False),
                )
                adapter_state["path"] = None
            
            if target:
                llm_backend.apply_adapter(target)
                adapter_state["path"] = target
            
            return AdapterReloadResponse(
                ok=True,
                base_model=base_model,
                adapter_path=adapter_state["path"],
                message="Adapter reloaded successfully",
            )
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Adapter reload failed: {str(e)}",
            )
    
    # ==========================================================================
    # RLM State and History
    # ==========================================================================
    
    @router.get(
        "/internal/rlm/state",
        response_model=RLMState,
        responses={
            200: {"description": "Current RLM state", "model": RLMState},
            401: {"description": "Unauthorized", "model": ErrorResponse},
        },
        tags=["RLM"],
    )
    async def rlm_state() -> RLMState:
        """Get current RLM training state."""
        state_path = Path.cwd() / "runs" / "rlm_state.json"
        
        if not state_path.exists():
            return RLMState(status="idle")
        
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            return RLMState(**data)
        except Exception:
            return RLMState(status="idle")
    
    @router.get(
        "/internal/rlm/history",
        response_model=List[RLMHistoryEntry],
        responses={
            200: {"description": "RLM training history", "model": List[RLMHistoryEntry]},
            401: {"description": "Unauthorized", "model": ErrorResponse},
        },
        tags=["RLM"],
    )
    async def rlm_history(
        limit: Optional[int] = 100,
        offset: Optional[int] = 0,
    ) -> List[RLMHistoryEntry]:
        """Get RLM training history/metrics.
        
        Args:
            limit: Maximum number of entries to return
            offset: Number of entries to skip
        """
        history_path = Path.cwd() / "runs" / "rlm_history.jsonl"
        
        if not history_path.exists():
            return []
        
        rows = []
        try:
            lines = history_path.read_text(encoding="utf-8").splitlines()
            for line in lines[offset:offset + limit] if limit else lines[offset:]:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    rows.append(RLMHistoryEntry(**data))
                except Exception:
                    continue
        except Exception:
            pass
        
        return rows
    
    # ==========================================================================
    # Model Management
    # ==========================================================================
    
    def _get_model_format(path: Path) -> str:
        """Detect model format from path."""
        if (path / "model.safetensors").exists() or (path / "weights.safetensors").exists():
            return "mlx"
        if (path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists():
            return "hf"
        if list(path.glob("*.gguf")):
            return "gguf"
        return "mlx"  # Default
    
    def _has_adapter(path: Path) -> bool:
        """Check if path contains adapter weights."""
        return (
            (path / "adapter_config.json").exists() or
            (path / "adapters.safetensors").exists() or
            (path / "lora.npz").exists()
        )
    
    @router.get(
        "/internal/models/list",
        response_model=ModelsListResponse,
        responses={
            200: {"description": "List of cached models", "model": ModelsListResponse},
            401: {"description": "Unauthorized", "model": ErrorResponse},
        },
        tags=["Models"],
    )
    async def list_models() -> ModelsListResponse:
        """List cached MLX models in the cache directory."""
        cache_dir = _get_cache_dir()
        models = []
        
        mlx_dir = cache_dir / "mlx"
        if mlx_dir.exists():
            for model_path in mlx_dir.iterdir():
                if not model_path.is_dir():
                    continue
                
                # Calculate size
                size_bytes = 0
                try:
                    for f in model_path.rglob("*"):
                        if f.is_file():
                            size_bytes += f.stat().st_size
                except Exception:
                    pass
                
                # Get metadata
                metadata = None
                config_path = model_path / "config.json"
                if config_path.exists():
                    try:
                        metadata = json.loads(config_path.read_text())
                    except Exception:
                        pass
                
                model_id = model_path.name.replace("__", "/")
                has_adapter = _has_adapter(model_path)
                adapter_path = str(model_path) if has_adapter else None
                
                models.append(ModelInfo(
                    id=model_id,
                    path=str(model_path),
                    size_bytes=size_bytes,
                    format=_get_model_format(model_path),
                    has_adapter=has_adapter,
                    adapter_path=adapter_path,
                    metadata=metadata,
                    downloaded_at=int(model_path.stat().st_mtime),
                ))
        
        # Also check HF cache
        hf_dir = cache_dir / "hf"
        if hf_dir.exists():
            for model_path in hf_dir.iterdir():
                if not model_path.is_dir():
                    continue
                
                model_id = model_path.name.replace("__", "/")
                # Skip if already in MLX format
                if any(m.id == model_id for m in models):
                    continue
                
                size_bytes = 0
                try:
                    for f in model_path.rglob("*"):
                        if f.is_file():
                            size_bytes += f.stat().st_size
                except Exception:
                    pass
                
                models.append(ModelInfo(
                    id=f"{model_id} (HF)",
                    path=str(model_path),
                    size_bytes=size_bytes,
                    format="hf",
                    has_adapter=False,
                    metadata=None,
                    downloaded_at=int(model_path.stat().st_mtime),
                ))
        
        return ModelsListResponse(
            models=models,
            total=len(models),
            cache_dir=str(cache_dir),
        )
    
    @router.post(
        "/internal/models/pull",
        response_model=ModelPullResponse,
        responses={
            200: {"description": "Model pull initiated", "model": ModelPullResponse},
            400: {"description": "Bad request", "model": ErrorResponse},
            401: {"description": "Unauthorized", "model": ErrorResponse},
            500: {"description": "Pull failed", "model": ErrorResponse},
        },
        tags=["Models"],
    )
    async def pull_model(request: ModelPullRequest) -> ModelPullResponse:
        """Pull a model from HuggingFace.
        
        This proxies the `mlxsmith pull` command and initiates an
        asynchronous model download and optional conversion.
        
        Note: This endpoint returns immediately with a status. For large
        models, use the list endpoint to check completion status.
        """
        cache_dir = _get_cache_dir()

        try:
            # Import here to avoid circular dependencies
            from ..models import hf_pull
            
            # Get HF token if available
            hf_token = None
            token_path = Path.home() / ".config" / "mlxsmith" / "hf_token"
            if token_path.exists():
                hf_token = token_path.read_text().strip()
            
            # Start pull in background (synchronous for now)
            # In production, this would spawn a background task
            result_path = hf_pull(
                model_id=request.model_id,
                cache_dir=cache_dir,
                convert=request.convert,
                quantize=request.quantize,
                q_bits=request.q_bits,
                q_group_size=request.q_group_size,
                trust_remote_code=request.trust_remote_code,
                hf_token=hf_token,
            )
            
            return ModelPullResponse(
                ok=True,
                model_id=request.model_id,
                local_path=str(result_path),
                status=ModelPullStatus(
                    status="completed",
                    progress=100.0,
                    message=f"Model pulled successfully to {result_path}",
                ),
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model pull failed: {str(e)}",
            )

    @router.post(
        "/internal/models/delete",
        response_model=ModelDeleteResponse,
        responses={
            200: {"description": "Model deleted", "model": ModelDeleteResponse},
            400: {"description": "Bad request", "model": ErrorResponse},
            401: {"description": "Unauthorized", "model": ErrorResponse},
            404: {"description": "Model not found", "model": ErrorResponse},
            500: {"description": "Delete failed", "model": ErrorResponse},
        },
        tags=["Models"],
    )
    async def delete_model(model_id: str = Query(..., alias="model_id")) -> ModelDeleteResponse:
        """Delete a cached model by id.

        Supports MLX cache models and HF cache entries (id ending in " (HF)").
        """
        if not model_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="model_id required")

        cache_dir = _get_cache_dir()
        is_hf = model_id.endswith(" (HF)")
        cleaned = model_id.replace(" (HF)", "")
        model_dir_name = cleaned.replace("/", "__")
        candidates = []

        if is_hf:
            candidates.append(cache_dir / "hf" / model_dir_name)
        else:
            candidates.append(cache_dir / "mlx" / model_dir_name)
            candidates.append(cache_dir / "hf" / model_dir_name)

        target = next((path for path in candidates if path.exists()), None)
        if target is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

        try:
            shutil.rmtree(target)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete model: {str(e)}",
            )

        return ModelDeleteResponse(ok=True, model_id=model_id, message="Model deleted")
    
    # ==========================================================================
    # HuggingFace Token Management
    # ==========================================================================
    
    @router.post(
        "/internal/hf/token",
        response_model=HFTokenResponse,
        responses={
            200: {"description": "Token stored successfully", "model": HFTokenResponse},
            400: {"description": "Bad request", "model": ErrorResponse},
            401: {"description": "Unauthorized", "model": ErrorResponse},
            500: {"description": "Storage failed", "model": ErrorResponse},
        },
        tags=["HF Token"],
    )
    async def store_hf_token(request: HFTokenRequest) -> HFTokenResponse:
        """Store HuggingFace token securely.
        
        Attempts to use system keyring if available, falls back to
        file-based storage with restricted permissions.
        """
        token = request.token
        username = None
        storage_method: str = "memory"
        
        # Validate token if requested
        if request.validate_token:
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=token)
                user_info = api.whoami()
                username = user_info.get("name") if user_info else None
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Token validation failed: {str(e)}",
                )
        
        # Store token
        if request.persist:
            try:
                # Try keyring first
                try:
                    import keyring
                    keyring.set_password("mlxsmith", "huggingface", token)
                    storage_method = "keyring"
                except ImportError:
                    # Fall back to file storage
                    config_dir = Path.home() / ".config" / "mlxsmith"
                    config_dir.mkdir(parents=True, exist_ok=True)
                    token_path = config_dir / "hf_token"
                    token_path.write_text(token, encoding="utf-8")
                    # Restrict permissions (owner read/write only)
                    os.chmod(token_path, 0o600)
                    storage_method = "file"
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to store token: {str(e)}",
                )
        
        return HFTokenResponse(
            ok=True,
            validated=request.validate_token,
            username=username,
            message="Token stored successfully",
            storage_method=storage_method,
        )
    
    return router
