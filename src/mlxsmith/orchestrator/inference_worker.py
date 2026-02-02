"""Inference Worker Process for MLXSmith Orchestrator.

Runs as a separate process with OpenAI-compatible API.
Handles weight update messages from orchestrator.
Supports explicit weight reloading without restart.
"""

from __future__ import annotations

import asyncio
import threading
import json
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

# Relative imports will work when run as module
from ..config import ProjectConfig
from ..llm.registry import get_llm_backend
from ..models import resolve_model_spec
from ..rlm.recursive import recursive_compact, RecursiveStats
from ..rlm.weights import WeightPointerStore
from .queue import MessageQueue, MessageType, Message


@dataclass
class InferenceConfig:
    """Configuration for inference worker."""
    model_spec: str
    backend: str = "mlx-lm"
    host: str = "0.0.0.0"
    port: int = 8080
    max_seq_len: int = 8192
    dtype: str = "bf16"
    trust_remote_code: bool = False
    use_chat_template: bool = True
    weights_dir: Optional[Path] = None
    hot_reload: bool = True
    reload_poll_interval: float = 2.0


class InferenceWorker:
    """Inference worker process with OpenAI-compatible API.
    
    Runs in a separate process, handles:
    - OpenAI-compatible /v1/chat/completions endpoint
    - Internal /internal/rollout endpoint for RLM
    - Hot-reloading of adapter weights via weight pointer updates
    - Queue-based communication with orchestrator
    """
    
    def __init__(
        self,
        config: InferenceConfig,
        queue: Optional[MessageQueue] = None,
    ):
        self.config = config
        self.queue = queue
        self._llm = None
        self._base_model: Optional[str] = None
        self._current_adapter: Optional[str] = None
        self._pointer_store: Optional[WeightPointerStore] = None
        self._current_version = -1
        self._shutdown_event = asyncio.Event()
        self._app: Optional[FastAPI] = None
        
    def _load_model(self) -> None:
        """Load the base model and initial adapter."""
        self._llm = get_llm_backend(self.config.backend)
        
        # Resolve model spec
        base_model, adapter_path, _ = resolve_model_spec(
            Path.cwd(), self.config.model_spec, ProjectConfig()
        )
        self._base_model = base_model
        
        # Load base model
        self._llm.load(
            base_model,
            max_seq_len=self.config.max_seq_len,
            dtype=self.config.dtype,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        # Apply initial adapter if exists
        if adapter_path:
            self._llm.apply_adapter(str(adapter_path))
            self._current_adapter = str(adapter_path)
        
        # Setup weight pointer store for hot-reloading
        if self.config.hot_reload and self.config.weights_dir:
            self._pointer_store = WeightPointerStore(self.config.weights_dir)
            # Initial load of inference pointer
            pointer = self._pointer_store.load("inference", base_model)
            self._current_version = pointer.version
            if pointer.adapter_path and pointer.adapter_path != self._current_adapter:
                self._apply_adapter(pointer.adapter_path)
    
    def _apply_adapter(self, adapter_path: str) -> bool:
        """Apply a new adapter, reloading if necessary."""
        try:
            if adapter_path == self._current_adapter:
                return True
            
            # Reload base model and apply new adapter
            self._llm.load(
                self._base_model,
                max_seq_len=self.config.max_seq_len,
                dtype=self.config.dtype,
                trust_remote_code=self.config.trust_remote_code,
            )
            self._llm.apply_adapter(adapter_path)
            self._current_adapter = adapter_path
            return True
        except Exception as e:
            print(f"[InferenceWorker] Failed to apply adapter {adapter_path}: {e}")
            return False
    
    def _check_weight_updates(self) -> None:
        """Check for weight pointer updates."""
        if not self._pointer_store:
            return
        
        pointer = self._pointer_store.load("inference", self._base_model)
        if pointer.version > self._current_version:
            self._current_version = pointer.version
            if pointer.adapter_path:
                success = self._apply_adapter(pointer.adapter_path)
                if success:
                    print(f"[InferenceWorker] Hot-reloaded adapter: {pointer.adapter_path}")

    def _maybe_compact_prompt(
        self,
        prompt: str,
        payload: Dict[str, Any],
    ) -> tuple[str, Optional[RecursiveStats]]:
        if not payload.get("recursive"):
            return prompt, None
        if not self._llm:
            return prompt, None
        try:
            return recursive_compact(
                self._llm,
                prompt,
                max_seq_len=int(self.config.max_seq_len),
                chunk_tokens=int(payload.get("recursive_chunk_tokens", 1024)),
                overlap_tokens=int(payload.get("recursive_overlap_tokens", 128)),
                keep_last_tokens=int(payload.get("recursive_keep_last_tokens", 384)),
                summary_tokens=int(payload.get("recursive_summary_tokens", 256)),
                max_depth=int(payload.get("recursive_max_depth", 3)),
                temperature=float(payload.get("recursive_temperature", 0.2)),
                seed=payload.get("seed"),
                summary_prompt=payload.get("recursive_summary_prompt"),
            )
        except Exception:
            return prompt, None
    
    def _handle_queue_message(self, msg: Message) -> Optional[Message]:
        """Handle a message from the queue."""
        if msg.msg_type == MessageType.ROLLOUT_REQUEST:
            return self._handle_rollout_request(msg)
        elif msg.msg_type == MessageType.WEIGHT_UPDATE:
            return self._handle_weight_update(msg)
        elif msg.msg_type == MessageType.HEALTH_CHECK:
            return self._handle_health_check(msg)
        elif msg.msg_type == MessageType.SHUTDOWN:
            self._shutdown_event.set()
            return None
        return None
    
    def _handle_rollout_request(self, msg: Message) -> Message:
        """Handle a rollout request."""
        payload = msg.payload
        prompt = payload.get("prompt", "")
        max_tokens = payload.get("max_tokens", 256)
        temperature = payload.get("temperature", 0.8)
        top_p = payload.get("top_p", 1.0)
        top_k = payload.get("top_k")
        seed = payload.get("seed")
        
        # Check for weight updates before generating
        self._check_weight_updates()
        
        prompt_used, recursion_stats = self._maybe_compact_prompt(prompt, payload)

        # Generate rollout
        try:
            gen = self._llm.generate_with_logprobs(
                prompt_used,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
            )
        except TypeError:
            try:
                gen = self._llm.generate_with_logprobs(
                    prompt_used,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                )
            except TypeError:
                gen = self._llm.generate_with_logprobs(
                    prompt_used,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k_sampling=top_k,
                    seed=seed,
                )
        
        completion = gen.text[len(prompt_used):] if gen.text.startswith(prompt_used) else gen.text
        
        return Message(
            msg_type=MessageType.ROLLOUT_RESPONSE,
            payload={
                "request_id": msg.msg_id,
                "prompt_len": gen.prompt_len,
                "token_ids": list(gen.token_ids),
                "logprobs": list(gen.logprobs) if gen.logprobs else None,
                "completion": completion,
                "adapter_version": self._current_version,
                "prompt_used": prompt_used if prompt_used != prompt else None,
                "recursion_depth": getattr(recursion_stats, "depth", None) if recursion_stats else None,
                "recursion_chunks": getattr(recursion_stats, "chunks", None) if recursion_stats else None,
                "recursion_truncated": getattr(recursion_stats, "truncated", None) if recursion_stats else None,
            },
            source="inference",
        )
    
    def _handle_weight_update(self, msg: Message) -> Message:
        """Handle a weight update from the trainer."""
        payload = msg.payload
        adapter_path = payload.get("adapter_path")
        version = payload.get("version", 0)
        
        success = False
        if adapter_path:
            success = self._apply_adapter(adapter_path)
            if success:
                self._current_version = version
        
        return Message(
            msg_type=MessageType.WEIGHT_ACK,
            payload={
                "request_id": msg.msg_id,
                "success": success,
                "adapter_path": self._current_adapter,
                "version": self._current_version,
            },
            source="inference",
        )
    
    def _handle_health_check(self, msg: Message) -> Message:
        """Handle a health check request."""
        return Message(
            msg_type=MessageType.HEALTH_RESPONSE,
            payload={
                "request_id": msg.msg_id,
                "status": "healthy",
                "base_model": self._base_model,
                "adapter_path": self._current_adapter,
                "adapter_version": self._current_version,
            },
            source="inference",
        )
    
    def _create_app(self) -> FastAPI:
        """Create the FastAPI application."""
        app = FastAPI(title="mlxsmith-inference")
        
        @app.get("/health")
        def health():
            return {
                "ok": True,
                "base_model": self._base_model,
                "adapter_path": self._current_adapter,
                "adapter_version": self._current_version,
            }
        
        @app.post("/v1/chat/completions")
        def chat_completions(request: Dict[str, Any]):
            """OpenAI-compatible chat completions endpoint."""
            messages = request.get("messages", [])
            max_tokens = request.get("max_tokens", 256)
            temperature = request.get("temperature", 0.7)
            top_p = request.get("top_p", 1.0)
            stream = request.get("stream", False)
            
            # Check for weight updates
            self._check_weight_updates()
            
            # Build prompt from messages
            prompt = self._messages_to_prompt(messages)
            
            if stream:
                return StreamingResponse(
                    self._stream_generate(prompt, max_tokens, temperature, top_p),
                    media_type="text/event-stream",
                )
            
            gen = self._llm.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            
            completion = gen.text[len(prompt):] if gen.text.startswith(prompt) else gen.text
            
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self._base_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": completion},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(self._llm.encode(prompt)),
                    "completion_tokens": len(self._llm.encode(completion)),
                    "total_tokens": len(self._llm.encode(prompt)) + len(self._llm.encode(completion)),
                },
            }
        
        @app.post("/internal/rollout")
        def internal_rollout(request: Dict[str, Any]):
            """Internal rollout endpoint for RLM."""
            prompt = request.get("prompt", "")
            max_tokens = request.get("max_tokens", 256)
            temperature = request.get("temperature", 0.7)
            top_p = request.get("top_p", 1.0)
            top_k = request.get("top_k")
            seed = request.get("seed")
            include_tokens = request.get("include_tokens", True)
            include_logprobs = request.get("include_logprobs", True)
            include_top_k_logprobs = request.get("include_top_k_logprobs")
            
            # Check for weight updates
            self._check_weight_updates()

            prompt_used, recursion_stats = self._maybe_compact_prompt(prompt, request)

            gen = self._llm.generate_with_logprobs(
                prompt_used,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k_sampling=top_k,
                seed=seed,
                logprobs=int(include_top_k_logprobs or 0),
            )
            
            completion = gen.text[len(prompt_used):] if gen.text.startswith(prompt_used) else gen.text
            
            return {
                "id": f"rollout-{uuid.uuid4().hex[:12]}",
                "created": int(time.time()),
                "model": self._base_model,
                "prompt_len": gen.prompt_len,
                "token_ids": list(gen.token_ids) if include_tokens else None,
                "logprobs": list(gen.logprobs) if (include_logprobs and gen.logprobs) else None,
                "top_k_logprobs": gen.top_k_logprobs if include_top_k_logprobs else None,
                "completion": completion,
                "adapter_version": self._current_version,
                "prompt_used": prompt_used if prompt_used != prompt else None,
                "recursion_depth": getattr(recursion_stats, "depth", None) if recursion_stats else None,
                "recursion_chunks": getattr(recursion_stats, "chunks", None) if recursion_stats else None,
                "recursion_truncated": getattr(recursion_stats, "truncated", None) if recursion_stats else None,
            }
        
        @app.post("/internal/adapter/reload")
        def reload_adapter(request: Dict[str, Any]):
            """Explicitly reload adapter weights."""
            adapter_path = request.get("adapter_path")
            
            if adapter_path:
                success = self._apply_adapter(adapter_path)
            else:
                # Reload from pointer store
                self._check_weight_updates()
                success = True
            
            return {
                "ok": success,
                "base_model": self._base_model,
                "adapter_path": self._current_adapter,
                "adapter_version": self._current_version,
            }
        
        return app
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to prompt."""
        if self.config.use_chat_template and hasattr(self._llm.tokenizer, "apply_chat_template"):
            msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
            return self._llm.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        
        # Fallback
        return "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"
    
    async def _stream_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ):
        """Stream generate tokens."""
        try:
            import mlx_lm
            
            acc = ""
            emitted = ""
            
            for out in mlx_lm.stream_generate(
                self._llm.model,
                self._llm.tokenizer,
                prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
            ):
                if out.text:
                    acc += out.text
                    delta = acc[len(emitted):]
                    emitted = acc
                    
                    payload = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                
                if out.finish_reason:
                    break
            
            yield "data: [DONE]\n\n"
            
        except ImportError:
            # Non-streaming fallback
            gen = self._llm.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            completion = gen.text[len(prompt):] if gen.text.startswith(prompt) else gen.text
            
            payload = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": completion}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
    
    def _queue_worker_loop(self) -> None:
        """Background thread for queue-based communication."""
        if not self.queue:
            return

        while not self._shutdown_event.is_set():
            try:
                msg = self.queue.receive("rollout_requests", timeout=0.1)
                if msg:
                    response = self._handle_queue_message(msg)
                    if response:
                        self.queue.get_queue("rollout_responses").put(response.to_dict())

                weight_msg = self.queue.receive("weight_forward", timeout=0)
                if weight_msg:
                    self._handle_weight_update(weight_msg)

            except Exception as e:
                print(f"[InferenceWorker] Queue error: {e}")

            time.sleep(0.01)
    
    def run(self) -> None:
        """Run the inference worker."""
        # Setup signal handlers
        def signal_handler(sig, frame):
            print("[InferenceWorker] Shutting down...")
            self._shutdown_event.set()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Load model
        print("[InferenceWorker] Loading model...")
        self._load_model()
        print(f"[InferenceWorker] Model loaded: {self._base_model}")
        print(f"[InferenceWorker] Adapter: {self._current_adapter}")
        
        # Create FastAPI app
        self._app = self._create_app()
        
        # Start queue worker if enabled
        if self.queue:
            thread = threading.Thread(target=self._queue_worker_loop, daemon=True)
            thread.start()
        
        # Start uvicorn server
        print(f"[InferenceWorker] Starting server on {self.config.host}:{self.config.port}")
        uvicorn.run(
            self._app,
            host=self.config.host,
            port=self.config.port,
            log_level="warning",
        )


def run_inference_worker(
    config: InferenceConfig,
    queue: Optional[MessageQueue] = None,
) -> None:
    """Entry point for inference worker process."""
    worker = InferenceWorker(config, queue)
    worker.run()
