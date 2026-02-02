from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict

from .registry import get_llm_backend
from .backend import DecodingConfig, Generation
from ..config import ProjectConfig
from ..models import resolve_model_spec
from ..train.lora import load_adapter_config


@dataclass
class LoadedModel:
    backend: any
    base_model: str
    adapter_path: Optional[str]


def load_base_model(model_id_or_path: str, cfg: ProjectConfig) -> LoadedModel:
    """Load a base model (and optionally adapter) with the configured backend."""
    backend = get_llm_backend(cfg.model.backend)
    from pathlib import Path

    base_model, adapter_path, _meta = resolve_model_spec(Path.cwd(), model_id_or_path, cfg)
    backend.load(
        base_model,
        max_seq_len=cfg.model.max_seq_len,
        dtype=cfg.model.dtype,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    return LoadedModel(backend=backend, base_model=base_model, adapter_path=str(adapter_path) if adapter_path else None)


def load_adapter(adapter_path: str):
    """Load adapter config without applying it."""
    return load_adapter_config(adapter_path)


def apply_adapter(backend, adapter_path: str) -> None:
    backend.apply_adapter(adapter_path)


def generate(
    backend, 
    tokenizer, 
    prompts: Iterable[str], 
    decoding_config: DecodingConfig,
    logprobs: int = 0,
) -> List[Generation]:
    """Generate completions for prompts.
    
    Args:
        backend: The LLM backend
        tokenizer: Tokenizer instance
        prompts: Iterable of prompt strings
        decoding_config: Decoding configuration
        logprobs: Number of top logprobs to return per token (0 = none)
        
    Returns:
        List of Generation results
    """
    results = []
    for p in prompts:
        if logprobs > 0:
            results.append(
                backend.generate_with_logprobs(
                    p,
                    max_new_tokens=decoding_config.max_new_tokens,
                    temperature=decoding_config.temperature,
                    top_p=decoding_config.top_p,
                    top_k_sampling=decoding_config.top_k,
                    seed=decoding_config.seed,
                    logprobs=logprobs,
                )
            )
        else:
            results.append(
                backend.generate(
                    p,
                    max_new_tokens=decoding_config.max_new_tokens,
                    temperature=decoding_config.temperature,
                    top_p=decoding_config.top_p,
                    top_k=decoding_config.top_k,
                    seed=decoding_config.seed,
                )
            )
    return results


def chat(
    backend,
    messages: List[Dict[str, str]],
    decoding_config: DecodingConfig,
    logprobs: int = 0,
    use_chat_template: bool = True,
) -> Generation:
    """Generate a chat completion.
    
    Args:
        backend: The LLM backend
        messages: List of message dicts with 'role' and 'content' keys
        decoding_config: Decoding configuration
        logprobs: Number of top logprobs to return per token (0 = none)
        use_chat_template: Whether to use the model's chat template
        
    Returns:
        Generation result
    """
    # Convert messages to prompt
    prompt = _messages_to_prompt(backend.tokenizer, messages, use_chat_template=use_chat_template)
    
    if logprobs > 0:
        return backend.generate_with_logprobs(
            prompt,
            max_new_tokens=decoding_config.max_new_tokens,
            temperature=decoding_config.temperature,
            top_p=decoding_config.top_p,
            top_k_sampling=decoding_config.top_k,
            seed=decoding_config.seed,
            logprobs=logprobs,
        )
    else:
        return backend.generate(
            prompt,
            max_new_tokens=decoding_config.max_new_tokens,
            temperature=decoding_config.temperature,
            top_p=decoding_config.top_p,
            top_k=decoding_config.top_k,
            seed=decoding_config.seed,
        )


def _messages_to_prompt(
    tokenizer,
    messages: List[Dict[str, str]],
    *,
    use_chat_template: bool = True
) -> str:
    """Convert chat messages to prompt string."""
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback
    return "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"


@dataclass
class LogprobResult:
    """Result from logprobs computation."""
    token_logprobs: List[float]
    top_k_logprobs: Optional[List[Dict[str, float]]] = None
    text: Optional[str] = None


def compute_logprobs(
    backend,
    prompt: str,
    completion: str,
    top_k: int = 0,
    max_seq_len: Optional[int] = None,
) -> LogprobResult:
    """Compute logprobs for a prompt-completion pair.
    
    Args:
        backend: The LLM backend
        prompt: Prompt text
        completion: Completion text
        top_k: Number of top logprobs per token to return (0 = none)
        max_seq_len: Maximum sequence length
        
    Returns:
        LogprobResult with token logprobs and optionally top-k logprobs
    """
    prompt_ids = backend.encode(prompt)
    ids = backend.encode(prompt + completion)
    
    # Truncate if needed
    if max_seq_len and len(ids) > max_seq_len:
        overflow = len(ids) - max_seq_len
        ids = ids[overflow:]
        prompt_len = max(0, len(prompt_ids) - overflow)
    else:
        prompt_len = len(prompt_ids)
    
    # Decode and compute sequence-level logprob (used by callers via backend state)
    backend.decode(ids)
    backend.sequence_logprob(ids, prompt_len=prompt_len)
    
    # For per-token logprobs, we'd need to do a forward pass
    # This is a simplified version
    token_logprobs = []
    if hasattr(backend, '_response_logprobs'):
        token_logprobs = backend._response_logprobs(ids, prompt_len=prompt_len)
    
    # Get top-k logprobs if requested
    top_k_logprobs = None
    if top_k > 0 and hasattr(backend, 'generate_with_logprobs'):
        # Generate to get top-k for each position
        gen = backend.generate_with_logprobs(
            prompt,
            max_new_tokens=len(ids) - prompt_len,
            logprobs=top_k,
        )
        top_k_logprobs = gen.top_k_logprobs
    
    return LogprobResult(
        token_logprobs=token_logprobs,
        top_k_logprobs=top_k_logprobs,
        text=completion,
    )
