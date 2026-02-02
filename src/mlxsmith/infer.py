from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .config import ProjectConfig
from .llm.registry import get_llm_backend
from .models import resolve_model_spec


@dataclass
class ChatMessage:
    role: str
    content: str


def _messages_to_prompt(messages: List[ChatMessage], tokenizer, *, use_chat_template: bool) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        payload = [{"role": m.role, "content": m.content} for m in messages]
        try:
            return tokenizer.apply_chat_template(payload, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    joined = "\n".join([f"{m.role}: {m.content}" for m in messages])
    return f"{joined}\nassistant:"


def _load_backend(cfg: ProjectConfig, model_spec: str):
    llm = get_llm_backend(cfg.model.backend)
    base_model, adapter_path, _meta = resolve_model_spec(Path.cwd(), model_spec, cfg)
    llm.load(
        base_model,
        max_seq_len=cfg.model.max_seq_len,
        dtype=cfg.model.dtype,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if adapter_path:
        llm.apply_adapter(str(adapter_path))
    return llm, base_model


def run_prompt(
    cfg: ProjectConfig,
    model_spec: str,
    prompt: str,
    *,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
) -> str:
    llm, _base_model = _load_backend(cfg, model_spec)
    gen = llm.generate(
        prompt,
        max_new_tokens=max_new_tokens or cfg.infer.max_new_tokens,
        temperature=temperature if temperature is not None else cfg.infer.temperature,
        top_p=top_p if top_p is not None else cfg.infer.top_p,
        top_k=top_k if top_k is not None else cfg.infer.top_k,
        seed=seed,
    )
    if gen.text.startswith(prompt):
        return gen.text[len(prompt) :]
    return gen.text


def run_chat(
    cfg: ProjectConfig,
    model_spec: str,
    messages: List[ChatMessage],
    *,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
) -> str:
    llm, _base_model = _load_backend(cfg, model_spec)
    prompt = _messages_to_prompt(messages, llm.tokenizer, use_chat_template=cfg.model.use_chat_template)
    gen = llm.generate(
        prompt,
        max_new_tokens=max_new_tokens or cfg.infer.max_new_tokens,
        temperature=temperature if temperature is not None else cfg.infer.temperature,
        top_p=top_p if top_p is not None else cfg.infer.top_p,
        top_k=top_k if top_k is not None else cfg.infer.top_k,
        seed=seed,
    )
    if gen.text.startswith(prompt):
        return gen.text[len(prompt) :]
    return gen.text


def chat_repl(
    cfg: ProjectConfig,
    model_spec: str,
    *,
    system: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    max_turns: Optional[int] = None,
) -> None:
    llm, _base_model = _load_backend(cfg, model_spec)
    messages: List[ChatMessage] = []
    if system:
        messages.append(ChatMessage(role="system", content=system))

    turns = 0
    while True:
        if max_turns is not None and turns >= max_turns:
            break
        try:
            user = input("user> ").strip()
        except EOFError:
            break
        if not user:
            continue
        if user.lower() in {"/exit", "/quit", "exit", "quit"}:
            break
        messages.append(ChatMessage(role="user", content=user))
        prompt = _messages_to_prompt(messages, llm.tokenizer, use_chat_template=cfg.model.use_chat_template)
        gen = llm.generate(
            prompt,
            max_new_tokens=max_new_tokens or cfg.infer.max_new_tokens,
            temperature=temperature if temperature is not None else cfg.infer.temperature,
            top_p=top_p if top_p is not None else cfg.infer.top_p,
            top_k=top_k if top_k is not None else cfg.infer.top_k,
            seed=seed,
        )
        if gen.text.startswith(prompt):
            reply = gen.text[len(prompt) :]
        else:
            reply = gen.text
        reply = reply.strip()
        print(f"assistant> {reply}\n")
        messages.append(ChatMessage(role="assistant", content=reply))
        turns += 1
