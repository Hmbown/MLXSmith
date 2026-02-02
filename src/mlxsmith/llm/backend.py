"""LLM backend abstraction.

We keep this intentionally small so mlxsmith can support multiple model loaders
without hard-binding to one ecosystem.

Primary target: mlx-lm (HF -> MLX format + common chat models).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, Any, Optional, List, Dict


@dataclass
class Generation:
    text: str
    token_ids: list[int]
    prompt_len: int
    # Log-probabilities for generated tokens only (len = completion tokens), if available.
    logprobs: list[float] | None = None
    # Top-k logprobs per token: list of {token_str: logprob} dicts for each generated token.
    top_k_logprobs: List[Dict[str, float]] | None = None


@dataclass
class DecodingConfig:
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 1.0
    top_k: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[Sequence[str]] = None


class LLMBackend(Protocol):
    name: str

    def load(self, model_id_or_path: str, *, max_seq_len: int | None = None, dtype: str | None = None) -> None:
        """Load model + tokenizer into memory."""

    def encode(self, text: str) -> list[int]:
        """Tokenize text -> token ids."""

    def decode(self, ids: Sequence[int]) -> str:
        """Token ids -> text."""

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 1.0,
        top_k: int | None = None,
        seed: int | None = None,
    ) -> Generation:
        """Sample a completion and return ids for prompt+completion."""

    def generate_with_logprobs(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 1.0,
        top_k: int | None = None,
        seed: int | None = None,
        logprobs: int = 0,  # Number of top logprobs to return per token (0 = just the sampled token)
    ) -> Generation:
        """Sample a completion and include per-token logprobs when available.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            seed: Random seed for reproducibility
            logprobs: Number of top logprobs to return per token (0 = return only sampled token's logprob)
            
        Returns:
            Generation with logprobs and optionally top_k_logprobs
        """

    def sft_loss(self, token_ids: Sequence[int], *, train_on_prompt: bool, prompt_len: int) -> Any:
        """Return a scalar loss suitable for backprop (MLX array)."""

    def rl_loss(self, token_ids: Sequence[int], *, prompt_len: int, advantage: float) -> Any:
        """Return policy-gradient-style loss for a sampled trajectory."""

    def sequence_logprob(self, token_ids: Sequence[int], *, prompt_len: int) -> Any:
        """Return log-probability sum of response tokens (differentiable)."""

    def token_logprobs(
        self,
        token_ids: Sequence[int],
        *,
        prompt_len: int,
        top_k: int = 0,
        include_prompt: bool = False,
    ) -> tuple[list[float], List[Dict[str, float]] | None]:
        """Return per-token logprobs (and optional top-k logprobs).

        Args:
            token_ids: Full token sequence (prompt + completion).
            prompt_len: Prompt length in tokens.
            top_k: Number of top logprobs per token to return (0 = none).
            include_prompt: If True, include prompt tokens; otherwise only response tokens.
        """

    def value_and_grad(self, loss_fn) -> tuple[Any, Any | None]:
        """Return (loss, grads) using backend autograd when available."""

    def optimizer_and_params(self, *, lr: float, weight_decay: float = 0.0) -> tuple[Any, Any]:
        """Return (optimizer, trainable_params_tree)."""

    def apply_grads(self, optimizer: Any, grads: Any) -> None:
        """Update model parameters given gradients."""

    def save_adapter(self, out_dir: str, *, metadata: dict | None = None) -> None:
        """Persist adapter weights (LoRA) to out_dir."""


class BackendNotAvailable(RuntimeError):
    pass
