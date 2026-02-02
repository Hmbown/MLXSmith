from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence, Any, List, Dict, Optional

from .backend import Generation


class _DummyMx:
    def array(self, x):
        return float(x)

    def log1p(self, x):
        import math

        return math.log1p(x)

    def exp(self, x):
        import math

        return math.exp(x)

    def log(self, x):
        import math

        return math.log(x)

    def sigmoid(self, x):
        import math

        return 1.0 / (1.0 + math.exp(-x))

    def minimum(self, a, b):
        return a if a < b else b

    def maximum(self, a, b):
        return a if a > b else b
    
    def log_softmax(self, x, axis=-1):
        """Mock log softmax - returns normalized log probs."""
        import math
        if isinstance(x, list):
            # Simple softmax
            max_x = max(x)
            exp_x = [math.exp(v - max_x) for v in x]
            sum_exp = sum(exp_x)
            return [math.log(v / sum_exp) for v in exp_x]
        return x
    
    def argsort(self, x, axis=-1):
        """Mock argsort - returns indices sorted by value."""
        if isinstance(x, list):
            return [i for i, _ in sorted(enumerate(x), key=lambda p: p[1], reverse=True)]
        return list(range(10))


@dataclass
class _DummyValue:
    value: float

    def item(self):
        return self.value

    def __float__(self):
        return float(self.value)


class MockBackend:
    """Lightweight backend for tests and CI (no MLX dependency)."""

    name = "mock"

    def __init__(self):
        self.model = object()
        self.tokenizer = object()
        self.mx = _DummyMx()
        self._vocab_size = 50000  # Mock vocab size

    def load(self, model_id_or_path: str, *, max_seq_len: int | None = None, dtype: str | None = None, **kwargs) -> None:
        return None

    def apply_adapter(self, adapter_path: str) -> None:
        return None

    def apply_lora_from_config(self, cfg) -> dict:
        return {
            "fine_tune_type": "lora",
            "num_layers": 0,
            "lora_parameters": {"rank": 1, "scale": 1.0, "dropout": 0.0},
        }

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 256 for c in text][-256:] or [1]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(i % 256) for i in ids)

    def generate(self, prompt: str, *, max_new_tokens: int = 16, temperature: float = 0.0, top_p: float = 1.0, top_k: int | None = None, seed: int | None = None) -> Generation:
        prompt_ids = self.encode(prompt)
        new_ids = [42] * max_new_tokens
        ids = list(prompt_ids) + new_ids
        return Generation(text=self.decode(ids), token_ids=ids, prompt_len=len(prompt_ids))

    def generate_with_logprobs(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k_sampling: int | None = None,
        seed: int | None = None,
        logprobs: int = 0,  # Number of top logprobs to return per token
    ) -> Generation:
        """Generate with mock logprobs and top-k logprobs.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (unused in mock)
            top_p: Nucleus sampling (unused in mock)
            top_k_sampling: Top-k sampling (unused in mock)
            seed: Random seed
            logprobs: Number of top logprobs to return per token
            
        Returns:
            Generation with mock logprobs and top_k_logprobs
        """
        if seed is not None:
            random.seed(seed)
        
        prompt_ids = self.encode(prompt)
        new_ids = [42 + i % 10 for i in range(max_new_tokens)]
        ids = list(prompt_ids) + new_ids
        
        # Generate mock logprobs
        per_token_logprobs = [random.uniform(-3.0, -0.1) for _ in range(max_new_tokens)]
        
        # Generate mock top-k logprobs if requested
        per_token_top_k = None
        if logprobs > 0:
            per_token_top_k = []
            for _ in range(max_new_tokens):
                # Generate k mock tokens with logprobs
                token_dict = {}
                for j in range(min(logprobs, 10)):  # Mock up to 10 top tokens
                    token_id = random.randint(0, 255)
                    token_str = self.decode([token_id])
                    # Ensure unique keys
                    if token_str in token_dict:
                        token_str = f"{token_str}_{j}"
                    token_dict[token_str] = random.uniform(-4.0, -0.1)
                per_token_top_k.append(token_dict)
        
        return Generation(
            text=self.decode(ids), 
            token_ids=ids, 
            prompt_len=len(prompt_ids),
            logprobs=per_token_logprobs,
            top_k_logprobs=per_token_top_k,
        )

    def sft_loss(self, token_ids: Sequence[int], *, train_on_prompt: bool, prompt_len: int) -> Any:
        return _DummyValue(1.0)

    def rl_loss(self, token_ids: Sequence[int], *, prompt_len: int, advantage: float) -> Any:
        return _DummyValue(0.5)

    def sequence_logprob(self, token_ids: Sequence[int], *, prompt_len: int) -> Any:
        return 0.1

    def token_logprobs(
        self,
        token_ids: Sequence[int],
        *,
        prompt_len: int,
        top_k: int = 0,
        include_prompt: bool = False,
    ) -> tuple[List[float], List[Dict[str, float]] | None]:
        if len(token_ids) < 2:
            return [], [] if top_k > 0 else None
        start = 0 if include_prompt else max(0, prompt_len - 1)
        count = max(0, len(token_ids) - 1 - start)
        logprobs = [random.uniform(-3.0, -0.1) for _ in range(count)]
        if top_k <= 0:
            return logprobs, None
        top_k_list: List[Dict[str, float]] = []
        for _ in range(count):
            token_dict: Dict[str, float] = {}
            for j in range(min(top_k, 10)):
                token_id = random.randint(0, 255)
                token_str = self.decode([token_id])
                if token_str in token_dict:
                    token_str = f"{token_str}_{j}"
                token_dict[token_str] = random.uniform(-4.0, -0.1)
            top_k_list.append(token_dict)
        return logprobs, top_k_list

    def value_and_grad(self, loss_fn):
        return loss_fn(self.model), None

    def optimizer_and_params(self, *, lr: float, weight_decay: float = 0.0):
        return object(), {}

    def apply_grads(self, optimizer: Any, grads: Any) -> None:
        return None

    def save_adapter(self, out_dir: str, *, metadata: dict | None = None) -> None:
        from pathlib import Path
        import json

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "ADAPTER.txt").write_text("mock adapter", encoding="utf-8")
        (out / "adapter_config.json").write_text(
            json.dumps(
                {
                    "fine_tune_type": "lora",
                    "num_layers": 0,
                    "lora_parameters": {"rank": 1, "scale": 1.0, "dropout": 0.0},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if metadata is not None:
            (out / "adapter_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
