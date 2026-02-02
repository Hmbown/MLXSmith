from __future__ import annotations

from .mlx_lm_backend import MlxLMBackend
from .mock_backend import MockBackend


def get_llm_backend(name: str):
    if name == "mlx-lm":
        return MlxLMBackend()
    if name == "mock":
        return MockBackend()
    raise ValueError(f"Unknown LLM backend: {name}")
