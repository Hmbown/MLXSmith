from __future__ import annotations

from .mlx_lm_backend import MlxLMBackend
from .mock_backend import MockBackend
from .openai_backend import OpenAIBackend
from .cli_backend import CliBackend


def get_llm_backend(name: str):
    if name == "mlx-lm":
        return MlxLMBackend()
    if name == "mock":
        return MockBackend()
    if name == "openai":
        return OpenAIBackend()
    if name == "cli":
        return CliBackend()
    raise ValueError(f"Unknown LLM backend: {name}")
