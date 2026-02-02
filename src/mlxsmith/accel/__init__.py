from __future__ import annotations
from .none import NoneBackend
from .zmlx_backend import ZMLXBackend

def get_backend(name: str):
    if name == "none":
        return NoneBackend()
    if name == "zmlx":
        return ZMLXBackend()
    raise ValueError(f"Unknown accel backend: {name}")
