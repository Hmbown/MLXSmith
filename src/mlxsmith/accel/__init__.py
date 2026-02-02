from __future__ import annotations
from .none import NoneBackend

def get_backend(name: str):
    if name == "none":
        return NoneBackend()
    raise ValueError(f"Unknown accel backend: {name}")
