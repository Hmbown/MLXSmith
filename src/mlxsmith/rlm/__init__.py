"""Recursive Language Model (RLM) module for MLXSmith.

Provides both single-process and multi-process orchestrated RLM training loops.
"""

from .loop import run_rlm, run_rlm_orchestrated
from .recursive import recursive_compact, RecursiveStats

__all__ = ["run_rlm", "run_rlm_orchestrated", "recursive_compact", "RecursiveStats"]
