"""Recursive Language Model (RLM) module for MLXSmith.

Provides both single-process and multi-process orchestrated RLM training loops,
as well as REPL-based RLM inference following the Zhang et al. paradigm.
"""

from .loop import run_rlm, run_rlm_orchestrated
from .recursive import recursive_compact, RecursiveStats
from .repl import RLMEnvironment, REPLConfig, REPLResult, extract_code_blocks, format_repl_output
from .rlm_inference import (
    run_rlm_inference,
    RLMInferenceConfig,
    RLMTrajectory,
    RLMTurn,
    save_trajectory,
    load_trajectory,
    trajectory_to_training_pairs,
    RLM_SYSTEM_PROMPT,
)
from .docker_repl import DockerRLMEnvironment, DockerREPLConfig

__all__ = [
    # Training loop
    "run_rlm",
    "run_rlm_orchestrated",
    # Legacy recursive compression
    "recursive_compact",
    "RecursiveStats",
    # REPL environment (canonical RLM)
    "RLMEnvironment",
    "REPLConfig",
    "REPLResult",
    "extract_code_blocks",
    "format_repl_output",
    # RLM inference
    "run_rlm_inference",
    "RLMInferenceConfig",
    "RLMTrajectory",
    "RLMTurn",
    "save_trajectory",
    "load_trajectory",
    "trajectory_to_training_pairs",
    "RLM_SYSTEM_PROMPT",
    # Docker sandbox
    "DockerRLMEnvironment",
    "DockerREPLConfig",
]
