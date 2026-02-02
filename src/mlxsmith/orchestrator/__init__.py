"""MLXSmith Multi-Process Orchestrator.

This module provides PrimeIntellect-style multi-process coordination for RLM training:
- Daemon: Central queue-based job scheduler
- Inference Worker: Separate process with OpenAI-compatible API
- Trainer Worker: Separate process for training batches
- Queue/Communication: Message passing via multiprocessing Queue
"""

from .daemon import OrchestratorDaemon, DaemonConfig
from .queue import MessageQueue, MessageType, Message
from .inference_worker import InferenceWorker, InferenceConfig
from .trainer_worker import TrainerWorker, TrainerConfig

__all__ = [
    "OrchestratorDaemon",
    "DaemonConfig",
    "MessageQueue",
    "MessageType",
    "Message",
    "InferenceWorker",
    "InferenceConfig",
    "TrainerWorker",
    "TrainerConfig",
]
