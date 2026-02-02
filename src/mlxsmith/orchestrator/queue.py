"""Queue and message passing system for orchestrator.

Provides multiprocessing-safe message passing between daemon, inference worker,
and trainer worker. Supports both in-memory queues (single-node) and Redis-like
stores (distributed).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List, Optional, Union


class MessageType(Enum):
    """Message types for orchestrator communication."""
    
    # Rollout requests (Daemon -> Inference Worker)
    ROLLOUT_REQUEST = auto()
    ROLLOUT_RESPONSE = auto()
    
    # Training batches (Daemon -> Trainer Worker)
    TRAIN_BATCH = auto()
    TRAIN_COMPLETE = auto()
    
    # Weight updates (Trainer -> Daemon -> Inference)
    WEIGHT_UPDATE = auto()
    WEIGHT_ACK = auto()
    
    # Checkpoints (Trainer -> Daemon)
    CHECKPOINT = auto()
    CHECKPOINT_ACK = auto()
    
    # Health and control
    HEALTH_CHECK = auto()
    HEALTH_RESPONSE = auto()
    SHUTDOWN = auto()
    
    # Task generation
    TASK_GENERATE = auto()
    TASK_RESPONSE = auto()


@dataclass
class Message:
    """A message in the orchestrator system.
    
    Attributes:
        msg_type: The type of message
        payload: Message-specific data
        msg_id: Unique message identifier
        timestamp: Unix timestamp
        source: Source process identifier
    """
    msg_type: MessageType
    payload: Dict[str, Any] = field(default_factory=dict)
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize message to dictionary."""
        return {
            "msg_type": self.msg_type.name,
            "payload": self.payload,
            "msg_id": self.msg_id,
            "timestamp": self.timestamp,
            "source": self.source,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Deserialize message from dictionary."""
        return cls(
            msg_type=MessageType[data["msg_type"]],
            payload=data.get("payload", {}),
            msg_id=data.get("msg_id", str(uuid.uuid4())[:12]),
            timestamp=data.get("timestamp", time.time()),
            source=data.get("source", "unknown"),
        )
    
    def to_json(self) -> str:
        """Serialize message to JSON."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "Message":
        """Deserialize message from JSON."""
        return cls.from_dict(json.loads(json_str))


class MessageQueue:
    """Multi-process message queue system.
    
    Uses multiprocessing.Queue for IPC. Supports multiple named queues
    for different communication patterns.
    
    Queues:
        - rollout_requests: Daemon -> Inference (tasks to generate rollouts)
        - rollout_responses: Inference -> Daemon (completed rollouts)
        - train_batches: Daemon -> Trainer (training data)
        - train_complete: Trainer -> Daemon (training metrics)
        - weight_updates: Trainer -> Daemon -> Inference (new adapter paths)
        - control: Bidirectional control messages
    """
    
    def __init__(self, maxsize: int = 1000):
        """Initialize message queues.
        
        Args:
            maxsize: Maximum size for each queue (0 = unlimited)
        """
        self._maxsize = maxsize
        self._queues: Dict[str, mp.Queue] = {}
        self._manager: Optional[mp.managers.SyncManager] = None
        
    def start(self) -> None:
        """Start the queue manager and create queues."""
        # Use multiprocessing Manager for shared queues
        self._manager = mp.Manager()
        
        queue_names = [
            "rollout_requests",   # Daemon -> Inference
            "rollout_responses",  # Inference -> Daemon
            "train_batches",      # Daemon -> Trainer
            "train_complete",     # Trainer -> Daemon
            "weight_updates",     # Trainer -> Daemon (then forwarded)
            "weight_forward",     # Daemon -> Inference
            "checkpoints",        # Trainer -> Daemon
            "control",           # Bidirectional control
        ]
        
        for name in queue_names:
            self._queues[name] = self._manager.Queue(maxsize=self._maxsize)
    
    def stop(self) -> None:
        """Stop the queue manager."""
        if self._manager:
            self._manager.shutdown()
            self._manager = None
    
    def get_queue(self, name: str) -> mp.Queue:
        """Get a named queue."""
        if name not in self._queues:
            raise KeyError(f"Unknown queue: {name}")
        return self._queues[name]
    
    def send(
        self,
        queue_name: str,
        msg_type: MessageType,
        payload: Dict[str, Any],
        source: str = "unknown",
        timeout: Optional[float] = None,
    ) -> Message:
        """Send a message to a queue.
        
        Args:
            queue_name: Name of the queue
            msg_type: Type of message
            payload: Message payload
            source: Source process identifier
            timeout: Queue put timeout (None = block)
            
        Returns:
            The sent Message object
        """
        msg = Message(msg_type=msg_type, payload=payload, source=source)
        queue = self.get_queue(queue_name)
        queue.put(msg.to_dict(), timeout=timeout)
        return msg
    
    def receive(
        self,
        queue_name: str,
        timeout: Optional[float] = None,
    ) -> Optional[Message]:
        """Receive a message from a queue.
        
        Args:
            queue_name: Name of the queue
            timeout: Queue get timeout (None = block, 0 = non-blocking)
            
        Returns:
            Message or None if timeout
        """
        queue = self.get_queue(queue_name)
        try:
            data = queue.get(timeout=timeout)
            return Message.from_dict(data)
        except Empty:
            return None
    
    def receive_batch(
        self,
        queue_name: str,
        batch_size: int,
        timeout: float = 0.0,
    ) -> List[Message]:
        """Receive a batch of messages from a queue.
        
        Args:
            queue_name: Name of the queue
            batch_size: Maximum number of messages to receive
            timeout: Timeout for first message (0 = non-blocking)
            
        Returns:
            List of received messages
        """
        messages = []
        
        # First message with timeout
        msg = self.receive(queue_name, timeout=timeout)
        if msg is None:
            return messages
        messages.append(msg)
        
        # Remaining messages without blocking
        while len(messages) < batch_size:
            msg = self.receive(queue_name, timeout=0)
            if msg is None:
                break
            messages.append(msg)
        
        return messages
    
    def broadcast(
        self,
        queue_names: List[str],
        msg_type: MessageType,
        payload: Dict[str, Any],
        source: str = "unknown",
    ) -> List[Message]:
        """Broadcast a message to multiple queues.
        
        Args:
            queue_names: List of queue names
            msg_type: Type of message
            payload: Message payload
            source: Source process identifier
            
        Returns:
            List of sent Message objects
        """
        messages = []
        for name in queue_names:
            msg = self.send(name, msg_type, payload, source)
            messages.append(msg)
        return messages
    
    def queue_size(self, queue_name: str) -> int:
        """Get approximate queue size."""
        queue = self.get_queue(queue_name)
        return queue.qsize()
    
    def clear(self, queue_name: str) -> int:
        """Clear all messages from a queue. Returns count cleared."""
        queue = self.get_queue(queue_name)
        count = 0
        while not queue.empty():
            try:
                queue.get_nowait()
                count += 1
            except Empty:
                break
        return count


class PersistentMessageQueue(MessageQueue):
    """Message queue with persistence to disk for crash recovery.
    
    Messages are written to disk before being acknowledged, allowing
    recovery after process crashes.
    """
    
    def __init__(self, persistence_dir: Path, maxsize: int = 1000):
        """Initialize persistent queue.
        
        Args:
            persistence_dir: Directory for message persistence
            maxsize: Maximum size for each queue
        """
        super().__init__(maxsize=maxsize)
        self._persistence_dir = Path(persistence_dir)
        self._pending_dir = self._persistence_dir / "pending"
        self._completed_dir = self._persistence_dir / "completed"
    
    def start(self) -> None:
        """Start the queue and create persistence directories."""
        super().start()
        self._pending_dir.mkdir(parents=True, exist_ok=True)
        self._completed_dir.mkdir(parents=True, exist_ok=True)
    
    def send_persistent(
        self,
        queue_name: str,
        msg_type: MessageType,
        payload: Dict[str, Any],
        source: str = "unknown",
    ) -> Message:
        """Send a message with persistence.
        
        The message is written to disk before being queued.
        """
        msg = Message(msg_type=msg_type, payload=payload, source=source)
        
        # Write to pending directory
        pending_path = self._pending_dir / f"{msg.msg_id}.json"
        pending_path.write_text(msg.to_json(), encoding="utf-8")
        
        # Send to queue
        queue = self.get_queue(queue_name)
        queue.put(msg.to_dict())
        
        return msg
    
    def ack(self, msg_id: str) -> None:
        """Acknowledge a message, moving it from pending to completed."""
        pending_path = self._pending_dir / f"{msg_id}.json"
        if pending_path.exists():
            completed_path = self._completed_dir / f"{msg_id}.json"
            pending_path.rename(completed_path)
    
    def recover_pending(self) -> List[Message]:
        """Recover pending messages after crash."""
        messages = []
        for path in self._pending_dir.glob("*.json"):
            try:
                msg = Message.from_json(path.read_text(encoding="utf-8"))
                messages.append(msg)
            except Exception:
                pass  # Skip corrupted files
        return messages
