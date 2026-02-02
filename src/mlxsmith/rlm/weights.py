"""Weight pointer system for tracking adapter weights across RLM iterations.

Extends to support IPC for multi-process orchestration with atomic updates
and hot-reload capabilities.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

from ..util import ensure_dir, now_ts


@dataclass
class WeightPointer:
    base_model: str
    adapter_path: Optional[str]
    iteration: int
    updated_at: str
    name: Optional[str] = None


@dataclass  
class WeightPointerIPC:
    """Extended WeightPointer with IPC support for multi-process orchestration.
    
    Includes versioning and atomic update mechanisms for hot-reloading.
    """
    base_model: str
    adapter_path: Optional[str]
    iteration: int
    updated_at: str
    version: int = 0  # Monotonic version for ordering updates
    checksum: Optional[str] = None  # Optional checksum for integrity
    name: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "base_model": self.base_model,
            "adapter_path": self.adapter_path,
            "iteration": self.iteration,
            "updated_at": self.updated_at,
            "version": self.version,
            "checksum": self.checksum,
            "name": self.name,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "WeightPointerIPC":
        return cls(
            base_model=data["base_model"],
            adapter_path=data.get("adapter_path"),
            iteration=data.get("iteration", 0),
            updated_at=data.get("updated_at", now_ts()),
            version=data.get("version", 0),
            checksum=data.get("checksum"),
            name=data.get("name"),
        )


class WeightPointerStore:
    """Atomic weight pointer store for IPC between processes.
    
    Uses file-based atomic updates with versioning to ensure
    inference workers always see consistent state.
    """
    
    def __init__(self, weights_dir: Path):
        self._weights_dir = Path(weights_dir)
        self._lock = mp.Lock()
        
    def get_path(self, name: str) -> Path:
        """Get the storage path for a named pointer."""
        return self._weights_dir / f"{name}.json"
    
    def get_atomic_path(self, name: str) -> Path:
        """Get the temporary path for atomic writes."""
        return self._weights_dir / f".{name}.tmp"
    
    def load(self, name: str, base_model: str) -> WeightPointerIPC:
        """Load a weight pointer from storage."""
        path = self.get_path(name)
        
        with self._lock:
            if not path.exists():
                return WeightPointerIPC(
                    base_model=base_model,
                    adapter_path=None,
                    iteration=0,
                    updated_at=now_ts(),
                    version=0,
                    name=name,
                )
            
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return WeightPointerIPC.from_dict(data)
            except Exception:
                return WeightPointerIPC(
                    base_model=base_model,
                    adapter_path=None,
                    iteration=0,
                    updated_at=now_ts(),
                    version=0,
                    name=name,
                )
    
    def save(self, pointer: WeightPointerIPC) -> None:
        """Atomically save a weight pointer."""
        path = self.get_path(pointer.name or "default")
        tmp_path = self.get_atomic_path(pointer.name or "default")
        
        ensure_dir(self._weights_dir)
        
        with self._lock:
            # Write to temp file
            tmp_path.write_text(
                json.dumps(pointer.to_dict(), indent=2),
                encoding="utf-8",
            )
            # Atomic rename
            tmp_path.rename(path)
    
    def update(
        self,
        name: str,
        adapter_path: Optional[str] = None,
        iteration: Optional[int] = None,
        checksum: Optional[str] = None,
    ) -> WeightPointerIPC:
        """Update a weight pointer atomically."""
        current = self.load(name, "")  # base_model will be preserved
        
        new_pointer = WeightPointerIPC(
            base_model=current.base_model,
            adapter_path=adapter_path if adapter_path is not None else current.adapter_path,
            iteration=iteration if iteration is not None else current.iteration,
            updated_at=now_ts(),
            version=current.version + 1,
            checksum=checksum,
            name=name,
        )
        
        self.save(new_pointer)
        return new_pointer
    
    def watch(
        self,
        name: str,
        base_model: str,
        callback: Callable[[WeightPointerIPC], None],
        poll_interval: float = 1.0,
    ) -> "WeightWatcher":
        """Create a watcher that monitors for pointer changes."""
        return WeightWatcher(self, name, base_model, callback, poll_interval)


class WeightWatcher:
    """Watches a weight pointer for changes and triggers callbacks.
    
    Used by inference workers to hot-reload weights when updates
    are published by the trainer.
    """
    
    def __init__(
        self,
        store: WeightPointerStore,
        name: str,
        base_model: str,
        callback: Callable[[WeightPointerIPC], None],
        poll_interval: float = 1.0,
    ):
        self._store = store
        self._name = name
        self._base_model = base_model
        self._callback = callback
        self._poll_interval = poll_interval
        self._last_version = -1
        self._running = False
        self._process: Optional[mp.Process] = None
    
    def start(self) -> None:
        """Start watching in a background process."""
        self._running = True
        self._process = mp.Process(target=self._watch_loop)
        self._process.start()
    
    def stop(self) -> None:
        """Stop the watcher."""
        self._running = False
        if self._process:
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
            self._process = None
    
    def _watch_loop(self) -> None:
        """Internal watch loop running in separate process."""
        while self._running:
            try:
                pointer = self._store.load(self._name, self._base_model)
                if pointer.version > self._last_version:
                    self._last_version = pointer.version
                    self._callback(pointer)
            except Exception:
                pass  # Continue watching despite errors
            
            time.sleep(self._poll_interval)


def load_pointer(path: Path, *, base_model: str, name: Optional[str] = None) -> WeightPointer:
    """Load a weight pointer from disk (backward compatible)."""
    if not path.exists():
        return WeightPointer(
            base_model=base_model,
            adapter_path=None,
            iteration=0,
            updated_at=now_ts(),
            name=name,
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return WeightPointer(
        base_model=data.get("base_model") or base_model,
        adapter_path=data.get("adapter_path"),
        iteration=int(data.get("iteration", 0)),
        updated_at=data.get("updated_at") or now_ts(),
        name=data.get("name") or name,
    )


def save_pointer(path: Path, pointer: WeightPointer) -> None:
    """Save a weight pointer to disk (backward compatible)."""
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(
            {
                "base_model": pointer.base_model,
                "adapter_path": pointer.adapter_path,
                "iteration": pointer.iteration,
                "updated_at": pointer.updated_at,
                "name": pointer.name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_pointer_ipc(path: Path, base_model: str, name: str) -> WeightPointerIPC:
    """Load an IPC-enabled weight pointer."""
    store = WeightPointerStore(path.parent)
    return store.load(name, base_model)


def save_pointer_ipc(path: Path, pointer: WeightPointerIPC) -> None:
    """Save an IPC-enabled weight pointer."""
    store = WeightPointerStore(path.parent)
    store.save(pointer)
