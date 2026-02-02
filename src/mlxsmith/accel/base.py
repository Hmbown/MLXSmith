from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol

@dataclass
class AccelStats:
    backend: str
    compiled: int = 0
    cache_hits: int = 0
    notes: Dict[str, Any] | None = None

class AccelBackend(Protocol):
    name: str
    def patch(self) -> None: ...
    def warmup(self, model: Any, example_batch: Any) -> Dict[str, Any]: ...
    def stats(self) -> AccelStats: ...
