from __future__ import annotations

from typing import Any, Dict
from .base import AccelStats

class NoneBackend:
    name = "none"
    def patch(self) -> None:
        return
    def warmup(self, model: Any, example_batch: Any) -> Dict[str, Any]:
        return {"warmup": "skipped"}
    def stats(self) -> AccelStats:
        return AccelStats(backend="none")
