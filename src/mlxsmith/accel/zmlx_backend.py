from __future__ import annotations

from typing import Any, Dict
from .base import AccelStats

class ZMLXBackend:
    name = "zmlx"

    def __init__(self):
        self._available = False
        self._notes = {}
        try:
            import zmlx  # type: ignore
            self._available = True
            self._notes["zmlx_version"] = getattr(zmlx, "__version__", None)
        except Exception as e:
            self._available = False
            self._notes["error"] = f"{type(e).__name__}: {e}"

    def patch(self) -> None:
        if not self._available:
            # soft fail; caller should report status
            return
        # ZMLX can patch ops/modules. We keep this intentionally minimal and safe.
        try:
            import zmlx  # type: ignore
            # If ZMLX provides a global patch hook, call it; otherwise, no-op.
            patch_fn = getattr(zmlx, "patch", None)
            if callable(patch_fn):
                patch_fn()
                self._notes["patched"] = True
            else:
                self._notes["patched"] = False
                self._notes["hint"] = "No zmlx.patch() found; implement patch hook or integrate per-module."
        except Exception as e:
            self._notes["patched_error"] = f"{type(e).__name__}: {e}"

    def warmup(self, model: Any, example_batch: Any) -> Dict[str, Any]:
        return {"warmup": "not_implemented", "notes": self._notes}

    def stats(self) -> AccelStats:
        return AccelStats(backend="zmlx", notes=self._notes)
