from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class VerifyResult:
    reward: float
    passed: bool
    info: Dict[str, Any]
    artifacts_dir: Optional[str] = None
