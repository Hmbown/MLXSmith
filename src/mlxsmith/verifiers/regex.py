from __future__ import annotations

import re

from .types import VerifyResult

def verify(prompt: str, completion: str, workdir: str, *, pattern: str, flags: int = 0, reward_pass: float = 1.0, reward_fail: float = 0.0) -> VerifyResult:
    m = re.search(pattern, completion, flags=flags)
    passed = m is not None
    return VerifyResult(
        reward=reward_pass if passed else reward_fail,
        passed=passed,
        info={"pattern": pattern, "match": m.group(0) if m else None},
        artifacts_dir=None,
    )
