from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import validate, ValidationError

from .types import VerifyResult


def _load_schema(schema: Any) -> Any:
    if isinstance(schema, str):
        p = Path(schema)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return schema


def verify(
    prompt: str,
    completion: str,
    workdir: str,
    *,
    schema: Any,
    reward_pass: float = 1.0,
    reward_fail: float = 0.0,
) -> VerifyResult:
    try:
        data = json.loads(completion)
    except json.JSONDecodeError as e:
        return VerifyResult(
            reward=reward_fail,
            passed=False,
            info={"error": "invalid_json", "detail": str(e)},
            artifacts_dir=None,
        )

    schema_obj = _load_schema(schema)
    try:
        validate(instance=data, schema=schema_obj)
        return VerifyResult(
            reward=reward_pass,
            passed=True,
            info={"schema": schema_obj},
            artifacts_dir=None,
        )
    except ValidationError as e:
        return VerifyResult(
            reward=reward_fail,
            passed=False,
            info={"error": "schema_validation", "detail": str(e)},
            artifacts_dir=None,
        )
