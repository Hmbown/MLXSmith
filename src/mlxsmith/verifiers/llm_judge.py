from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .types import VerifyResult
from ..llm.registry import get_llm_backend

_STATE: Dict[str, Any] = {
    "backend": None,
    "backend_name": None,
    "model_id": None,
}


def _read_text(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    if value.startswith("@"):
        path = Path(value[1:])
        if path.exists():
            return path.read_text(encoding="utf-8")
    path = Path(value)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return value


def _load_backend(model_id: str, backend_name: str, *, max_seq_len: Optional[int], dtype: Optional[str], trust_remote_code: bool) -> Any:
    if (
        _STATE["backend"] is None
        or _STATE["backend_name"] != backend_name
        or _STATE["model_id"] != model_id
    ):
        backend = get_llm_backend(backend_name)
        backend.load(
            model_id,
            max_seq_len=max_seq_len,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        _STATE["backend"] = backend
        _STATE["backend_name"] = backend_name
        _STATE["model_id"] = model_id
    return _STATE["backend"]


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1].strip()
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*}", "}", snippet)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        cleaned = cleaned.replace("'", "\"")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def _coerce_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _aggregate_scores(scores: list[float], mode: str) -> Optional[float]:
    if not scores:
        return None
    mode = (mode or "product").lower()
    if mode == "min":
        return min(scores)
    if mode == "mean":
        return sum(scores) / float(len(scores))
    prod = 1.0
    for s in scores:
        prod *= s
    return prod


def _score_step(
    judge,
    *,
    system_prompt: str,
    step_text: str,
    prompt: str,
    completion: str,
    rubric_text: str,
    temperature: float,
    max_new_tokens: int,
) -> Optional[float]:
    step_prompt = (
        f"{system_prompt}\n\n"
        "Score this single step from a solution.\n\n"
        f"## Task\n{prompt}\n\n"
        f"## Model Answer\n{completion}\n\n"
        f"## Step\n{step_text}\n\n"
        f"## Rubric\n{rubric_text}\n\n"
        "Return JSON only."
    )
    gen = judge.generate(
        step_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=1.0,
        top_k=None,
    )
    raw = gen.text[len(step_prompt) :] if gen.text.startswith(step_prompt) else gen.text
    parsed = _extract_json(raw) or {}
    return _coerce_float(parsed.get("score"))


def verify(
    prompt: str,
    completion: str,
    workdir: str,
    *,
    model: Optional[str] = None,
    backend: str = "mlx-lm",
    system_prompt: Optional[str] = None,
    rubric: Optional[str] = None,
    mode: str = "judge",
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    min_score: float = 0.5,
    reward_pass: float = 1.0,
    reward_fail: float = 0.0,
    reward_mode: str = "score",
    max_seq_len: Optional[int] = None,
    dtype: Optional[str] = None,
    trust_remote_code: bool = False,
    mock_response: Optional[str] = None,
    process_agg: str = "product",
    max_steps: int = 8,
    **kwargs,
) -> VerifyResult:
    """LLM-based verifier with JSON output.

    The judge should return JSON: {"passed": bool, "score": 0..1, "reason": "..."}.
    Set mock_response to bypass backend loading (useful for tests).
    """
    model_id = model or os.environ.get("MLXSMITH_JUDGE_MODEL")
    if not model_id and not mock_response:
        raise RuntimeError("llm_judge requires `model` or MLXSMITH_JUDGE_MODEL")

    rubric_text = _read_text(rubric) or "Assess correctness and completeness."
    mode = (mode or "judge").strip().lower()
    sys_prompt = system_prompt or (
        "You are a strict verifier. Return ONLY JSON with keys: "
        "passed (bool), score (0-1), reason (string)."
    )

    if mode == "thinkprm":
        sys_prompt = system_prompt or (
            "You are a process reward model. Evaluate the reasoning quality. "
            "Return ONLY JSON with keys: passed (bool), score (0-1), reason (string), steps (array)."
        )

    user_prompt = (
        "## Task\n"
        f"{prompt}\n\n"
        "## Model Answer\n"
        f"{completion}\n\n"
        "## Rubric\n"
        f"{rubric_text}\n\n"
        "Return JSON only."
    )

    t0 = time.time()
    if mock_response is not None:
        raw = str(mock_response)
    else:
        judge = _load_backend(
            model_id,
            backend,
            max_seq_len=max_seq_len,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        full_prompt = f"{sys_prompt}\n\n{user_prompt}"
        gen = judge.generate(
            full_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            top_k=None,
        )
        raw = gen.text[len(full_prompt) :] if gen.text.startswith(full_prompt) else gen.text

    parsed = _extract_json(raw) or {}
    score = _coerce_float(parsed.get("score"))
    passed_val = parsed.get("passed")
    steps_raw = parsed.get("steps") if isinstance(parsed, dict) else None
    step_texts: list[str] = []
    step_scores: list[float] = []
    if mode == "thinkprm" and isinstance(steps_raw, list):
        for step in steps_raw[: max(1, int(max_steps))]:
            if isinstance(step, dict):
                text = step.get("text") or step.get("step") or step.get("content") or ""
                if text:
                    step_texts.append(str(text))
                s_val = _coerce_float(step.get("score"))
                if s_val is not None:
                    step_scores.append(float(s_val))
            elif isinstance(step, str):
                step_texts.append(step)

        if step_texts and (len(step_scores) < len(step_texts)) and mock_response is None:
            judge = _load_backend(
                model_id,
                backend,
                max_seq_len=max_seq_len,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
            for idx, step_text in enumerate(step_texts):
                if idx < len(step_scores):
                    continue
                s_val = _score_step(
                    judge,
                    system_prompt=sys_prompt,
                    step_text=step_text,
                    prompt=prompt,
                    completion=completion,
                    rubric_text=rubric_text,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
                if s_val is not None:
                    step_scores.append(float(s_val))

    process_score = _aggregate_scores(step_scores, process_agg) if step_scores else None
    if mode == "thinkprm" and process_score is not None:
        score = process_score
    if passed_val is None and score is not None:
        passed_val = score >= min_score
    passed = bool(passed_val) if passed_val is not None else False
    reason = parsed.get("reason") or parsed.get("explanation") or ""

    if reward_mode == "score" and score is not None:
        reward = max(0.0, min(1.0, float(score)))
    else:
        reward = reward_pass if passed else reward_fail

    latency_ms = (time.time() - t0) * 1000.0

    return VerifyResult(
        reward=reward,
        passed=passed,
        info={
            "mode": mode,
            "model": model_id,
            "score": score,
            "process_score": process_score,
            "process_agg": process_agg,
            "steps": step_texts,
            "step_scores": step_scores,
            "passed": passed,
            "reason": reason,
            "raw": raw,
            "verifier_latency_ms": latency_ms,
        },
        artifacts_dir=workdir,
    )
