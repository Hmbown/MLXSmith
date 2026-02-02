"""FluxEM tool-calling verifier for MLXSmith GRPO.

Evaluates tool-calling completions by:
1. Parsing <tool_call> blocks from the model's completion
2. Checking tool selection against expected tool name (0.3 reward)
3. Executing the tool via fluxem_tools and comparing result (0.7 reward)

Verifier kwargs (passed per-task from environment YAML):
    expected_tool (str): Expected tool name
    expected_result: Expected result value (number, list, dict, string, etc.)
    tolerance (float): Relative tolerance for numeric comparison (default: 1e-3)
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from mlxsmith.verifiers.types import VerifyResult


def _parse_tool_call(text: str) -> dict[str, Any] | None:
    """Extract first <tool_call> JSON from completion text."""
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if not match:
        # Fallback: try bare JSON with "name" and "arguments"
        match = re.search(
            r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}',
            text,
            re.DOTALL,
        )
        if not match:
            return None
    try:
        return json.loads(match.group(1) if "<tool_call>" in text else match.group(0))
    except json.JSONDecodeError:
        return None


def _is_close(actual: Any, expected: Any, rtol: float = 1e-3, atol: float = 1e-9) -> bool:
    """Compare values with tolerance for numeric types."""
    if actual is None or expected is None:
        return actual == expected

    # Numeric comparison
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if math.isnan(expected) and math.isnan(actual):
            return True
        if expected == 0:
            return abs(actual) < atol
        return abs(actual - expected) / max(abs(expected), atol) < rtol

    # List/array comparison
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return False
        return all(_is_close(a, e, rtol, atol) for a, e in zip(actual, expected))

    # Dict comparison
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False
        return all(_is_close(actual[k], expected[k], rtol, atol) for k in expected)

    # String comparison (case-insensitive, stripped)
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()

    # Boolean
    if isinstance(expected, bool) and isinstance(actual, bool):
        return expected == actual

    # Fallback: try numeric coercion
    try:
        return _is_close(float(actual), float(expected), rtol, atol)
    except (ValueError, TypeError):
        pass

    return str(actual).strip() == str(expected).strip()


def _extract_result(tool_output: Any) -> Any:
    """Extract the core result from tool output (handles {"result": ...} wrapping)."""
    if isinstance(tool_output, dict):
        if "result" in tool_output:
            return tool_output["result"]
        if "value" in tool_output:
            return tool_output["value"]
    return tool_output


def verify(
    prompt: str,
    completion: str,
    workdir: str,
    *,
    expected_tool: str | None = None,
    expected_result: Any = None,
    tolerance: float = 1e-3,
    **kwargs: Any,
) -> VerifyResult:
    """Verify a tool-calling completion."""
    info: dict[str, Any] = {}

    # 1. Parse tool call from completion
    call = _parse_tool_call(completion)
    if call is None:
        return VerifyResult(
            reward=0.0,
            passed=False,
            info={"error": "no_tool_call_found", "completion_preview": completion[:200]},
        )

    tool_name = call.get("name", "")
    arguments = call.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return VerifyResult(
                reward=0.0,
                passed=False,
                info={"error": "invalid_arguments_json", "arguments_raw": arguments[:200]},
            )

    info["parsed_tool"] = tool_name
    info["parsed_args"] = arguments

    # 2. Score tool selection (0.3 reward)
    tool_score = 0.0
    if expected_tool:
        tool_correct = tool_name == expected_tool
        tool_score = 1.0 if tool_correct else 0.0
        info["tool_correct"] = tool_correct
        info["expected_tool"] = expected_tool
    else:
        # No expected tool specified — give credit if any tool was called
        tool_score = 1.0 if tool_name else 0.0

    # 3. Execute tool and compare result (0.7 reward)
    result_score = 0.0
    try:
        from fluxem_tools import call_tool as fluxem_call_tool
        raw_output = fluxem_call_tool(tool_name, **arguments)
        actual_result = _extract_result(raw_output)
        info["tool_output"] = str(actual_result)[:500]

        if expected_result is not None:
            result_correct = _is_close(actual_result, expected_result, rtol=tolerance)
            result_score = 1.0 if result_correct else 0.0
            info["result_correct"] = result_correct
            info["expected_result"] = str(expected_result)[:200]
        else:
            # No expected result — credit for successful execution
            result_score = 1.0
            info["result_correct"] = True

    except Exception as e:
        info["tool_error"] = str(e)[:300]
        # If tool name was wrong, execution failure is expected
        if not (expected_tool and tool_name != expected_tool):
            result_score = 0.0

    # Combined reward: 30% tool selection + 70% result accuracy
    reward = 0.3 * tool_score + 0.7 * result_score
    passed = reward > 0.8

    info["reward_breakdown"] = {
        "tool_selection": round(tool_score, 3),
        "result_accuracy": round(result_score, 3),
        "total": round(reward, 3),
    }

    return VerifyResult(
        reward=reward,
        passed=passed,
        info=info,
    )
