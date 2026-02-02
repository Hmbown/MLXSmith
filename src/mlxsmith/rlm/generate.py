from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set

from ..util import sha1_text


@dataclass
class GeneratedTask:
    id: str
    prompt: str
    tests: str
    description: Optional[str] = None


_FALLBACK_TASKS = [
    {
        "id": "task_factorial",
        "description": "Write a function solve(n: int) -> int that returns n! for n >= 0.",
        "signature": "def solve(n: int) -> int",
        "tests": [
            {"input": [0], "expected": 1},
            {"input": [5], "expected": 120},
            {"input": [7], "expected": 5040},
        ],
    },
    {
        "id": "task_reverse",
        "description": "Write a function solve(s: str) -> str that returns the reverse of s.",
        "signature": "def solve(s: str) -> str",
        "tests": [
            {"input": ["abc"], "expected": "cba"},
            {"input": [""], "expected": ""},
            {"input": ["racecar"], "expected": "racecar"},
        ],
    },
]

_DEFAULT_BLOCKLIST = [
    r"\bsubprocess\b",
    r"\bos\.system\b",
    r"\bshutil\.rmtree\b",
    r"\brm\s+-rf\b",
    r"\brequests\b",
    r"\burllib\b",
    r"\bsocket\b",
    r"\bhttp[s]?://",
    r"\bpip\s+install\b",
    r"\bapt-get\b",
    r"\bbrew\s+install\b",
]


def extract_json_objects(text: str) -> List[dict]:
    # Try fenced json blocks first.
    fenced = re.findall(r"```json\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    chunks = fenced if fenced else [text]

    items: List[dict] = []
    for chunk in chunks:
        buf = ""
        depth = 0
        for ch in chunk:
            if ch == "{":
                depth += 1
            if depth > 0:
                buf += ch
            if ch == "}" and depth > 0:
                depth -= 1
                if depth == 0:
                    try:
                        items.append(json.loads(buf))
                    except Exception:
                        pass
                    buf = ""
    return items


def _tests_from_cases(cases: Iterable[dict]) -> str:
    lines = ["from main import solve", "", ""]
    for idx, case in enumerate(cases):
        args = case.get("input", [])
        expected = case.get("expected")
        if not isinstance(args, (list, tuple)):
            args = [args]
        lines.append(f"def test_case_{idx}():")
        args_expr = ", ".join(repr(a) for a in args)
        lines.append(f"    assert solve({args_expr}) == {repr(expected)}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def task_to_prompt(task: dict, *, require_recursion: bool) -> str:
    desc = task.get("description") or task.get("prompt") or ""
    sig = task.get("signature") or "def solve(...):"
    suffix = "\nUse recursion." if require_recursion else ""
    return f"{desc}\n{sig}\nReturn only Python code.{suffix}".strip()


def task_to_tests(task: dict) -> str:
    tests = task.get("tests")
    if isinstance(tests, str):
        return tests
    if isinstance(tests, list):
        # List of pre-formatted test strings (e.g. from Qwen3-style generation)
        if tests and isinstance(tests[0], str):
            return "\n\n".join(tests).strip() + "\n"
        # List of structured {input, expected} dicts
        return _tests_from_cases(tests)
    # fallback: trivial test that always fails to avoid false positives
    return "def test_placeholder():\n    assert False\n"


def _token_set(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    denom = len(a | b)
    if denom == 0:
        return 0.0
    return len(a & b) / denom


def filter_tasks(
    tasks: Sequence[GeneratedTask],
    *,
    existing_prompts: Optional[Sequence[str]] = None,
    similarity_threshold: float = 0.85,
    min_desc_len: int = 10,
    min_asserts: int = 2,
    max_prompt_len: int = 2000,
    min_tests_len: int = 20,
    max_tests_len: int = 8000,
    blocked_patterns: Optional[Sequence[str]] = None,
) -> List[GeneratedTask]:
    """Filter tasks by basic quality + similarity + dedup."""
    existing_prompts = existing_prompts or []
    existing_tokens = [_token_set(p) for p in existing_prompts if p]
    seen_hashes: Set[str] = set()
    filtered: List[GeneratedTask] = []
    patterns = list(blocked_patterns) if blocked_patterns else _DEFAULT_BLOCKLIST

    for task in tasks:
        prompt = task.prompt or ""
        if len(prompt) > max_prompt_len:
            continue
        desc = task.description or prompt
        if len(desc.strip()) < min_desc_len:
            continue
        tests = task.tests or ""
        if len(tests.strip()) < min_tests_len or len(tests) > max_tests_len:
            continue
        if patterns:
            blocked = False
            for pattern in patterns:
                if re.search(pattern, prompt, flags=re.IGNORECASE) or re.search(pattern, tests, flags=re.IGNORECASE):
                    blocked = True
                    break
            if blocked:
                continue
        asserts = sum(1 for line in task.tests.splitlines() if line.strip().startswith("assert"))
        if asserts < min_asserts:
            continue

        key = sha1_text(prompt + "\n" + task.tests)
        if key in seen_hashes:
            continue

        tokens = _token_set(prompt)
        too_similar = False
        for ex in existing_tokens:
            if _jaccard(tokens, ex) >= similarity_threshold:
                too_similar = True
                break
        if too_similar:
            continue

        seen_hashes.add(key)
        filtered.append(task)

    return filtered


def generate_tasks(
    llm,
    *,
    tasks_per_iter: int,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    top_k: Optional[int],
    require_recursion: bool,
    task_domains: Iterable[str],
) -> List[GeneratedTask]:
    if tasks_per_iter <= 0:
        return []

    domain_list = ", ".join(task_domains) if task_domains else "general"
    prompt = (
        "Generate JSON objects for coding tasks. Each JSON must include: "
        "id, description, signature, tests (array of {input, expected}). "
        f"Domain focus: {domain_list}. Return only JSON objects."
    )

    gen = llm.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    items = extract_json_objects(gen.text)
    tasks: List[GeneratedTask] = []
    for item in items:
        tid = item.get("id") or sha1_text(json.dumps(item, sort_keys=True))[:12]
        task_prompt = task_to_prompt(item, require_recursion=require_recursion)
        tests = task_to_tests(item)
        tasks.append(
            GeneratedTask(
                id=str(tid),
                prompt=task_prompt,
                tests=tests,
                description=item.get("description"),
            )
        )
        if len(tasks) >= tasks_per_iter:
            break

    if not tasks:
        for item in _FALLBACK_TASKS:
            tid = item.get("id") or sha1_text(json.dumps(item, sort_keys=True))[:12]
            tasks.append(
                GeneratedTask(
                    id=str(tid),
                    prompt=task_to_prompt(item, require_recursion=require_recursion),
                    tests=task_to_tests(item),
                    description=item.get("description"),
                )
            )
            if len(tasks) >= tasks_per_iter:
                break

    return tasks
