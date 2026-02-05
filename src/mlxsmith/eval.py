from __future__ import annotations

import json
import re
import time
from pathlib import Path

from rich.console import Console

from .util import ensure_dir, now_ts
from .config import ProjectConfig, load_config
from .models import resolve_model_spec
from .llm.registry import get_llm_backend

console = Console()

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL | re.IGNORECASE)
_SPECIAL_MARKER_RE = re.compile(r"<\|[^|]{1,80}\|>")
_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", flags=re.DOTALL | re.IGNORECASE)


def _strip_qwen_artifacts(text: str) -> str:
    if not text:
        return text
    lower = text.lower()
    if "<think" not in lower and "</think>" not in lower and "<|" not in text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in cleaned.lower():
        cleaned = re.split(r"</think>", cleaned, flags=re.IGNORECASE)[-1]
    lower_cleaned = cleaned.lower()
    idx = lower_cleaned.find("<think>")
    if idx != -1:
        after = cleaned[idx + len("<think>") :]
        m = re.search(r"\n\s*\n", after)
        if m:
            cleaned = (cleaned[:idx] + after[m.end() :]).lstrip()
        else:
            cleaned = (cleaned[:idx] + after).lstrip()
    cleaned = _SPECIAL_MARKER_RE.sub("", cleaned)
    return cleaned.lstrip("\n")


def _extract_code(text: str) -> str:
    if not text:
        return text
    matches = _CODE_BLOCK_RE.findall(text)
    if matches:
        return matches[0].strip()
    return text.strip()


def _write_tests(workdir: Path, tests: str) -> None:
    tests_dir = ensure_dir(workdir / "tests")
    (tests_dir / "test_main.py").write_text(tests, encoding="utf-8")


def _load_verifier(verifier_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(verifier_path.stem, verifier_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load verifier: {verifier_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    verify_fn = getattr(module, "verify", None)
    if not callable(verify_fn):
        raise RuntimeError(f"Verifier must define verify(...): {verifier_path}")
    return verify_fn


def run_eval(project_root: Path, suite_path: Path, model_path: Path) -> Path:
    import yaml

    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8")) or {}
    out_dir = ensure_dir(project_root / "eval" / "last")
    artifacts_dir = ensure_dir(out_dir / "artifacts")
    out_path = out_dir / "results.json"

    cfg_path = project_root / "mlxsmith.yaml"
    if cfg_path.exists():
        cfg = load_config(cfg_path)
    else:
        cfg = ProjectConfig()
    if suite.get("config"):
        merged = cfg.model_dump()
        merged.update(suite.get("config") or {})
        cfg = ProjectConfig.model_validate(merged)

    llm = get_llm_backend(cfg.model.backend)
    base_model, adapter_path, _meta = resolve_model_spec(project_root, str(model_path), cfg)
    llm.load(
        base_model,
        max_seq_len=cfg.model.max_seq_len,
        dtype=cfg.model.dtype,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if adapter_path:
        llm.apply_adapter(str(adapter_path))

    strip_think = bool(getattr(getattr(cfg, "infer", None), "strip_think", False))

    tasks = suite.get("tasks") or []
    if not tasks:
        results = {
            "model": str(model_path),
            "suite": suite.get("name", suite_path.name),
            "error": "no tasks",
        }
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return out_path

    summaries = []
    for task in tasks:
        prompt = task.get("prompt", "")
        k = int(task.get("k", 1))
        verifier_path = task.get("verifier")
        tests_text = task.get("tests")
        timeout_s = int(task.get("timeout_s", 30))
        verify_fn = None
        if verifier_path:
            verify_fn = _load_verifier(project_root / verifier_path)
        passes = 0
        t0 = time.time()
        for i in range(k):
            task_id = str(task.get("id") or prompt[:32]).strip() or "task"
            safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", task_id)[:80]
            workdir = ensure_dir(artifacts_dir / safe_id / f"attempt_{i:02d}")
            if tests_text:
                _write_tests(workdir, str(tests_text))

            gen = llm.generate(
                prompt,
                max_new_tokens=int(task.get("max_new_tokens", 256)),
                temperature=float(task.get("temperature", 0.7)),
                top_p=float(task.get("top_p", 1.0)),
                seed=int(task.get("seed", 0)) if task.get("seed") is not None else None,
            )
            completion = gen.text[len(prompt) :] if gen.text.startswith(prompt) else gen.text
            completion = completion.strip()
            if strip_think:
                completion = _strip_qwen_artifacts(completion)
            completion = _extract_code(completion)

            res = None
            if verify_fn:
                res = verify_fn(prompt, completion, str(workdir), **(task.get("verifier_kwargs") or {}))
            elif tests_text:
                from .verifiers.pytest_verifier import verify as pytest_verify

                res = pytest_verify(
                    prompt,
                    completion,
                    str(workdir),
                    timeout_s=timeout_s,
                )

            if res is not None and bool(getattr(res, "passed", False)):
                passes += 1
        elapsed = max(time.time() - t0, 1e-6)
        summaries.append(
            {
                "task_id": task.get("id") or prompt[:32],
                "k": k,
                "pass@k": passes / max(1, k),
                "latency_s": elapsed,
            }
        )

    results = {
        "model": str(model_path),
        "suite": suite.get("name", suite_path.name),
        "ts": now_ts(),
        "summary": summaries,
    }
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    console.print(f"[green]Wrote[/green] {out_path}")
    return out_path
