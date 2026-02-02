from __future__ import annotations

import json
import time
from pathlib import Path

from rich.console import Console

from .util import ensure_dir, now_ts
from .config import ProjectConfig, load_config
from .models import resolve_model_spec
from .llm.registry import get_llm_backend

console = Console()


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
        verify_fn = None
        if verifier_path:
            verify_fn = _load_verifier(project_root / verifier_path)
        passes = 0
        responses = []
        t0 = time.time()
        for i in range(k):
            gen = llm.generate(
                prompt,
                max_new_tokens=int(task.get("max_new_tokens", 256)),
                temperature=float(task.get("temperature", 0.7)),
                top_p=float(task.get("top_p", 1.0)),
                seed=int(task.get("seed", 0)) if task.get("seed") is not None else None,
            )
            completion = gen.text[len(prompt) :] if gen.text.startswith(prompt) else gen.text
            responses.append(completion)
            if verify_fn:
                res = verify_fn(prompt, completion, str(out_dir), **(task.get("verifier_kwargs") or {}))
                if bool(getattr(res, "passed", False)):
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
