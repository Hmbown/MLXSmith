from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from rich.console import Console

from ..config import ProjectConfig
from ..models import resolve_model_spec
from ..runs import RunPaths, new_run, snapshot_config
from ..llm.registry import get_llm_backend
from ..llm.backend import BackendNotAvailable
from ..util import ensure_dir, write_jsonl, now_ts
from .sft import run_sft
from .pref import run_pref

console = Console()


def _iter_prompts(path: Path) -> Iterable[str]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = row.get("prompt") or row.get("instruction") or row.get("input") or ""
        if not prompt and "messages" in row:
            msgs = row.get("messages") or []
            if msgs:
                prompt = "\n".join([m.get("content", "") for m in msgs])
        if prompt:
            yield str(prompt)


def run_distill(
    project_root: Path,
    cfg: ProjectConfig,
    data_path: Path,
    *,
    teacher_model: str,
    student_model: str,
    accel: str,
    mode: str = "offline",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> RunPaths:
    run = new_run(project_root, "distill")
    snapshot_config(cfg.model_dump(), run.config_snapshot_path)

    mode = mode.lower()
    prompts = list(_iter_prompts(data_path))
    if not prompts:
        raise RuntimeError("No prompts found in distillation dataset")

    teacher = get_llm_backend(cfg.model.backend)
    student = get_llm_backend(cfg.model.backend)

    base_teacher, _, _meta_t = resolve_model_spec(project_root, teacher_model, cfg)
    base_student, _, _meta_s = resolve_model_spec(project_root, student_model, cfg)

    try:
        teacher.load(
            base_teacher,
            max_seq_len=cfg.model.max_seq_len,
            dtype=cfg.model.dtype,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        student.load(
            base_student,
            max_seq_len=cfg.model.max_seq_len,
            dtype=cfg.model.dtype,
            trust_remote_code=cfg.model.trust_remote_code,
        )
    except BackendNotAvailable as e:
        console.print(f"[yellow]MLX backend unavailable[/yellow]: {e}")
        (run.adapter_dir / "ADAPTER.txt").write_text(
            f"Backend unavailable in this environment.\nteacher={teacher_model}\nstudent={student_model}\n",
            encoding="utf-8",
        )
        return run

    distill_dir = ensure_dir(run.artifacts_dir / "distill_data")
    train_path = distill_dir / "train.jsonl"

    if mode == "opd":
        rows = []
        for prompt in prompts:
            t_gen = teacher.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            s_gen = student.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            chosen = t_gen.text[len(prompt) :] if t_gen.text.startswith(prompt) else t_gen.text
            rejected = s_gen.text[len(prompt) :] if s_gen.text.startswith(prompt) else s_gen.text
            rows.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )
        write_jsonl(train_path, rows)
        console.print(f"[green]Distill dataset (OPD)[/green] {train_path}")
        child = run_pref(project_root, cfg, distill_dir, Path(student_model), accel)
    else:
        rows = []
        for prompt in prompts:
            gen = teacher.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            completion = gen.text[len(prompt) :] if gen.text.startswith(prompt) else gen.text
            rows.append({"prompt": prompt, "response": completion})
        write_jsonl(train_path, rows)
        console.print(f"[green]Distill dataset (offline)[/green] {train_path}")
        child = run_sft(project_root, cfg, distill_dir, student_model, accel)

    write_jsonl(
        run.metrics_path,
        [
            {
                "ts": now_ts(),
                "kind": "distill",
                "mode": mode,
                "teacher": teacher_model,
                "student": student_model,
                "child_run": str(child.run_dir),
                "samples": len(prompts),
            }
        ],
    )
    return run
