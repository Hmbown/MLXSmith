from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from rich.console import Console

from ..accel import get_backend
from ..config import ProjectConfig
from ..llm.interface import compute_logprobs
from ..models import resolve_model_spec
from ..runs import RunPaths, new_run, snapshot_config
from ..llm.registry import get_llm_backend
from ..llm.backend import BackendNotAvailable
from ..sdk.losses import importance_sampling_loss
from ..train.lora import LoRAConfig
from ..util import ensure_dir, write_jsonl, now_ts, tree_add, tree_scale
from .sft import run_sft

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

    backend = get_backend(accel)
    backend.patch()
    console.print(f"[bold]DISTILL[/bold] run: {run.run_dir.name} mode={mode} accel={backend.name}")

    teacher = get_llm_backend(cfg.model.backend)
    student = get_llm_backend(cfg.model.backend)

    base_teacher, _, _meta_t = resolve_model_spec(project_root, teacher_model, cfg)
    base_student, adapter_path, _meta_s = resolve_model_spec(project_root, student_model, cfg)

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
        rows = [{"prompt": prompt} for prompt in prompts]
        write_jsonl(train_path, rows)
        console.print(f"[green]Distill dataset (OPD)[/green] {train_path}")

        if adapter_path:
            student.apply_adapter(str(adapter_path))
        else:
            lora_cfg = LoRAConfig(
                r=cfg.lora.r,
                alpha=cfg.lora.alpha,
                dropout=cfg.lora.dropout,
                target_modules=list(cfg.lora.target_modules or []),
                num_layers=cfg.lora.num_layers,
                scale=cfg.lora.scale,
                fine_tune_type=cfg.lora.fine_tune_type,
            )
            student.apply_lora_from_config(lora_cfg)

        opt, _params = student.optimizer_and_params(lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

        rng = random.Random(cfg.train.seed)
        total = int(cfg.train.iters)
        grad_accum = max(1, int(cfg.train.grad_accum))
        max_len = int(cfg.model.max_seq_len)
        accum_grads = None

        def _to_float(val):
            if hasattr(val, "item"):
                try:
                    return float(val.item())
                except Exception:
                    pass
            return float(val)

        for step in range(1, total + 1):
            prompt = rng.choice(prompts)
            gen = student.generate_with_logprobs(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                logprobs=0,
            )
            completion = gen.text[len(prompt) :] if gen.text.startswith(prompt) else gen.text
            behavior_logprobs = list(gen.logprobs) if gen.logprobs else []
            if not behavior_logprobs:
                continue

            teacher_logprobs: list[float] = []
            teacher_avg = None
            teacher_res = compute_logprobs(
                teacher,
                prompt,
                completion,
                top_k=0,
                max_seq_len=max_len,
            )
            if teacher_res.token_logprobs:
                teacher_logprobs = [float(lp) for lp in teacher_res.token_logprobs]
            else:
                prompt_ids = teacher.encode(prompt)
                ids = teacher.encode(prompt + completion)
                if max_len and len(ids) > max_len:
                    overflow = len(ids) - max_len
                    ids = ids[overflow:]
                    prompt_len = max(0, len(prompt_ids) - overflow)
                else:
                    prompt_len = len(prompt_ids)
                teacher_len = max(0, len(ids) - prompt_len)
                if teacher_len > 0:
                    teacher_total = teacher.sequence_logprob(ids, prompt_len=prompt_len)
                    teacher_avg = _to_float(teacher_total) / float(teacher_len)

            ids = list(gen.token_ids)
            prompt_len = gen.prompt_len
            if max_len and len(ids) > max_len:
                overflow = len(ids) - max_len
                ids = ids[overflow:]
                if overflow >= prompt_len:
                    removed_completion = overflow - prompt_len
                    prompt_len = 0
                    if removed_completion > 0:
                        behavior_logprobs = behavior_logprobs[removed_completion:]
                        if teacher_logprobs:
                            teacher_logprobs = teacher_logprobs[removed_completion:]
                else:
                    prompt_len = max(0, prompt_len - overflow)

            completion_len = max(0, len(ids) - prompt_len)
            if completion_len == 0 or not behavior_logprobs:
                continue

            if teacher_logprobs:
                n = min(len(teacher_logprobs), len(behavior_logprobs), completion_len)
                if n <= 0:
                    continue
                if completion_len != n:
                    ids = ids[: prompt_len + n]
                    completion_len = n
                behavior_logprobs = behavior_logprobs[:n]
                teacher_logprobs = teacher_logprobs[:n]
                behavior_logprob = sum(behavior_logprobs)
                advantage = sum(t - b for t, b in zip(teacher_logprobs, behavior_logprobs)) / float(n)
            else:
                if teacher_avg is None:
                    continue
                if len(behavior_logprobs) > completion_len:
                    behavior_logprobs = behavior_logprobs[:completion_len]
                behavior_logprob = sum(behavior_logprobs)
                if not behavior_logprobs:
                    continue
                behavior_avg = behavior_logprob / float(len(behavior_logprobs))
                advantage = teacher_avg - behavior_avg

            def loss_fn(_model):
                return importance_sampling_loss(
                    student,
                    ids,
                    prompt_len=prompt_len,
                    advantage=advantage,
                    behavior_logprob=behavior_logprob,
                )

            lval, grads = student.value_and_grad(loss_fn)
            if grads is not None:
                accum_grads = tree_add(accum_grads, grads)

            if step % grad_accum == 0:
                if accum_grads is not None:
                    student.apply_grads(opt, tree_scale(accum_grads, 1.0 / grad_accum))
                accum_grads = None

            if step % cfg.train.log_every == 0 or step == 1 or step == total:
                write_jsonl(
                    run.metrics_path,
                    [
                        {
                            "ts": now_ts(),
                            "step": step,
                            "kind": "distill_opd",
                            "loss": _to_float(lval),
                            "advantage": float(advantage),
                            "accel": backend.name,
                        }
                    ],
                )

            if step % cfg.train.save_every == 0 or step == total:
                student.save_adapter(
                    str(run.adapter_dir),
                    metadata={
                        "base_model": base_student,
                        "source_adapter": str(adapter_path) if adapter_path else None,
                        "run": run.run_dir.name,
                        "kind": "distill_opd",
                    },
                )
        child = run
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
