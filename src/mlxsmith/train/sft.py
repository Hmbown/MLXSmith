from __future__ import annotations

import json
import random
from pathlib import Path

from rich.console import Console

from ..accel import get_backend
from ..config import ProjectConfig
from ..models import resolve_model_spec
from ..runs import RunPaths, new_run, snapshot_config
from ..util import write_jsonl, now_ts, tree_add, tree_scale, clip_grad_norm
from ..llm.registry import get_llm_backend
from ..llm.backend import BackendNotAvailable
from .lora import LoRAConfig

console = Console()


def _load_sft_rows(train_path: Path) -> list[dict]:
    return [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _row_to_prompt_response(row: dict) -> tuple[str, str]:
    prompt = row.get("prompt") or row.get("instruction") or row.get("input") or ""
    response = row.get("response") or row.get("output") or row.get("completion") or row.get("answer") or ""
    if not response and "messages" in row:
        msgs = row.get("messages") or []
        if msgs:
            prompt = "\n".join([m.get("content", "") for m in msgs[:-1]])
            response = msgs[-1].get("content", "") or ""
    return prompt, response


def run_sft(
    project_root: Path,
    cfg: ProjectConfig,
    data_dir: Path,
    model_id_or_path: str,
    accel: str,
    run_kind: str = "sft",
    metrics_kind: str | None = None,
) -> RunPaths:
    run = new_run(project_root, run_kind)
    snapshot_config(cfg.model_dump(), run.config_snapshot_path)

    backend = get_backend(accel)
    backend.patch()
    label = (run_kind if run_kind else "sft").upper()
    console.print(f"[bold]{label}[/bold] run: {run.run_dir.name}  accel={backend.name}")

    train_path = data_dir / "train.jsonl"
    if not train_path.exists():
        raise RuntimeError(
            "Missing train.jsonl. Run `mlxsmith data split` or point --data to a dir containing train.jsonl"
        )
    rows = _load_sft_rows(train_path)

    llm = get_llm_backend(cfg.model.backend)
    base_model, adapter_path, adapter_meta = resolve_model_spec(project_root, model_id_or_path, cfg)

    try:
        llm.load(
            base_model,
            max_seq_len=cfg.model.max_seq_len,
            dtype=cfg.model.dtype,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        if adapter_path:
            llm.apply_adapter(str(adapter_path))
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
            llm.apply_lora_from_config(lora_cfg)
    except BackendNotAvailable as e:
        console.print(f"[yellow]MLX backend unavailable[/yellow]: {e}")
        (run.adapter_dir / "ADAPTER.txt").write_text(
            f"Backend unavailable in this environment.\nmodel={model_id_or_path}\naccel={backend.name}\n",
            encoding="utf-8",
        )
        return run

    opt, _params = llm.optimizer_and_params(
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        optimizer=cfg.train.optimizer,
        optimizer_kwargs=cfg.train.optimizer_kwargs,
    )

    total = int(cfg.train.iters)
    grad_accum = max(1, int(cfg.train.grad_accum))
    train_on_prompt = bool(getattr(cfg.train, "train_on_prompt", False))
    max_grad_norm = float(getattr(cfg.train, "max_grad_norm", 1.0))

    rng = random.Random(cfg.train.seed)
    accum_grads = None
    accum_loss = 0.0
    accum_count = 0

    metrics_kind = metrics_kind or run_kind or "sft"

    for step in range(1, total + 1):
        row = rng.choice(rows)
        prompt, response = _row_to_prompt_response(row)
        if not response:
            continue

        text = f"{prompt}{response}"
        prompt_ids = llm.encode(prompt)
        ids = llm.encode(text)
        max_len = int(cfg.model.max_seq_len)
        if max_len and len(ids) > max_len:
            overflow = len(ids) - max_len
            ids = ids[overflow:]
            prompt_ids = prompt_ids[overflow:] if overflow < len(prompt_ids) else []

        def loss_fn(_model):
            return llm.sft_loss(ids, train_on_prompt=train_on_prompt, prompt_len=len(prompt_ids))

        lval, grads = llm.value_and_grad(loss_fn)
        accum_loss += float(lval.item()) if hasattr(lval, "item") else float(lval)
        accum_count += 1
        if grads is not None:
            accum_grads = tree_add(accum_grads, grads)

        if step % grad_accum == 0:
            if accum_grads is not None:
                scaled = tree_scale(accum_grads, 1.0 / grad_accum)
                if max_grad_norm > 0:
                    scaled = clip_grad_norm(scaled, max_grad_norm)
                llm.apply_grads(opt, scaled)
            accum_grads = None
            accum_loss = 0.0
            accum_count = 0

        if step % cfg.train.log_every == 0 or step == 1 or step == total:
            avg_loss = (accum_loss / max(1, accum_count)) if accum_count else (
                float(lval.item()) if hasattr(lval, "item") else float(lval)
            )
            write_jsonl(
                run.metrics_path,
                [
                    {
                        "ts": now_ts(),
                        "step": step,
                        "kind": metrics_kind,
                        "loss": avg_loss,
                        "accel": backend.name,
                    }
                ],
            )

        if step % cfg.train.save_every == 0 or step == total:
            llm.save_adapter(
                str(run.adapter_dir),
                metadata={
                    "base_model": base_model,
                    "source_adapter": str(adapter_path) if adapter_path else None,
                    "run": run.run_dir.name,
                    "kind": metrics_kind,
                },
            )

    console.print(f"[green]Saved adapter[/green] {run.adapter_dir}")
    return run
