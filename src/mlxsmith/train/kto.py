from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from rich.console import Console

from ..accel import get_backend
from ..config import ProjectConfig
from ..models import resolve_model_spec
from ..runs import RunPaths, new_run, snapshot_config
from ..util import write_jsonl, now_ts, tree_add, tree_scale, clip_grad_norm
from ..llm.registry import get_llm_backend
from ..llm.backend import BackendNotAvailable
from ..sdk.losses import kto_loss
from .lora import LoRAConfig

console = Console()


def _load_kto_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _to_bool(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(float(value) >= 0.5)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "y", "positive", "pos", "good", "accepted", "preferred"):
            return True
        if v in ("0", "false", "no", "n", "negative", "neg", "bad", "rejected"):
            return False
    return None


def _row_to_prompt_response_label(row: dict) -> tuple[str, str, Optional[bool]]:
    prompt = row.get("prompt") or row.get("instruction") or row.get("input") or row.get("question") or ""
    response = row.get("response") or row.get("output") or row.get("completion") or row.get("answer") or ""
    if not response and "messages" in row:
        msgs = row.get("messages") or []
        if msgs:
            prompt = "\n".join([m.get("content", "") for m in msgs[:-1]])
            response = msgs[-1].get("content", "") or ""
    label = row.get("label")
    if label is None:
        label = row.get("desired") or row.get("accepted") or row.get("preferred") or row.get("is_positive")
    if label is None and "score" in row:
        label = row.get("score")
    if label is None and "reward" in row:
        label = row.get("reward")
    desired = _to_bool(label)
    return prompt, response, desired


def run_kto(
    project_root: Path,
    cfg: ProjectConfig,
    data_path: Path,
    model_id_or_path: str,
    accel: str,
) -> RunPaths:
    run = new_run(project_root, "kto")
    snapshot_config(cfg.model_dump(), run.config_snapshot_path)

    backend = get_backend(accel)
    backend.patch()
    console.print(f"[bold]KTO[/bold] run: {run.run_dir.name} accel={backend.name}")

    if data_path.is_dir():
        data_path = data_path / "train.jsonl"
    if not data_path.exists():
        raise RuntimeError("Missing training data. Expect JSONL with {prompt, response, label}")
    rows = _load_kto_rows(data_path)
    if not rows:
        raise RuntimeError("No rows found in KTO dataset")

    llm = get_llm_backend(cfg.model.backend)
    base_model, adapter_path, _meta = resolve_model_spec(project_root, model_id_or_path, cfg)

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

    ref_llm = None
    if getattr(cfg, "kto", None) and cfg.kto.reference_model:
        ref_llm = get_llm_backend(cfg.model.backend)
        try:
            ref_llm.load(
                cfg.kto.reference_model,
                max_seq_len=cfg.model.max_seq_len,
                dtype=cfg.model.dtype,
                trust_remote_code=cfg.model.trust_remote_code,
            )
        except BackendNotAvailable:
            ref_llm = None

    opt, _params = llm.optimizer_and_params(
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        optimizer=cfg.train.optimizer,
        optimizer_kwargs=cfg.train.optimizer_kwargs,
    )

    total = int(cfg.train.iters)
    grad_accum = max(1, int(cfg.train.grad_accum))
    max_grad_norm = float(getattr(cfg.train, "max_grad_norm", 1.0))

    rng = random.Random(cfg.train.seed)
    accum_grads = None
    accum_loss = 0.0
    accum_count = 0

    for step in range(1, total + 1):
        row = rng.choice(rows)
        prompt, response, desired = _row_to_prompt_response_label(row)
        if not response or desired is None:
            continue

        prompt_ids = llm.encode(prompt)
        ids = llm.encode(prompt + response)
        prompt_len = len(prompt_ids)
        max_len = int(cfg.model.max_seq_len)
        if max_len and len(ids) > max_len:
            overflow = len(ids) - max_len
            ids = ids[overflow:]
            prompt_len = max(0, prompt_len - overflow)

        def loss_fn(_model):
            return kto_loss(
                llm,
                ids,
                prompt_len=prompt_len,
                desired=desired,
                reference_backend=ref_llm,
                beta=float(cfg.kto.beta),
                gain_power=float(cfg.kto.gain_power),
                loss_power=float(cfg.kto.loss_power),
                loss_aversion=float(cfg.kto.loss_aversion),
                reference_point=float(cfg.kto.reference_point),
            )

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
            avg_loss = accum_loss / max(1, accum_count) if accum_count else (
                float(lval.item()) if hasattr(lval, "item") else float(lval)
            )
            write_jsonl(
                run.metrics_path,
                [
                    {
                        "ts": now_ts(),
                        "step": step,
                        "kind": "kto",
                        "loss": avg_loss,
                        "desired": bool(desired),
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
                    "kind": "kto",
                },
            )

    console.print(f"[green]Saved adapter[/green] {run.adapter_dir}")
    return run
