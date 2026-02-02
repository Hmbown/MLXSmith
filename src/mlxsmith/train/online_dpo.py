from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Optional

from rich.console import Console

from ..accel import get_backend
from ..config import ProjectConfig
from ..models import resolve_model_spec
from ..runs import RunPaths, new_run, snapshot_config
from ..util import write_jsonl, now_ts, tree_add, tree_scale, clip_grad_norm
from ..llm.registry import get_llm_backend
from ..llm.backend import BackendNotAvailable
from ..sdk.losses import preference_loss
from ..verifiers.llm_judge import verify as judge_verify
from .lora import LoRAConfig

console = Console()


def _iter_prompts(path: Path) -> Iterable[str]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = row.get("prompt") or row.get("instruction") or row.get("input") or row.get("question") or ""
        if not prompt and "messages" in row:
            msgs = row.get("messages") or []
            if msgs:
                prompt = "\n".join([m.get("content", "") for m in msgs])
        if prompt:
            yield str(prompt)


def run_online_dpo(
    project_root: Path,
    cfg: ProjectConfig,
    data_path: Path,
    model_id_or_path: str,
    accel: str,
    *,
    judge_model: Optional[str] = None,
    judge_backend: str = "mlx-lm",
    rubric: Optional[str] = None,
    group_size: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    judge_mock_response: Optional[str | list[str]] = None,
) -> RunPaths:
    run = new_run(project_root, "online_dpo")
    snapshot_config(cfg.model_dump(), run.config_snapshot_path)

    prompts = list(_iter_prompts(data_path))
    if not prompts:
        raise RuntimeError("No prompts found in online DPO dataset")

    backend = get_backend(accel)
    backend.patch()
    console.print(f"[bold]ONLINE-DPO[/bold] run: {run.run_dir.name} accel={backend.name}")

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
    if cfg.pref.reference_model:
        ref_llm = get_llm_backend(cfg.model.backend)
        try:
            ref_llm.load(
                cfg.pref.reference_model,
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

    loss_type = str(cfg.pref.loss_type or cfg.pref.algo)
    beta = float(cfg.pref.beta)
    kl_coeff = float(cfg.pref.kl_coeff)
    delta = float(cfg.pref.delta)

    total = int(cfg.train.iters)
    grad_accum = max(1, int(cfg.train.grad_accum))
    max_grad_norm = float(getattr(cfg.train, "max_grad_norm", 0.0))
    group = int(group_size or cfg.rft.rollouts or 4)
    max_new = int(max_new_tokens or cfg.rft.max_new_tokens)
    temp = float(temperature if temperature is not None else cfg.rft.temperature)

    rng = random.Random(cfg.train.seed)
    accum_grads = None
    accum_loss = 0.0
    accum_count = 0

    def _next_mock(idx: int) -> Optional[str]:
        if judge_mock_response is None:
            return None
        if isinstance(judge_mock_response, list):
            if not judge_mock_response:
                return None
            return judge_mock_response[min(idx, len(judge_mock_response) - 1)]
        return judge_mock_response

    for step in range(1, total + 1):
        prompt = rng.choice(prompts)
        candidates: list[tuple[str, float]] = []
        for k in range(group):
            gen = llm.generate(
                prompt,
                max_new_tokens=max_new,
                temperature=temp,
                seed=rng.randint(0, 2**31 - 1),
            )
            completion = gen.text[len(prompt) :] if gen.text.startswith(prompt) else gen.text
            res = judge_verify(
                prompt,
                completion,
                str(run.artifacts_dir),
                model=judge_model,
                backend=judge_backend,
                rubric=rubric,
                reward_mode="score",
                mock_response=_next_mock(k),
            )
            reward = float(getattr(res, "reward", 0.0))
            candidates.append((completion, reward))

        if len(candidates) < 2:
            continue
        chosen, chosen_r = max(candidates, key=lambda x: x[1])
        rejected, rejected_r = min(candidates, key=lambda x: x[1])
        if chosen == rejected:
            continue

        prompt_ids = llm.encode(prompt)
        chosen_ids = llm.encode(prompt + chosen)
        rejected_ids = llm.encode(prompt + rejected)
        p_len_c = len(prompt_ids)
        p_len_r = len(prompt_ids)
        max_len = int(cfg.model.max_seq_len)
        if max_len:
            if len(chosen_ids) > max_len:
                overflow = len(chosen_ids) - max_len
                chosen_ids = chosen_ids[overflow:]
                p_len_c = max(0, p_len_c - overflow)
            if len(rejected_ids) > max_len:
                overflow = len(rejected_ids) - max_len
                rejected_ids = rejected_ids[overflow:]
                p_len_r = max(0, p_len_r - overflow)

        def loss_fn(_model):
            return preference_loss(
                llm,
                chosen_ids,
                rejected_ids,
                prompt_len_chosen=p_len_c,
                prompt_len_rejected=p_len_r,
                algo=loss_type,
                beta=beta,
                reference_backend=ref_llm,
                kl_coeff=kl_coeff,
                train_on_prompt=bool(cfg.train.train_on_prompt),
                delta=delta,
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
            avg_loss = accum_loss / max(1, accum_count) if accum_count else float(lval)
            write_jsonl(
                run.metrics_path,
                [
                    {
                        "ts": now_ts(),
                        "step": step,
                        "kind": "online_dpo",
                        "algo": loss_type,
                        "loss": avg_loss,
                        "reward_best": chosen_r,
                        "reward_worst": rejected_r,
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
                    "kind": "online_dpo",
                },
            )

    console.print(f"[green]Saved adapter[/green] {run.adapter_dir}")
    return run
