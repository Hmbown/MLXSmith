from __future__ import annotations

import json
import random
from pathlib import Path

from rich.console import Console

from ..accel import get_backend
from ..config import ProjectConfig
from ..models import resolve_model_spec
from ..runs import RunPaths, new_run, snapshot_config
from ..util import write_jsonl, now_ts, tree_add, tree_scale
from ..llm.registry import get_llm_backend
from ..llm.backend import BackendNotAvailable
from .lora import LoRAConfig

console = Console()


def _load_pref_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_pref(project_root: Path, cfg: ProjectConfig, data_dir: Path, base_model_path: Path, accel: str) -> RunPaths:
    run = new_run(project_root, "pref")
    snapshot_config(cfg.model_dump(), run.config_snapshot_path)

    backend = get_backend(accel)
    backend.patch()
    console.print(f"[bold]PREF[/bold] run: {run.run_dir.name} algo={cfg.pref.algo} accel={backend.name}")

    prefs_path = data_dir / "train.jsonl"
    if not prefs_path.exists():
        raise RuntimeError("Preference data missing. Expect data/prefs/train.jsonl with {prompt, chosen, rejected}")
    rows = _load_pref_rows(prefs_path)

    llm = get_llm_backend(cfg.model.backend)
    base_model, adapter_path, adapter_meta = resolve_model_spec(project_root, str(base_model_path), cfg)

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
            f"Backend unavailable in this environment.\nbase={base_model}\naccel={backend.name}\n",
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

    opt, _params = llm.optimizer_and_params(lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    beta = float(cfg.pref.beta)
    kl_coeff = float(cfg.pref.kl_coeff)
    rng = random.Random(cfg.train.seed)
    total = int(cfg.train.iters)
    grad_accum = max(1, int(cfg.train.grad_accum))
    train_on_prompt = bool(getattr(cfg.train, "train_on_prompt", False))

    accum_grads = None
    for step in range(1, total + 1):
        row = rng.choice(rows)
        prompt = row.get("prompt") or ""
        chosen = row.get("chosen") or row.get("accepted") or ""
        rejected = row.get("rejected") or row.get("rejected_response") or ""
        if not (prompt and chosen and rejected):
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
            logp_c = llm.sequence_logprob(chosen_ids, prompt_len=p_len_c)
            logp_r = llm.sequence_logprob(rejected_ids, prompt_len=p_len_r)
            ref_diff = 0.0
            if ref_llm is not None:
                ref_logp_c = ref_llm.sequence_logprob(chosen_ids, prompt_len=p_len_c)
                ref_logp_r = ref_llm.sequence_logprob(rejected_ids, prompt_len=p_len_r)
                ref_diff = ref_logp_c - ref_logp_r
            diff = (logp_c - logp_r) - ref_diff

            if cfg.pref.algo == "orpo":
                # ORPO loss = NLL(chosen) - beta * log(sigmoid(diff))
                nll = llm.sft_loss(chosen_ids, train_on_prompt=train_on_prompt, prompt_len=p_len_c)
                or_loss = -beta * llm.mx.log(llm.mx.sigmoid(diff))  # type: ignore
                loss = nll + or_loss
            else:
                # DPO loss
                scaled = llm.mx.array(beta) * diff  # type: ignore
                loss = llm.mx.log1p(llm.mx.exp(-scaled))  # type: ignore

            if ref_llm is not None and kl_coeff > 0:
                # Simple KL penalty on chosen responses
                kl = (logp_c - ref_logp_c) if ref_llm is not None else 0.0
                loss = loss + llm.mx.array(kl_coeff) * kl  # type: ignore
            return loss

        lval, grads = llm.value_and_grad(loss_fn)
        if grads is not None:
            accum_grads = tree_add(accum_grads, grads)

        if step % grad_accum == 0:
            if accum_grads is not None:
                llm.apply_grads(opt, tree_scale(accum_grads, 1.0 / grad_accum))
            accum_grads = None

        if step % cfg.train.log_every == 0 or step == 1 or step == total:
            write_jsonl(
                run.metrics_path,
                [
                    {
                        "ts": now_ts(),
                        "step": step,
                        "kind": "pref",
                        "algo": cfg.pref.algo,
                        "beta": beta,
                        "kl_coeff": kl_coeff,
                        "loss": float(lval.item()) if hasattr(lval, "item") else float(lval),
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
                    "kind": "pref",
                },
            )

    console.print(f"[green]Saved adapter[/green] {run.adapter_dir}")
    return run
