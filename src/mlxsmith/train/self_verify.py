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


def run_self_verify(
    project_root: Path,
    cfg: ProjectConfig,
    data_path: Path,
    model_id_or_path: str,
    accel: str,
    *,
    verifier_model: Optional[str] = None,
    verifier_backend: str = "mlx-lm",
    rubric: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    judge_mock_response: Optional[str | list[str]] = None,
) -> RunPaths:
    run = new_run(project_root, "self_verify")
    snapshot_config(cfg.model_dump(), run.config_snapshot_path)

    prompts = list(_iter_prompts(data_path))
    if not prompts:
        raise RuntimeError("No prompts found in self-verify dataset")

    backend = get_backend(accel)
    backend.patch()
    console.print(f"[bold]SELF-VERIFY[/bold] run: {run.run_dir.name} accel={backend.name}")

    policy = get_llm_backend(cfg.model.backend)
    base_model, adapter_path, _meta = resolve_model_spec(project_root, model_id_or_path, cfg)

    try:
        policy.load(
            base_model,
            max_seq_len=cfg.model.max_seq_len,
            dtype=cfg.model.dtype,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        if adapter_path:
            policy.apply_adapter(str(adapter_path))
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
            policy.apply_lora_from_config(lora_cfg)
    except BackendNotAvailable as e:
        console.print(f"[yellow]MLX backend unavailable[/yellow]: {e}")
        (run.adapter_dir / "ADAPTER.txt").write_text(
            f"Backend unavailable in this environment.\nmodel={model_id_or_path}\naccel={backend.name}\n",
            encoding="utf-8",
        )
        return run

    opt, _params = policy.optimizer_and_params(
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        optimizer=cfg.train.optimizer,
        optimizer_kwargs=cfg.train.optimizer_kwargs,
    )

    total = int(cfg.train.iters)
    grad_accum = max(1, int(cfg.train.grad_accum))
    max_grad_norm = float(getattr(cfg.train, "max_grad_norm", 0.0))
    max_new = int(max_new_tokens or cfg.rft.max_new_tokens)
    temp = float(temperature if temperature is not None else cfg.rft.temperature)

    rng = random.Random(cfg.train.seed)
    accum_grads = None
    accum_loss = 0.0
    accum_count = 0
    reward_ema = 0.0
    ema_alpha = 0.1

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
        gen = policy.generate_with_logprobs(
            prompt,
            max_new_tokens=max_new,
            temperature=temp,
            seed=rng.randint(0, 2**31 - 1),
            logprobs=0,
        )
        completion = gen.text[len(prompt) :] if gen.text.startswith(prompt) else gen.text

        res = judge_verify(
            prompt,
            completion,
            str(run.artifacts_dir),
            model=verifier_model or model_id_or_path,
            backend=verifier_backend,
            rubric=rubric,
            reward_mode="score",
            mock_response=_next_mock(0),
        )
        reward = float(getattr(res, "reward", 0.0))
        reward_ema = (1.0 - ema_alpha) * reward_ema + ema_alpha * reward
        advantage = reward - reward_ema

        token_ids = list(gen.token_ids)
        prompt_len = int(gen.prompt_len)

        def loss_fn(_model):
            logp = policy.sequence_logprob(token_ids, prompt_len=prompt_len)
            return -policy.mx.array(float(advantage)) * logp  # type: ignore

        lval, grads = policy.value_and_grad(loss_fn)
        accum_loss += float(lval.item()) if hasattr(lval, "item") else float(lval)
        accum_count += 1
        if grads is not None:
            accum_grads = tree_add(accum_grads, grads)

        if step % grad_accum == 0:
            if accum_grads is not None:
                scaled = tree_scale(accum_grads, 1.0 / grad_accum)
                if max_grad_norm > 0:
                    scaled = clip_grad_norm(scaled, max_grad_norm)
                policy.apply_grads(opt, scaled)
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
                        "kind": "self_verify",
                        "loss": avg_loss,
                        "reward": reward,
                        "advantage": advantage,
                        "accel": backend.name,
                    }
                ],
            )

        if step % cfg.train.save_every == 0 or step == total:
            policy.save_adapter(
                str(run.adapter_dir),
                metadata={
                    "base_model": base_model,
                    "source_adapter": str(adapter_path) if adapter_path else None,
                    "run": run.run_dir.name,
                    "kind": "self_verify",
                },
            )

    console.print(f"[green]Saved adapter[/green] {run.adapter_dir}")
    return run
