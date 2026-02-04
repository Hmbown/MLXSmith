from __future__ import annotations

import json
import time
from pathlib import Path

from .config import ProjectConfig
from .models import resolve_model_spec
from .util import ensure_dir, now_ts
from .llm.registry import get_llm_backend
from .accel import get_backend


def run_bench(
    project_root: Path,
    cfg: ProjectConfig,
    model_id_or_path: str,
    accel: str,
    *,
    prompt: str,
    max_tokens: int,
    reps: int,
    mode: str = "inference",
    steps: int = 5,
) -> Path:
    out_dir = ensure_dir(project_root / "bench")
    out_path = out_dir / f"bench_{now_ts()}.json"

    accel_backend = get_backend(accel)
    accel_backend.patch()

    llm = get_llm_backend(cfg.model.backend)
    base_model, adapter_path, _meta = resolve_model_spec(project_root, model_id_or_path, cfg)
    llm.load(
        base_model,
        max_seq_len=cfg.model.max_seq_len,
        dtype=cfg.model.dtype,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if adapter_path:
        llm.apply_adapter(str(adapter_path))
    if getattr(cfg.accel, "mhc", False):
        try:
            from .mhc import apply_mhc

            model = getattr(llm, "model", None)
            if model is not None:
                apply_mhc(
                    model,
                    n=int(getattr(cfg.accel, "mhc_n", 4)),
                    tmax=int(getattr(cfg.accel, "mhc_tmax", 20)),
                    verbose=False,
                )
        except Exception as e:
            raise RuntimeError(f"Failed to apply mHC adapters: {e}") from e

    results = []
    mode = (mode or "inference").lower()

    if mode == "trainer":
        opt, _params = llm.optimizer_and_params(
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            optimizer=cfg.train.optimizer,
            optimizer_kwargs=cfg.train.optimizer_kwargs,
        )
        prompt_ids = llm.encode(prompt)
        ids = llm.encode(prompt + " " + "x" * max_tokens)
        for i in range(max(1, reps)):
            t0 = time.time()
            for _ in range(max(1, steps)):
                def loss_fn(_model):
                    return llm.sft_loss(ids, train_on_prompt=cfg.train.train_on_prompt, prompt_len=len(prompt_ids))

                _loss, grads = llm.value_and_grad(loss_fn)
                if grads is not None:
                    llm.apply_grads(opt, grads)
            elapsed = max(time.time() - t0, 1e-6)
            results.append({"rep": i, "steps": steps, "time_s": elapsed, "steps_per_s": steps / elapsed})
    elif mode == "end_to_end":
        opt, _params = llm.optimizer_and_params(
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            optimizer=cfg.train.optimizer,
            optimizer_kwargs=cfg.train.optimizer_kwargs,
        )
        for i in range(max(1, reps)):
            t0 = time.time()
            gen = llm.generate(prompt, max_new_tokens=max_tokens, temperature=0.0)
            def loss_fn(_model):
                return llm.rl_loss(gen.token_ids, prompt_len=gen.prompt_len, advantage=1.0)
            _loss, grads = llm.value_and_grad(loss_fn)
            if grads is not None:
                llm.apply_grads(opt, grads)
            elapsed = max(time.time() - t0, 1e-6)
            gen_tokens = max(0, len(gen.token_ids) - gen.prompt_len)
            results.append({"rep": i, "tokens": gen_tokens, "time_s": elapsed, "tps": gen_tokens / elapsed})
    else:
        for i in range(max(1, reps)):
            t0 = time.time()
            gen = llm.generate(prompt, max_new_tokens=max_tokens, temperature=0.0)
            elapsed = max(time.time() - t0, 1e-6)
            gen_tokens = max(0, len(gen.token_ids) - gen.prompt_len)
            results.append({"rep": i, "tokens": gen_tokens, "time_s": elapsed, "tps": gen_tokens / elapsed})

    if mode == "trainer":
        avg_metric = sum(r["steps_per_s"] for r in results) / max(1, len(results))
        metric_name = "avg_steps_per_s"
    else:
        avg_metric = sum(r["tps"] for r in results) / max(1, len(results))
        metric_name = "avg_tps"

    summary = {
        "model": base_model,
        "adapter": str(adapter_path) if adapter_path else None,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "reps": reps,
        "mode": mode,
        "steps": steps if mode == "trainer" else None,
        "results": results,
        metric_name: avg_metric,
        "accel": accel_backend.name,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path
