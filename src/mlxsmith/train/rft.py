from __future__ import annotations

import random
import time
from pathlib import Path

from rich.console import Console

from ..accel import get_backend
from ..config import ProjectConfig
from ..models import resolve_model_spec
from ..runs import RunPaths, new_run, snapshot_config
from ..util import ensure_dir, write_jsonl, now_ts, sha1_text
from ..llm.registry import get_llm_backend
from ..llm.backend import BackendNotAvailable
from .lora import LoRAConfig

console = Console()


def load_verifier(verifier_path: Path):
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


def run_rft(project_root: Path, cfg: ProjectConfig, env_path: Path, verifier_path: Path, base_model_path: Path, accel: str) -> RunPaths:
    run = new_run(project_root, "rft")
    snapshot_config(cfg.model_dump(), run.config_snapshot_path)

    backend = get_backend(accel)
    backend.patch()
    console.print(f"[bold]RFT[/bold] run: {run.run_dir.name} algo={cfg.rft.algo} accel={backend.name}")

    verify = load_verifier(verifier_path)

    import yaml

    env = yaml.safe_load(env_path.read_text(encoding="utf-8")) or {}
    tasks = env.get("tasks") or []
    if not tasks:
        raise RuntimeError("Env has no tasks. Add `tasks:` list in env YAML.")

    accepted_path = run.run_dir / "accepted.jsonl"

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
    if cfg.rft.reference_model:
        ref_llm = get_llm_backend(cfg.model.backend)
        try:
            ref_llm.load(
                cfg.rft.reference_model,
                max_seq_len=cfg.model.max_seq_len,
                dtype=cfg.model.dtype,
                trust_remote_code=cfg.model.trust_remote_code,
            )
        except BackendNotAvailable:
            ref_llm = None

    opt, _params = llm.optimizer_and_params(lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    rng = random.Random(cfg.train.seed)
    total_iters = int(cfg.train.iters)
    rollouts = int(cfg.rft.rollouts)
    temperature = float(cfg.rft.temperature)
    max_new = int(getattr(cfg.rft, "max_new_tokens", 256))
    kl_coeff = float(cfg.rft.kl_coeff)
    normalize_adv = bool(cfg.rft.normalize_advantage)

    for step in range(1, total_iters + 1):
        task = tasks[(step - 1) % len(tasks)]
        prompt = task.get("prompt", "")
        task_id = task.get("id") or sha1_text(prompt)[:12]

        gens = []
        gen_tokens = 0
        gen_start = time.time()
        verifier_times = []

        for k in range(rollouts):
            gen = llm.generate(
                prompt,
                max_new_tokens=max_new,
                temperature=temperature,
                seed=rng.randint(0, 2**31 - 1),
            )
            completion = gen.text[len(prompt) :] if gen.text.startswith(prompt) else gen.text
            gen_tokens += max(0, len(gen.token_ids) - gen.prompt_len)

            wdir = ensure_dir(run.artifacts_dir / task_id / f"step_{step:06d}" / f"rollout_{k:02d}")
            if "tests" in task:
                tdir = ensure_dir(wdir / "tests")
                (tdir / "test_task.py").write_text(task["tests"], encoding="utf-8")

            t0 = time.time()
            res = verify(prompt, completion, str(wdir), **(task.get("verifier_kwargs") or {}))
            verifier_times.append(time.time() - t0)

            passed = bool(getattr(res, "passed", False))
            reward = float(getattr(res, "reward", 0.0))
            gens.append((gen, completion, passed, reward))

        gen_elapsed = max(time.time() - gen_start, 1e-6)
        tps = gen_tokens / gen_elapsed

        mean_r = sum(r for *_rest, r in gens) / max(1, len(gens))
        std_r = (
            sum((r - mean_r) ** 2 for *_rest, r in gens) / max(1, len(gens))
        ) ** 0.5
        advs = [r - mean_r for *_rest, r in gens]
        if normalize_adv and std_r > 1e-6:
            advs = [a / std_r for a in advs]

        def loss_fn(_model):
            loss = llm.mx.array(0.0)  # type: ignore
            for (gen, _comp, _passed, _reward), adv in zip(gens, advs):
                logp = llm.sequence_logprob(gen.token_ids, prompt_len=gen.prompt_len)
                pg = -llm.mx.array(float(adv)) * logp  # type: ignore
                if ref_llm is not None and kl_coeff > 0:
                    ref_logp = ref_llm.sequence_logprob(gen.token_ids, prompt_len=gen.prompt_len)
                    pg = pg + llm.mx.array(kl_coeff) * (logp - ref_logp)  # type: ignore
                loss = loss + pg
            return loss / llm.mx.array(float(len(gens)))  # type: ignore

        lval, grads = llm.value_and_grad(loss_fn)
        if grads is not None:
            llm.apply_grads(opt, grads)

        best_idx = max(range(len(gens)), key=lambda i: gens[i][3])
        best = gens[best_idx]
        pass_at_1 = 1.0 if gens[0][2] else 0.0
        pass_at_k = 1.0 if any(passed for _g, _c, passed, _r in gens) else 0.0
        acceptance = sum(1 for *_rest, passed, _reward in gens if passed) / max(1, len(gens))

        if step % cfg.train.log_every == 0 or step == 1 or step == total_iters:
            write_jsonl(
                run.metrics_path,
                [
                    {
                        "ts": now_ts(),
                        "step": step,
                        "kind": "rft",
                        "algo": cfg.rft.algo,
                        "task_id": task_id,
                        "mean_reward": mean_r,
                        "std_reward": std_r,
                        "best_reward": best[3],
                        "best_passed": best[2],
                        "pass@1": pass_at_1,
                        "pass@k": pass_at_k,
                        "acceptance": acceptance,
                        "verifier_latency_ms": (sum(verifier_times) / max(1, len(verifier_times))) * 1000.0,
                        "tokens_per_sec": tps,
                        "loss": float(lval.item()) if hasattr(lval, "item") else float(lval),
                        "accel": backend.name,
                    }
                ],
            )

        for (gen, completion, passed, reward) in gens:
            if passed:
                write_jsonl(
                    accepted_path,
                    [
                        {
                            "prompt": prompt,
                            "response": completion,
                            "reward": reward,
                            "task_id": task_id,
                        }
                    ],
                )

        if step % cfg.train.save_every == 0 or step == total_iters:
            llm.save_adapter(
                str(run.adapter_dir),
                metadata={
                    "base_model": base_model,
                    "source_adapter": str(adapter_path) if adapter_path else None,
                    "run": run.run_dir.name,
                    "kind": "rft",
                },
            )

    console.print(f"[green]Saved adapter[/green] {run.adapter_dir}")
    return run
