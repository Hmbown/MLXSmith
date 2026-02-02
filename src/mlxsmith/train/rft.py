from __future__ import annotations

import random
import time
from pathlib import Path

from rich.console import Console

from ..accel import get_backend
from ..config import ProjectConfig
from ..models import resolve_model_spec
from ..runs import RunPaths, new_run, snapshot_config
from ..envs.token_env import TokenEnvStep, StringTaskTokenEnv, create_token_env, load_token_env_spec
from ..util import ensure_dir, write_jsonl, now_ts, sha1_text, latency_summary_ms
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


def _normalize_observation(obs: list[int] | TokenEnvStep) -> tuple[list[int], float, bool, dict]:
    if isinstance(obs, TokenEnvStep):
        return list(obs.observation), float(obs.reward), bool(obs.done), dict(obs.info or {})
    return list(obs), 0.0, False, {}


def _rollout_token_env(
    llm,
    env,
    *,
    max_steps: int,
    temperature: float,
    seed: int,
) -> tuple[list[int], int, str, float, dict, int]:
    obs = env.initial_observation()
    obs_tokens, reward, done, info = _normalize_observation(obs)
    prompt_len = len(obs_tokens)
    full_tokens = list(obs_tokens)
    gen_tokens = 0

    for idx in range(max_steps):
        if done:
            break
        prompt_text = llm.decode(obs_tokens)
        gen = llm.generate_with_logprobs(
            prompt_text,
            max_new_tokens=1,
            temperature=temperature,
            top_p=1.0,
            top_k_sampling=None,
            seed=(seed + idx) % (2**31 - 1),
            logprobs=0,
        )
        new_token = int(gen.token_ids[-1])
        full_tokens.append(new_token)
        gen_tokens += 1

        step = env.step(new_token)
        reward += float(step.reward)
        done = bool(step.done)
        info = dict(step.info or {})
        obs_tokens = list(step.observation) if step.observation else list(full_tokens)

    completion = llm.decode(full_tokens[prompt_len:])
    return full_tokens, prompt_len, completion, reward, info, gen_tokens


def run_rft(project_root: Path, cfg: ProjectConfig, env_path: Path, verifier_path: Path, base_model_path: Path, accel: str) -> RunPaths:
    run = new_run(project_root, "rft")
    snapshot_config(cfg.model_dump(), run.config_snapshot_path)

    backend = get_backend(accel)
    backend.patch()
    console.print(f"[bold]RFT[/bold] run: {run.run_dir.name} algo={cfg.rft.algo} accel={backend.name}")

    verify = load_verifier(verifier_path)

    import yaml

    env = yaml.safe_load(env_path.read_text(encoding="utf-8")) or {}
    token_env_spec = load_token_env_spec(project_root, env)
    tasks = env.get("tasks") or []
    if token_env_spec is None and not tasks:
        raise RuntimeError("Env has no tasks. Add `tasks:` list in env YAML.")
    if token_env_spec is not None and token_env_spec.kind == "tasks" and not tasks:
        raise RuntimeError("token_env is set to tasks shim but env has no tasks.")

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

    if token_env_spec is not None:
        base_name = env.get("name") or "token_env"
        eos_token_id = getattr(getattr(llm, "tokenizer", None), "eos_token_id", None)

        for step in range(1, total_iters + 1):
            if token_env_spec.kind == "tasks":
                task = tasks[(step - 1) % len(tasks)]
                prompt = task.get("prompt", "")
                task_id = task.get("id") or sha1_text(prompt)[:12]
                tests = task.get("tests", "")
                verifier_kwargs = task.get("verifier_kwargs") or {}
            else:
                prompt = ""
                tests = ""
                verifier_kwargs = {}
                task_id = f"{base_name}_{step:06d}"

            gens = []
            gen_tokens = 0
            gen_start = time.time()
            verifier_latencies_ms: list[float] = []
            per_verifier_latencies: dict[str, list[float]] = {}

            for k in range(rollouts):
                wdir = ensure_dir(run.artifacts_dir / task_id / f"step_{step:06d}" / f"rollout_{k:02d}")
                if token_env_spec.kind == "tasks":
                    env_instance = StringTaskTokenEnv(
                        prompt=prompt,
                        tests=tests,
                        verifier_fn=verify,
                        workdir=wdir,
                        max_steps=max_new,
                        encode=llm.encode,
                        decode=llm.decode,
                        verifier_kwargs=verifier_kwargs,
                        eos_token_id=eos_token_id,
                    )
                else:
                    env_instance = create_token_env(
                        token_env_spec,
                        workdir=wdir,
                        encode=llm.encode,
                        decode=llm.decode,
                        tokenizer=getattr(llm, "tokenizer", None),
                        max_steps=max_new,
                        seed=rng.randint(0, 2**31 - 1),
                    )

                token_ids, prompt_len, completion, reward, info, gen_count = _rollout_token_env(
                    llm,
                    env_instance,
                    max_steps=max_new,
                    temperature=temperature,
                    seed=rng.randint(0, 2**31 - 1),
                )
                gen_tokens += gen_count

                verifier_latency = info.get("verifier_latency_ms")
                if verifier_latency is not None:
                    verifier_latencies_ms.append(float(verifier_latency))
                per_lat = info.get("verifier_latencies_ms")
                if isinstance(per_lat, dict):
                    for path, val in per_lat.items():
                        try:
                            per_verifier_latencies.setdefault(str(path), []).append(float(val))
                        except (TypeError, ValueError):
                            continue

                passed = bool(info.get("passed", reward > 0.0))
                gens.append((token_ids, prompt_len, completion, passed, reward, info))

            gen_elapsed = max(time.time() - gen_start, 1e-6)
            tps = gen_tokens / gen_elapsed

            mean_r = sum(r for *_rest, r, _info in gens) / max(1, len(gens))
            std_r = (
                sum((r - mean_r) ** 2 for *_rest, r, _info in gens) / max(1, len(gens))
            ) ** 0.5
            advs = [r - mean_r for *_rest, r, _info in gens]
            if normalize_adv and std_r > 1e-6:
                advs = [a / std_r for a in advs]

            def loss_fn(_model):
                loss = llm.mx.array(0.0)  # type: ignore
                for (token_ids, prompt_len, _comp, _passed, _reward, _info), adv in zip(gens, advs):
                    logp = llm.sequence_logprob(token_ids, prompt_len=prompt_len)
                    pg = -llm.mx.array(float(adv)) * logp  # type: ignore
                    if ref_llm is not None and kl_coeff > 0:
                        ref_logp = ref_llm.sequence_logprob(token_ids, prompt_len=prompt_len)
                        pg = pg + llm.mx.array(kl_coeff) * (logp - ref_logp)  # type: ignore
                    loss = loss + pg
                return loss / llm.mx.array(float(len(gens)))  # type: ignore

            lval, grads = llm.value_and_grad(loss_fn)
            if grads is not None:
                llm.apply_grads(opt, grads)

            best_idx = max(range(len(gens)), key=lambda i: gens[i][4])
            best = gens[best_idx]
            pass_at_1 = 1.0 if gens[0][3] else 0.0
            pass_at_k = 1.0 if any(passed for *_g, passed, _r, _i in gens) else 0.0
            acceptance = sum(1 for *_g, passed, _r, _i in gens if passed) / max(1, len(gens))

            latency_summary = latency_summary_ms(verifier_latencies_ms)
            per_verifier_summary = {
                path: latency_summary_ms(vals) for path, vals in per_verifier_latencies.items()
            }

            if step % cfg.train.log_every == 0 or step == 1 or step == total_iters:
                metrics = {
                    "ts": now_ts(),
                    "step": step,
                    "kind": "rft",
                    "algo": cfg.rft.algo,
                    "task_id": task_id,
                    "mean_reward": mean_r,
                    "std_reward": std_r,
                    "best_reward": best[4],
                    "best_passed": best[3],
                    "pass@1": pass_at_1,
                    "pass@k": pass_at_k,
                    "acceptance": acceptance,
                    "tokens_per_sec": tps,
                    "loss": float(lval.item()) if hasattr(lval, "item") else float(lval),
                    "accel": backend.name,
                }
                if latency_summary:
                    metrics["verifier_latency_ms"] = latency_summary["mean"]
                    for key, val in latency_summary.items():
                        metrics[f"verifier_latency_ms_{key}"] = val
                if per_verifier_summary:
                    metrics["verifier_latency_ms_by_path"] = per_verifier_summary
                write_jsonl(run.metrics_path, [metrics])

            for (token_ids, prompt_len, completion, passed, reward, _info) in gens:
                if passed:
                    prompt_text = llm.decode(token_ids[:prompt_len]) if prompt_len > 0 else ""
                    write_jsonl(
                        accepted_path,
                        [
                            {
                                "prompt": prompt_text,
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

    for step in range(1, total_iters + 1):
        task = tasks[(step - 1) % len(tasks)]
        prompt = task.get("prompt", "")
        task_id = task.get("id") or sha1_text(prompt)[:12]

        gens = []
        gen_tokens = 0
        gen_start = time.time()
        verifier_times = []
        per_verifier_latencies: dict[str, list[float]] = {}

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
            per_lat = getattr(res, "info", {}) or {}
            per_lat = per_lat.get("verifier_latencies_ms") if isinstance(per_lat, dict) else None
            if isinstance(per_lat, dict):
                for path, val in per_lat.items():
                    try:
                        per_verifier_latencies.setdefault(str(path), []).append(float(val))
                    except (TypeError, ValueError):
                        continue

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

        latency_summary = latency_summary_ms([t * 1000.0 for t in verifier_times])
        per_verifier_summary = {
            path: latency_summary_ms(vals) for path, vals in per_verifier_latencies.items()
        }

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
                        "verifier_latency_ms": latency_summary.get("mean", 0.0),
                        "verifier_latency_ms_mean": latency_summary.get("mean", 0.0),
                        "verifier_latency_ms_p50": latency_summary.get("p50", 0.0),
                        "verifier_latency_ms_p90": latency_summary.get("p90", 0.0),
                        "verifier_latency_ms_p99": latency_summary.get("p99", 0.0),
                        "verifier_latency_ms_max": latency_summary.get("max", 0.0),
                        "verifier_latency_ms_by_path": per_verifier_summary,
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
