from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from ..config import ProjectConfig
from ..util import ensure_dir, now_ts
from ..verifiers.docker_verifier import verify as docker_verify
from ..verifiers.pytest_verifier import verify as pytest_verify
from .generate import GeneratedTask, generate_tasks, filter_tasks
from .recursive import recursive_compact
from .mutate import mutate_tasks


@dataclass
class Rollout:
    task_id: str
    prompt: str
    completion: str
    token_ids: list[int]
    prompt_len: int
    logprobs: Optional[list[float]]
    passed: bool
    reward: float
    verifier_latency_ms: float
    weight_adapter: Optional[str]


def build_tasks(
    llm,
    cfg: ProjectConfig,
    *,
    require_recursion: bool,
    tasks_per_iter: int,
    mutations_per_task: int,
    max_total: int,
    existing_prompts: Optional[list[str]] = None,
) -> List[GeneratedTask]:
    tasks = generate_tasks(
        llm,
        tasks_per_iter=tasks_per_iter,
        temperature=float(cfg.rft.temperature),
        max_new_tokens=int(cfg.rlm.task_gen_max_new_tokens),
        top_p=float(cfg.infer.top_p),
        top_k=cfg.infer.top_k,
        require_recursion=require_recursion,
        task_domains=list(cfg.rlm.task_domains or []),
    )
    if bool(cfg.rlm.use_task_mutation) and int(mutations_per_task) > 0:
        tasks = mutate_tasks(
            llm,
            tasks,
            mutations_per_task=mutations_per_task,
            max_total=max_total,
            temperature=float(cfg.rft.temperature),
            max_new_tokens=int(cfg.rlm.task_gen_max_new_tokens),
            top_p=float(cfg.infer.top_p),
            top_k=cfg.infer.top_k,
            require_recursion=require_recursion,
        )
    filtered = filter_tasks(
        tasks,
        existing_prompts=existing_prompts,
        similarity_threshold=float(getattr(cfg.rlm, "similarity_threshold", 0.85)),
        min_desc_len=int(getattr(cfg.rlm, "min_task_desc_len", 10)),
        min_asserts=int(getattr(cfg.rlm, "min_task_asserts", 2)),
        max_prompt_len=int(getattr(cfg.rlm, "max_task_prompt_len", 2000)),
        min_tests_len=int(getattr(cfg.rlm, "min_task_tests_len", 20)),
        max_tests_len=int(getattr(cfg.rlm, "max_task_tests_len", 8000)),
        blocked_patterns=getattr(cfg.rlm, "blocked_task_patterns", None),
    )
    return filtered or tasks


def _write_task_tests(task: GeneratedTask, workdir: Path) -> None:
    tests_dir = ensure_dir(workdir / "tests")
    (tests_dir / "test_task.py").write_text(task.tests, encoding="utf-8")


def collect_rollouts(
    llm,
    tasks: Iterable[GeneratedTask],
    cfg: ProjectConfig,
    *,
    artifacts_dir: Path,
    verifier_backend: str,
    weight_adapter: Optional[str],
) -> tuple[List[Rollout], list[dict]]:
    rollouts: List[Rollout] = []
    passed_samples: list[dict] = []

    for task in tasks:
        for k in range(int(cfg.rlm.rollouts_per_task)):
            prompt_text = task.prompt
            if bool(getattr(cfg.rlm, "recursive_inference", False)):
                prompt_text, _stats = recursive_compact(
                    llm,
                    prompt_text,
                    max_seq_len=int(cfg.model.max_seq_len),
                    chunk_tokens=int(cfg.rlm.recursive_chunk_tokens),
                    overlap_tokens=int(cfg.rlm.recursive_overlap_tokens),
                    keep_last_tokens=int(cfg.rlm.recursive_keep_last_tokens),
                    summary_tokens=int(cfg.rlm.recursive_summary_tokens),
                    max_depth=int(cfg.rlm.recursive_max_depth),
                    temperature=float(cfg.rlm.recursive_temperature),
                    seed=int(time.time() * 1000) % (2**31 - 1),
                    summary_prompt=cfg.rlm.recursive_summary_prompt,
                )

            gen = llm.generate_with_logprobs(
                prompt_text,
                max_new_tokens=int(cfg.rft.max_new_tokens),
                temperature=float(cfg.rft.temperature),
                seed=int(time.time() * 1000) % (2**31 - 1),
            )
            completion = gen.text[len(prompt_text) :] if gen.text.startswith(prompt_text) else gen.text
            wdir = ensure_dir(artifacts_dir / task.id / f"rollout_{k:02d}")
            (wdir / "main.py").write_text(completion, encoding="utf-8")
            _write_task_tests(task, wdir)

            t0 = time.time()
            if verifier_backend == "docker":
                res = docker_verify(
                    task.prompt,
                    completion,
                    str(wdir),
                    timeout_s=int(cfg.rlm.verifier_timeout_s),
                    image=cfg.rlm.docker_image,
                    memory_mb=int(cfg.rlm.docker_memory_mb),
                    cpus=float(cfg.rlm.docker_cpus),
                    pids=int(cfg.rlm.docker_pids),
                )
            else:
                res = pytest_verify(task.prompt, completion, str(wdir), timeout_s=int(cfg.rlm.verifier_timeout_s))
            latency_ms = (time.time() - t0) * 1000.0

            passed = bool(getattr(res, "passed", False))
            reward = float(getattr(res, "reward", 0.0))
            rollouts.append(
                Rollout(
                    task_id=task.id,
                    prompt=prompt_text,
                    completion=completion,
                    token_ids=list(gen.token_ids),
                    prompt_len=int(gen.prompt_len),
                    logprobs=list(gen.logprobs) if gen.logprobs is not None else None,
                    passed=passed,
                    reward=reward,
                    verifier_latency_ms=latency_ms,
                    weight_adapter=weight_adapter,
                )
            )

            if passed:
                passed_samples.append(
                    {
                        "id": task.id,
                        "prompt": task.prompt,
                        "response": completion,
                        "reward": reward,
                        "ts": now_ts(),
                    }
                )

    return rollouts, passed_samples
