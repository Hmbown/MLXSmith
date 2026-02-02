"""RLM Loop - Orchestrator Entry Point for MLXSmith.

This module provides both:
1. Legacy single-process RLM loop (run_rlm)
2. Multi-process orchestrator mode (run_rlm_orchestrated)

The orchestrator mode splits the RLM loop into:
- Orchestrator Daemon: Queue-based job scheduler
- Inference Worker Process: OpenAI-compatible API server
- Trainer Worker Process: Training batch consumer
"""

from __future__ import annotations

import json
import multiprocessing as mp
import signal
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console

from ..config import ProjectConfig
from ..eval import run_eval
from ..llm.registry import get_llm_backend
from ..models import resolve_model_spec
from ..runs import new_run, snapshot_config
from ..train.lora import LoRAConfig
from ..util import copytree, ensure_dir, now_ts, write_jsonl
from ..verifiers.docker_verifier import verify as docker_verify
from ..verifiers.pytest_verifier import verify as pytest_verify
from .corpus import append_corpus, load_corpus, sample_corpus
from .gating import load_state, save_state, should_accept, update_state
from .generate import GeneratedTask
from .history import append_history
from .inference import Rollout, build_tasks
from .trainer import train_on_rollouts
from .weights import (
    WeightPointer,
    WeightPointerIPC,
    WeightPointerStore,
    load_pointer,
    save_pointer,
)

console = Console()


def _score_from_eval(result_path: Path) -> float:
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
        summary = data.get("summary") or []
        if not summary:
            return 0.0
        return sum(item.get("pass@k", 0.0) for item in summary) / max(1, len(summary))
    except Exception:
        return 0.0


def _load_suite_prompts(path: Path) -> list[str]:
    try:
        import yaml

        suite = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        tasks = suite.get("tasks") or []
        return [str(t.get("prompt")) for t in tasks if t.get("prompt")]
    except Exception:
        return []


# =============================================================================
# Legacy Single-Process RLM Loop
# =============================================================================

def run_rlm(
    project_root: Path,
    cfg: ProjectConfig,
    *,
    model_spec: Optional[str] = None,
    iterations: Optional[int] = None,
    resume: bool = False,
) -> None:
    """Run single-process RLM loop (legacy mode)."""
    rlm_cfg = cfg.rlm
    state_path = project_root / "runs" / "rlm_state.json"
    history_path = project_root / "runs" / "rlm_history.jsonl"
    corpus_path = project_root / "runs" / "rlm_corpus.jsonl"
    weights_dir = ensure_dir(project_root / "runs" / "rlm_weights")

    state = load_state(state_path)

    if model_spec is None:
        model_spec = cfg.model.id

    base_model, initial_adapter, _meta = resolve_model_spec(project_root, model_spec, cfg)
    infer_ptr_path = weights_dir / "infer.json"
    train_ptr_path = weights_dir / "train.json"
    infer_ptr = load_pointer(infer_ptr_path, base_model=base_model, name="inference")
    train_ptr = load_pointer(train_ptr_path, base_model=base_model, name="trainer")

    if initial_adapter and not infer_ptr.adapter_path:
        infer_ptr = WeightPointer(
            base_model=base_model,
            adapter_path=str(initial_adapter),
            iteration=state.last_iteration,
            updated_at=now_ts(),
            name="inference",
        )
        save_pointer(infer_ptr_path, infer_ptr)

    if initial_adapter and not train_ptr.adapter_path:
        train_ptr = WeightPointer(
            base_model=base_model,
            adapter_path=str(initial_adapter),
            iteration=state.last_iteration,
            updated_at=now_ts(),
            name="trainer",
        )
        save_pointer(train_ptr_path, train_ptr)

    if resume and state.current_adapter:
        train_ptr = WeightPointer(
            base_model=base_model,
            adapter_path=state.current_adapter,
            iteration=state.last_iteration,
            updated_at=now_ts(),
            name="trainer",
        )
        save_pointer(train_ptr_path, train_ptr)
        if not infer_ptr.adapter_path:
            infer_ptr = WeightPointer(
                base_model=base_model,
                adapter_path=state.current_adapter,
                iteration=state.last_iteration,
                updated_at=now_ts(),
                name="inference",
            )
            save_pointer(infer_ptr_path, infer_ptr)

    start_iter = state.last_iteration + 1 if resume else 1
    total_iters = iterations if iterations is not None else int(rlm_cfg.iterations)

    def iter_range():
        if total_iters == 0:
            i = start_iter
            while True:
                yield i
                i += 1
        else:
            for i in range(start_iter, start_iter + total_iters):
                yield i

    for iteration in iter_range():
        run = new_run(project_root, "rlm")
        snapshot_config(cfg.model_dump(), run.config_snapshot_path)
        console.print(f"[bold]RLM[/bold] iteration {iteration} run={run.run_dir.name}")

        infer_llm = get_llm_backend(cfg.model.backend)
        infer_llm.load(
            infer_ptr.base_model,
            max_seq_len=cfg.model.max_seq_len,
            dtype=cfg.model.dtype,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        if infer_ptr.adapter_path:
            infer_llm.apply_adapter(str(infer_ptr.adapter_path))

        train_llm = get_llm_backend(cfg.model.backend)
        train_llm.load(
            train_ptr.base_model,
            max_seq_len=cfg.model.max_seq_len,
            dtype=cfg.model.dtype,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        if train_ptr.adapter_path:
            train_llm.apply_adapter(str(train_ptr.adapter_path))
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
            train_llm.apply_lora_from_config(lora_cfg)

        ref_llm = None
        if cfg.rft.reference_model:
            ref_llm = get_llm_backend(cfg.model.backend)
            ref_llm.load(
                cfg.rft.reference_model,
                max_seq_len=cfg.model.max_seq_len,
                dtype=cfg.model.dtype,
                trust_remote_code=cfg.model.trust_remote_code,
            )

        opt, _params = train_llm.optimizer_and_params(lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

        corpus_rows = load_corpus(corpus_path, max_size=int(rlm_cfg.corpus_max))
        existing_prompts = [row.get("prompt", "") for row in corpus_rows if row.get("prompt")]
        if rlm_cfg.benchmark_suite:
            suite_path = project_root / rlm_cfg.benchmark_suite
            if suite_path.exists():
                existing_prompts.extend(_load_suite_prompts(suite_path))
        if rlm_cfg.holdout_suite:
            holdout_path = project_root / rlm_cfg.holdout_suite
            if holdout_path.exists():
                existing_prompts.extend(_load_suite_prompts(holdout_path))

        tasks = build_tasks(
            infer_llm,
            cfg,
            require_recursion=bool(rlm_cfg.require_recursion),
            tasks_per_iter=int(rlm_cfg.tasks_per_iter),
            mutations_per_task=int(rlm_cfg.mutations_per_task),
            max_total=int(rlm_cfg.tasks_per_iter),
            existing_prompts=existing_prompts,
        )

        write_jsonl(run.run_dir / "tasks.jsonl", [task.__dict__ for task in tasks])
        rollouts, passed_samples = collect_rollouts_via_api(
            tasks,
            cfg,
            api_url=f"http://localhost:{cfg.serve.port}",
            artifacts_dir=run.artifacts_dir,
            verifier_backend=str(rlm_cfg.verifier_backend),
            weight_adapter=infer_ptr.adapter_path,
        )

        metrics_rows = train_on_rollouts(
            train_llm,
            rollouts,
            cfg,
            optimizer=opt,
            train_adapter=train_ptr.adapter_path,
            ref_llm=ref_llm,
        )
        for row in metrics_rows:
            row["iteration"] = iteration
        write_jsonl(run.metrics_path, metrics_rows)

        if passed_samples:
            append_corpus(corpus_path, passed_samples, max_size=int(rlm_cfg.corpus_max))

        # Optional corpus rehearsal via SFT
        mix_ratio = float(rlm_cfg.mix_old_ratio)
        if mix_ratio > 0 and corpus_rows:
            n_samples = int(max(1, len(tasks) * mix_ratio))
            for row in sample_corpus(corpus_rows, n=n_samples, hard_ratio=float(rlm_cfg.hard_ratio)):
                prompt = row.get("prompt", "")
                response = row.get("response", "")
                if not prompt or not response:
                    continue
                prompt_ids = train_llm.encode(prompt)
                ids = train_llm.encode(prompt + response)
                max_len = int(cfg.model.max_seq_len)
                if max_len and len(ids) > max_len:
                    overflow = len(ids) - max_len
                    ids = ids[overflow:]
                    prompt_ids = prompt_ids[overflow:] if overflow < len(prompt_ids) else []

                def sft_loss_fn(_model):
                    return train_llm.sft_loss(ids, train_on_prompt=cfg.train.train_on_prompt, prompt_len=len(prompt_ids))

                lval, grads = train_llm.value_and_grad(sft_loss_fn)
                if grads is not None:
                    train_llm.apply_grads(opt, grads)

        train_llm.save_adapter(
            str(run.adapter_dir),
            metadata={
                "base_model": train_ptr.base_model,
                "source_adapter": str(train_ptr.adapter_path) if train_ptr.adapter_path else None,
                "run": run.run_dir.name,
                "kind": "rlm",
                "iteration": iteration,
            },
        )

        # Evaluate
        adapter_score = 0.0
        if rlm_cfg.benchmark_suite:
            suite_path = project_root / rlm_cfg.benchmark_suite
            if suite_path.exists():
                eval_path = run_eval(project_root, suite_path, run.adapter_dir)
                adapter_score = _score_from_eval(eval_path)

        holdout_score = None
        if rlm_cfg.holdout_suite:
            holdout_path = project_root / rlm_cfg.holdout_suite
            if holdout_path.exists():
                holdout_eval = run_eval(project_root, holdout_path, run.adapter_dir)
                holdout_score = _score_from_eval(holdout_eval)

        accepted = should_accept(
            adapter_score,
            state,
            mode=rlm_cfg.gating,
            threshold=float(rlm_cfg.gating_threshold),
            ema_alpha=float(rlm_cfg.gating_ema_alpha),
        )
        state = update_state(
            state,
            iteration=iteration,
            score=adapter_score,
            adapter_path=str(run.adapter_dir),
            accepted=accepted,
            ema_alpha=float(rlm_cfg.gating_ema_alpha),
        )
        save_state(state_path, state)

        if state.current_adapter:
            train_ptr = WeightPointer(
                base_model=base_model,
                adapter_path=state.current_adapter,
                iteration=iteration,
                updated_at=now_ts(),
                name="trainer",
            )
            save_pointer(train_ptr_path, train_ptr)

        infer_staleness = int(getattr(rlm_cfg, "infer_staleness", 0))
        if infer_staleness <= 0:
            infer_ptr = train_ptr
            save_pointer(infer_ptr_path, infer_ptr)
        else:
            lag = max(0, int(train_ptr.iteration) - int(infer_ptr.iteration))
            if lag >= infer_staleness:
                infer_ptr = train_ptr
                save_pointer(infer_ptr_path, infer_ptr)

        append_history(
            history_path,
            {
                "iteration": iteration,
                "timestamp": now_ts(),
                "adapter_score": adapter_score,
                "holdout_score": holdout_score,
                "best_score": state.best_score,
                "accepted": accepted,
                "adapter_dir": str(run.adapter_dir),
            },
        )

        gating_path = run.run_dir / "gating.json"
        gating_path.write_text(
            json.dumps(
                {
                    "iteration": iteration,
                    "accepted": accepted,
                    "adapter_score": adapter_score,
                    "holdout_score": holdout_score,
                    "best_score": state.best_score,
                    "current_adapter": state.current_adapter,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if rlm_cfg.sleep_between > 0:
            time.sleep(float(rlm_cfg.sleep_between))


# =============================================================================
# Multi-Process Orchestrated RLM
# =============================================================================

from ..orchestrator.queue import MessageQueue, MessageType, Message  # noqa: E402
from ..orchestrator.inference_worker import InferenceConfig, run_inference_worker  # noqa: E402
from ..orchestrator.trainer_worker import TrainerConfig, run_trainer_worker  # noqa: E402


@dataclass
class OrchestratorState:
    """State for the orchestrated RLM loop."""
    iteration: int = 0
    run_id: str = ""
    pending_rollouts: int = 0
    pending_training: bool = False
    current_adapter: Optional[str] = None
    best_score: float = 0.0


class RLMOrchestrator:
    """Multi-process RLM orchestrator.
    
    Spawns and manages inference and trainer processes,
    coordinates rollout generation and training via queues.
    """
    
    def __init__(
        self,
        project_root: Path,
        cfg: ProjectConfig,
        model_spec: str,
        iterations: int = 50,
        resume: bool = False,
    ):
        self.project_root = project_root
        self.cfg = cfg
        self.model_spec = model_spec
        self.iterations = iterations
        self.resume = resume

        self._base_model, self._initial_adapter, _ = resolve_model_spec(
            self.project_root, self.model_spec, self.cfg
        )
        self._rollout_timeout_s = 120.0
        self._train_timeout_s = 900.0
        
        # Paths
        self.state_path = project_root / "runs" / "rlm_state.json"
        self.history_path = project_root / "runs" / "rlm_history.jsonl"
        self.corpus_path = project_root / "runs" / "rlm_corpus.jsonl"
        self.weights_dir = ensure_dir(project_root / "runs" / "rlm_weights")
        
        # State
        self.gating_state = load_state(self.state_path)
        self.orchestrator_state = OrchestratorState()
        
        # IPC
        self.queue = MessageQueue(maxsize=10000)
        self._pointer_store = WeightPointerStore(self.weights_dir)
        
        # Processes
        self._inference_process: Optional[mp.Process] = None
        self._trainer_process: Optional[mp.Process] = None
        self._shutdown = False
        
        # Rollout buffer
        self._rollout_buffer: List[Rollout] = []
        self._passed_samples: List[Dict] = []
        
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(sig, frame):
            console.print("[yellow]Orchestrator received shutdown signal[/yellow]")
            self._shutdown = True
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def _start_inference_worker(self) -> None:
        """Start the inference worker process."""
        inf_config = InferenceConfig(
            model_spec=self.model_spec,
            backend=self.cfg.model.backend,
            host=self.cfg.serve.host,
            port=self.cfg.serve.port,
            max_seq_len=self.cfg.model.max_seq_len,
            dtype=self.cfg.model.dtype,
            trust_remote_code=self.cfg.model.trust_remote_code,
            use_chat_template=self.cfg.model.use_chat_template,
            weights_dir=self.weights_dir,
            hot_reload=True,
        )
        
        # Initialize inference pointer
        pointer = WeightPointerIPC(
            base_model=self._base_model,
            adapter_path=str(self._initial_adapter) if self._initial_adapter else None,
            iteration=self.gating_state.last_iteration,
            updated_at=now_ts(),
            version=self.gating_state.last_iteration,
            name="inference",
        )
        self._pointer_store.save(pointer)
        
        self._inference_process = mp.Process(
            target=run_inference_worker,
            args=(inf_config, self.queue),
            name="inference_worker",
            daemon=False,
        )
        self._inference_process.start()
        console.print(f"[green]Started inference worker (PID: {self._inference_process.pid})[/green]")
    
    def _start_trainer_worker(self) -> None:
        """Start the trainer worker process."""
        trainer_config = TrainerConfig(
            model_spec=self.model_spec,
            base_model=self._base_model,
            backend=self.cfg.model.backend,
            max_seq_len=self.cfg.model.max_seq_len,
            dtype=self.cfg.model.dtype,
            trust_remote_code=self.cfg.model.trust_remote_code,
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
            kl_coeff=self.cfg.rft.kl_coeff,
            normalize_advantage=self.cfg.rft.normalize_advantage,
            lora_r=self.cfg.lora.r,
            lora_alpha=self.cfg.lora.alpha,
            lora_dropout=self.cfg.lora.dropout,
            lora_target_modules=list(self.cfg.lora.target_modules or []),
            lora_num_layers=self.cfg.lora.num_layers,
            weights_dir=self.weights_dir,
            checkpoint_dir=self.project_root / "runs" / "rlm_checkpoints",
            reference_model=self.cfg.rft.reference_model,
        )
        
        # Initialize trainer pointer
        pointer = WeightPointerIPC(
            base_model=self._base_model,
            adapter_path=str(self._initial_adapter) if self._initial_adapter else None,
            iteration=self.gating_state.last_iteration,
            updated_at=now_ts(),
            version=self.gating_state.last_iteration,
            name="trainer",
        )
        self._pointer_store.save(pointer)
        
        self._trainer_process = mp.Process(
            target=run_trainer_worker,
            args=(trainer_config, self.queue),
            name="trainer_worker",
            daemon=False,
        )
        self._trainer_process.start()
        console.print(f"[green]Started trainer worker (PID: {self._trainer_process.pid})[/green]")
    
    def _stop_workers(self) -> None:
        """Stop all worker processes."""
        console.print("[yellow]Stopping workers...[/yellow]")
        try:
            self.queue.send("rollout_requests", MessageType.SHUTDOWN, {}, source="orchestrator")
            self.queue.send("control", MessageType.SHUTDOWN, {}, source="orchestrator")
        except Exception:
            pass
        
        # Terminate processes
        for name, proc in [("inference", self._inference_process), ("trainer", self._trainer_process)]:
            if proc and proc.is_alive():
                console.print(f"  Stopping {name} worker...")
                proc.terminate()
                proc.join(timeout=10.0)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=5.0)
        
        self.queue.stop()
        console.print("[green]Workers stopped[/green]")
    
    def _wait_for_inference(self, timeout: float = 60.0) -> bool:
        """Wait for inference worker to be ready via queue health check."""
        start = time.time()
        while time.time() - start < timeout and not self._shutdown:
            try:
                self.queue.send("rollout_requests", MessageType.HEALTH_CHECK, {}, source="orchestrator")
                msg = self.queue.receive("rollout_responses", timeout=1.0)
                if msg and msg.msg_type == MessageType.HEALTH_RESPONSE:
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        return False
    
    def _generate_rollout_via_api(
        self,
        task: GeneratedTask,
        rollouts_per_task: int,
    ) -> List[Rollout]:
        """Generate rollouts for a task via inference API."""
        import requests
        rollouts = []
        
        for k in range(rollouts_per_task):
            try:
                resp = requests.post(
                    f"http://localhost:{self.cfg.serve.port}/internal/rollout",
                    json={
                        "prompt": task.prompt,
                        "max_tokens": int(self.cfg.rft.max_new_tokens),
                        "temperature": float(self.cfg.rft.temperature),
                        "top_p": float(self.cfg.infer.top_p),
                        "top_k": self.cfg.infer.top_k,
                        "seed": int(time.time() * 1000) % (2**31 - 1),
                        "include_tokens": True,
                        "include_logprobs": True,
                    },
                    timeout=120.0,
                )
                
                if resp.status_code != 200:
                    console.print(f"[red]Rollout API error: {resp.status_code}[/red]")
                    continue
                
                data = resp.json()
                completion = data.get("completion", "")
                
                # Run verifier
                from ..util import ensure_dir
                wdir = ensure_dir(self.project_root / "runs" / ".temp" / task.id / f"rollout_{k:02d}")
                (wdir / "main.py").write_text(completion, encoding="utf-8")
                
                # Write tests
                tests_dir = ensure_dir(wdir / "tests")
                (tests_dir / "test_task.py").write_text(task.tests, encoding="utf-8")
                
                t0 = time.time()
                if self.cfg.rlm.verifier_backend == "docker":
                    res = docker_verify(
                        task.prompt,
                        completion,
                        str(wdir),
                        timeout_s=int(self.cfg.rlm.verifier_timeout_s),
                        image=self.cfg.rlm.docker_image,
                        memory_mb=int(self.cfg.rlm.docker_memory_mb),
                        cpus=float(self.cfg.rlm.docker_cpus),
                        pids=int(self.cfg.rlm.docker_pids),
                    )
                else:
                    from ..verifiers.pytest_verifier import verify as pytest_verify
                    res = pytest_verify(
                        task.prompt,
                        completion,
                        str(wdir),
                        timeout_s=int(self.cfg.rlm.verifier_timeout_s),
                    )
                latency_ms = (time.time() - t0) * 1000.0
                
                passed = bool(getattr(res, "passed", False))
                reward = float(getattr(res, "reward", 0.0))
                
                rollouts.append(Rollout(
                    task_id=task.id,
                    prompt=task.prompt,
                    completion=completion,
                    token_ids=data.get("token_ids", []),
                    prompt_len=data.get("prompt_len", 0),
                    logprobs=data.get("logprobs"),
                    passed=passed,
                    reward=reward,
                    verifier_latency_ms=latency_ms,
                    weight_adapter=self._pointer_store.load("inference", self._base_model).adapter_path,
                ))
                
                if passed:
                    self._passed_samples.append({
                        "id": task.id,
                        "prompt": task.prompt,
                        "response": completion,
                        "reward": reward,
                        "ts": now_ts(),
                    })
                    
            except Exception as e:
                console.print(f"[red]Rollout error: {e}[/red]")
                continue
        
        return rollouts

    def _generate_rollout_via_queue(
        self,
        task: GeneratedTask,
        rollouts_per_task: int,
    ) -> List[Rollout]:
        """Generate rollouts for a task via the message queue."""
        rollouts: List[Rollout] = []

        for k in range(rollouts_per_task):
            try:
                req = self.queue.send(
                    "rollout_requests",
                    MessageType.ROLLOUT_REQUEST,
                    {
                        "prompt": task.prompt,
                        "max_tokens": int(self.cfg.rft.max_new_tokens),
                        "temperature": float(self.cfg.rft.temperature),
                        "top_p": float(self.cfg.infer.top_p),
                        "top_k": self.cfg.infer.top_k,
                        "seed": int(time.time() * 1000) % (2**31 - 1),
                    },
                    source="orchestrator",
                )

                # Wait for matching response
                response = None
                start = time.time()
                while time.time() - start < self._rollout_timeout_s:
                    msg = self.queue.receive("rollout_responses", timeout=0.5)
                    if not msg:
                        continue
                    if msg.msg_type != MessageType.ROLLOUT_RESPONSE:
                        continue
                    if msg.payload.get("request_id") == req.msg_id:
                        response = msg
                        break

                if response is None:
                    console.print("[red]Rollout queue timeout[/red]")
                    continue

                data = response.payload
                completion = data.get("completion", "")

                # Run verifier
                wdir = ensure_dir(self.project_root / "runs" / ".temp" / task.id / f"rollout_{k:02d}")
                (wdir / "main.py").write_text(completion, encoding="utf-8")

                tests_dir = ensure_dir(wdir / "tests")
                (tests_dir / "test_task.py").write_text(task.tests, encoding="utf-8")

                t0 = time.time()
                if self.cfg.rlm.verifier_backend == "docker":
                    res = docker_verify(
                        task.prompt,
                        completion,
                        str(wdir),
                        timeout_s=int(self.cfg.rlm.verifier_timeout_s),
                        image=self.cfg.rlm.docker_image,
                        memory_mb=int(self.cfg.rlm.docker_memory_mb),
                        cpus=float(self.cfg.rlm.docker_cpus),
                        pids=int(self.cfg.rlm.docker_pids),
                    )
                else:
                    from ..verifiers.pytest_verifier import verify as pytest_verify
                    res = pytest_verify(
                        task.prompt,
                        completion,
                        str(wdir),
                        timeout_s=int(self.cfg.rlm.verifier_timeout_s),
                    )
                latency_ms = (time.time() - t0) * 1000.0

                passed = bool(getattr(res, "passed", False))
                reward = float(getattr(res, "reward", 0.0))

                rollouts.append(
                    Rollout(
                        task_id=task.id,
                        prompt=task.prompt,
                        completion=completion,
                        token_ids=data.get("token_ids", []),
                        prompt_len=data.get("prompt_len", 0),
                        logprobs=data.get("logprobs"),
                        passed=passed,
                        reward=reward,
                        verifier_latency_ms=latency_ms,
                        weight_adapter=self._pointer_store.load("inference", self._base_model).adapter_path,
                    )
                )

                if passed:
                    self._passed_samples.append(
                        {
                            "id": task.id,
                            "prompt": task.prompt,
                            "response": completion,
                            "reward": reward,
                            "ts": now_ts(),
                        }
                    )

            except Exception as e:
                console.print(f"[red]Rollout error: {e}[/red]")
                continue

        return rollouts
    
    def _send_training_batch(self, rollouts: List[Rollout], iteration: int, run_id: str) -> Message:
        """Send a training batch to the trainer worker via queue."""
        save_checkpoint = True

        payload = {
            "iteration": iteration,
            "run_id": run_id,
            "save_checkpoint": save_checkpoint,
            "rollouts": [
                {
                    "task_id": r.task_id,
                    "prompt": r.prompt,
                    "completion": r.completion,
                    "token_ids": r.token_ids,
                    "prompt_len": r.prompt_len,
                    "logprobs": r.logprobs,
                    "passed": r.passed,
                    "reward": r.reward,
                    "verifier_latency_ms": r.verifier_latency_ms,
                    "weight_adapter": r.weight_adapter,
                }
                for r in rollouts
            ],
        }
        return self.queue.send("train_batches", MessageType.TRAIN_BATCH, payload, source="orchestrator")

    def _drain_queue(self, queue_name: str) -> None:
        """Drain all pending messages from a queue."""
        while True:
            msg = self.queue.receive(queue_name, timeout=0)
            if msg is None:
                break
    
    def run_iteration(self, iteration: int) -> bool:
        """Run a single orchestrated RLM iteration."""
        console.print(f"\n[bold blue]=== Orchestrated RLM Iteration {iteration} ===[/bold blue]")
        
        run = new_run(self.project_root, "rlm")
        snapshot_config(self.cfg.model_dump(), run.config_snapshot_path)
        
        # Generate tasks using a temporary LLM instance
        # (In future: task generation could also go through inference worker)
        console.print("  [dim]Generating tasks...[/dim]")
        
        llm = get_llm_backend(self.cfg.model.backend)
        pointer = self._pointer_store.load("inference", self._base_model)
        llm.load(
            pointer.base_model,
            max_seq_len=self.cfg.model.max_seq_len,
            dtype=self.cfg.model.dtype,
            trust_remote_code=self.cfg.model.trust_remote_code,
        )
        if pointer.adapter_path:
            llm.apply_adapter(pointer.adapter_path)
        
        corpus_rows = load_corpus(self.corpus_path, max_size=int(self.cfg.rlm.corpus_max))
        existing_prompts = [row.get("prompt", "") for row in corpus_rows if row.get("prompt")]
        
        tasks = build_tasks(
            llm,
            self.cfg,
            require_recursion=bool(self.cfg.rlm.require_recursion),
            tasks_per_iter=int(self.cfg.rlm.tasks_per_iter),
            mutations_per_task=int(self.cfg.rlm.mutations_per_task),
            max_total=int(self.cfg.rlm.tasks_per_iter),
            existing_prompts=existing_prompts,
        )
        
        write_jsonl(run.run_dir / "tasks.jsonl", [task.__dict__ for task in tasks])
        
        # Generate rollouts via inference queue
        console.print(f"  [dim]Generating {len(tasks) * self.cfg.rlm.rollouts_per_task} rollouts...[/dim]")
        all_rollouts = []
        for i, task in enumerate(tasks):
            rollouts = self._generate_rollout_via_queue(
                task,
                rollouts_per_task=int(self.cfg.rlm.rollouts_per_task),
            )
            all_rollouts.extend(rollouts)
            if (i + 1) % 10 == 0:
                console.print(f"    {i + 1}/{len(tasks)} tasks completed")
        
        # Save rollouts
        write_jsonl(run.artifacts_dir / "rollouts.jsonl", [
            {
                "task_id": r.task_id,
                "prompt": r.prompt,
                "completion": r.completion,
                "token_ids": r.token_ids,
                "prompt_len": r.prompt_len,
                "logprobs": r.logprobs,
                "passed": r.passed,
                "reward": r.reward,
            }
            for r in all_rollouts
        ])
        
        # Train via trainer worker (queue)
        console.print("  [dim]Training on rollouts...[/dim]")
        train_msg = self._send_training_batch(all_rollouts, iteration, run.run_dir.name)

        train_resp = None
        start = time.time()
        while time.time() - start < self._train_timeout_s:
            msg = self.queue.receive("train_complete", timeout=1.0)
            if not msg:
                continue
            if msg.payload.get("request_id") == train_msg.msg_id:
                train_resp = msg
                break

        if train_resp is None:
            console.print("[red]Trainer timed out[/red]")
            return False

        train_result = train_resp.payload.get("result") or {}
        checkpoint_path = train_resp.payload.get("checkpoint_path")

        write_jsonl(
            run.metrics_path,
            [
                {
                    "ts": now_ts(),
                    "kind": "rlm_train",
                    "iteration": iteration,
                    "loss": train_result.get("loss"),
                    "num_tasks": train_result.get("num_tasks"),
                    "num_rollouts": train_result.get("num_rollouts"),
                }
            ],
        )

        if not checkpoint_path:
            console.print("[red]Trainer returned no checkpoint[/red]")
            return False

        copytree(Path(checkpoint_path), run.adapter_dir)

        # Drain any weight update notifications from trainer
        self._drain_queue("weight_updates")
        self._drain_queue("checkpoints")
        
        # Update corpus
        if self._passed_samples:
            append_corpus(self.corpus_path, self._passed_samples, max_size=int(self.cfg.rlm.corpus_max))
            self._passed_samples = []
        
        # Evaluate
        adapter_score = 0.0
        if self.cfg.rlm.benchmark_suite:
            suite_path = self.project_root / self.cfg.rlm.benchmark_suite
            if suite_path.exists():
                eval_path = run_eval(self.project_root, suite_path, run.adapter_dir)
                adapter_score = _score_from_eval(eval_path)
        
        holdout_score = None
        if self.cfg.rlm.holdout_suite:
            holdout_path = self.project_root / self.cfg.rlm.holdout_suite
            if holdout_path.exists():
                holdout_eval = run_eval(self.project_root, holdout_path, run.adapter_dir)
                holdout_score = _score_from_eval(holdout_eval)
        
        # Gating
        accepted = should_accept(
            adapter_score,
            self.gating_state,
            mode=self.cfg.rlm.gating,
            threshold=float(self.cfg.rlm.gating_threshold),
            ema_alpha=float(self.cfg.rlm.gating_ema_alpha),
        )
        self.gating_state = update_state(
            self.gating_state,
            iteration=iteration,
            score=adapter_score,
            adapter_path=str(run.adapter_dir),
            accepted=accepted,
            ema_alpha=float(self.cfg.rlm.gating_ema_alpha),
        )
        save_state(self.state_path, self.gating_state)
        
        # Update weight pointers
        if self.gating_state.current_adapter:
            train_pointer = WeightPointerIPC(
                base_model=self._base_model,
                adapter_path=self.gating_state.current_adapter,
                iteration=iteration,
                updated_at=now_ts(),
                version=iteration,
                name="trainer",
            )
            self._pointer_store.save(train_pointer)
            
            # Update inference pointer (hot reload)
            infer_staleness = int(getattr(self.cfg.rlm, "infer_staleness", 0))
            current_infer = self._pointer_store.load("inference", self._base_model)
            update_infer = infer_staleness <= 0
            if not update_infer:
                lag = max(0, int(iteration) - int(current_infer.iteration))
                update_infer = lag >= infer_staleness

            if update_infer:
                infer_pointer = WeightPointerIPC(
                    base_model=self._base_model,
                    adapter_path=self.gating_state.current_adapter,
                    iteration=iteration,
                    updated_at=now_ts(),
                    version=iteration,
                    name="inference",
                )
                self._pointer_store.save(infer_pointer)
                try:
                    self.queue.send(
                        "weight_forward",
                        MessageType.WEIGHT_UPDATE,
                        {
                            "adapter_path": self.gating_state.current_adapter,
                            "version": iteration,
                            "base_model": self._base_model,
                        },
                        source="orchestrator",
                    )
                except Exception as e:
                    console.print(f"[yellow]Hot reload trigger failed: {e}[/yellow]")
        
        # History
        append_history(
            self.history_path,
            {
                "iteration": iteration,
                "timestamp": now_ts(),
                "adapter_score": adapter_score,
                "holdout_score": holdout_score,
                "best_score": self.gating_state.best_score,
                "accepted": accepted,
                "adapter_dir": str(run.adapter_dir),
            },
        )
        
        gating_path = run.run_dir / "gating.json"
        gating_path.write_text(
            json.dumps(
                {
                    "iteration": iteration,
                    "accepted": accepted,
                    "adapter_score": adapter_score,
                    "holdout_score": holdout_score,
                    "best_score": self.gating_state.best_score,
                    "current_adapter": self.gating_state.current_adapter,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        
        if self.cfg.rlm.sleep_between > 0:
            time.sleep(float(self.cfg.rlm.sleep_between))
        
        return True
    
    def run(self) -> None:
        """Run the orchestrated RLM loop."""
        self._setup_signal_handlers()
        
        console.print("[bold green]Starting Orchestrated RLM[/bold green]")
        
        # Start queue manager
        self.queue.start()
        
        # Start workers
        self._start_inference_worker()
        self._start_trainer_worker()
        
        console.print("[dim]Waiting for inference server...[/dim]")
        if not self._wait_for_inference(timeout=120.0):
            console.print("[red]Inference server failed to start[/red]")
            self._stop_workers()
            return
        console.print("[green]Inference server ready[/green]")
        
        # Determine iteration range
        start_iter = self.gating_state.last_iteration + 1 if self.resume else 1
        total_iters = self.iterations
        
        def iter_range():
            if total_iters == 0:
                i = start_iter
                while True:
                    yield i
                    i += 1
            else:
                for i in range(start_iter, start_iter + total_iters):
                    yield i
        
        try:
            for iteration in iter_range():
                if self._shutdown:
                    break
                
                success = self.run_iteration(iteration)
                if not success:
                    console.print("[red]Iteration failed, stopping[/red]")
                    break
                    
        except KeyboardInterrupt:
            console.print("[yellow]Interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"[red]Orchestrator error: {e}[/red]")
            traceback.print_exc()
        finally:
            self._stop_workers()


def run_rlm_orchestrated(
    project_root: Path,
    cfg: ProjectConfig,
    *,
    model_spec: Optional[str] = None,
    iterations: Optional[int] = None,
    resume: bool = False,
) -> None:
    """Run multi-process orchestrated RLM loop.
    
    This mode spawns separate inference and trainer processes,
    coordinating via weight pointers and queue messages.
    
    Benefits:
    - Inference server remains responsive during training
    - Hot-reload of weights without restart
    - Better resource isolation
    - Foundation for distributed training
    """
    spec = model_spec or cfg.model.id
    iters = iterations or cfg.rlm.iterations
    
    orchestrator = RLMOrchestrator(
        project_root=project_root,
        cfg=cfg,
        model_spec=spec,
        iterations=iters,
        resume=resume,
    )
    orchestrator.run()


def collect_rollouts_via_api(
    tasks: List[GeneratedTask],
    cfg: ProjectConfig,
    api_url: str,
    artifacts_dir: Path,
    verifier_backend: str,
    weight_adapter: Optional[str],
) -> tuple[List[Rollout], list[dict]]:
    """Collect rollouts via inference API (for legacy loop with external inference)."""
    try:
        import requests
    except ModuleNotFoundError:
        requests = None
    rollouts: List[Rollout] = []
    passed_samples: list[dict] = []

    # Probe the API endpoint; fall back to local inference if unreachable.
    _api_available = False
    if requests is not None:
        try:
            requests.get(api_url, timeout=2.0)
            _api_available = True
        except Exception:
            _api_available = False

    if not _api_available:
        # Resolve model spec to separate base model from adapter.
        base_model, resolved_adapter, _meta = resolve_model_spec(
            Path.cwd(), cfg.model.id, cfg
        )
        llm = get_llm_backend(cfg.model.backend)
        llm.load(
            base_model,
            max_seq_len=cfg.model.max_seq_len,
            dtype=cfg.model.dtype,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        adapter_to_apply = weight_adapter or (str(resolved_adapter) if resolved_adapter else None)
        if adapter_to_apply:
            llm.apply_adapter(adapter_to_apply)
        for task in tasks:
            for k in range(int(cfg.rlm.rollouts_per_task)):
                try:
                    gen = llm.generate_with_logprobs(
                        task.prompt,
                        max_new_tokens=int(cfg.rft.max_new_tokens),
                        temperature=float(cfg.rft.temperature),
                        top_p=float(cfg.infer.top_p),
                        top_k=cfg.infer.top_k,
                        seed=int(time.time() * 1000) % (2**31 - 1),
                    )
                except TypeError:
                    gen = llm.generate_with_logprobs(
                        task.prompt,
                        max_new_tokens=int(cfg.rft.max_new_tokens),
                        temperature=float(cfg.rft.temperature),
                        top_p=float(cfg.infer.top_p),
                        top_k_sampling=cfg.infer.top_k,
                        seed=int(time.time() * 1000) % (2**31 - 1),
                    )
                completion = gen.text[len(task.prompt) :] if gen.text.startswith(task.prompt) else gen.text
                wdir = ensure_dir(artifacts_dir / task.id / f"rollout_{k:02d}")
                (wdir / "main.py").write_text(completion, encoding="utf-8")
                (ensure_dir(wdir / "tests") / "test_task.py").write_text(task.tests, encoding="utf-8")
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
                    res = pytest_verify(
                        task.prompt,
                        completion,
                        str(wdir),
                        timeout_s=int(cfg.rlm.verifier_timeout_s),
                    )
                latency_ms = (time.time() - t0) * 1000.0
                passed = bool(getattr(res, "passed", False))
                reward = float(getattr(res, "reward", 0.0))
                rollouts.append(
                    Rollout(
                        task_id=task.id,
                        prompt=task.prompt,
                        completion=completion,
                        token_ids=list(gen.token_ids),
                        prompt_len=gen.prompt_len,
                        logprobs=list(gen.logprobs) if gen.logprobs else None,
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
    
    for task in tasks:
        for k in range(int(cfg.rlm.rollouts_per_task)):
            try:
                resp = requests.post(
                    f"{api_url}/internal/rollout",
                    json={
                        "prompt": task.prompt,
                        "max_tokens": int(cfg.rft.max_new_tokens),
                        "temperature": float(cfg.rft.temperature),
                        "top_p": float(cfg.infer.top_p),
                        "top_k": cfg.infer.top_k,
                        "seed": int(time.time() * 1000) % (2**31 - 1),
                        "include_tokens": True,
                        "include_logprobs": True,
                    },
                    timeout=120.0,
                )
                
                if resp.status_code != 200:
                    continue
                
                data = resp.json()
                completion = data.get("completion", "")
                
                wdir = ensure_dir(artifacts_dir / task.id / f"rollout_{k:02d}")
                (wdir / "main.py").write_text(completion, encoding="utf-8")
                (ensure_dir(wdir / "tests") / "test_task.py").write_text(task.tests, encoding="utf-8")
                
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
                
                rollouts.append(Rollout(
                    task_id=task.id,
                    prompt=task.prompt,
                    completion=completion,
                    token_ids=data.get("token_ids", []),
                    prompt_len=data.get("prompt_len", 0),
                    logprobs=data.get("logprobs"),
                    passed=passed,
                    reward=reward,
                    verifier_latency_ms=latency_ms,
                    weight_adapter=weight_adapter,
                ))
                
                if passed:
                    passed_samples.append({
                        "id": task.id,
                        "prompt": task.prompt,
                        "response": completion,
                        "reward": reward,
                        "ts": now_ts(),
                    })
                    
            except Exception as e:
                console.print(f"[red]Rollout error: {e}[/red]")
                continue
    
    return rollouts, passed_samples
