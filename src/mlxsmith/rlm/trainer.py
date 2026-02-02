from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

from ..config import ProjectConfig
from ..util import now_ts
from .inference import Rollout


def train_on_rollouts(
    llm,
    rollouts: Iterable[Rollout],
    cfg: ProjectConfig,
    *,
    optimizer: object,
    train_adapter: Optional[str] = None,
    ref_llm: Optional[object] = None,
) -> list[dict]:
    grouped = defaultdict(list)
    for r in rollouts:
        grouped[r.task_id].append(r)

    metrics_rows: list[dict] = []

    for task_id, rows in grouped.items():
        if not rows:
            continue

        mean_r = sum(r.reward for r in rows) / max(1, len(rows))
        std_r = (sum((r.reward - mean_r) ** 2 for r in rows) / max(1, len(rows))) ** 0.5
        advs = [r.reward - mean_r for r in rows]
        if bool(cfg.rft.normalize_advantage) and std_r > 1e-6:
            advs = [a / std_r for a in advs]

        def loss_fn(_model):
            loss = llm.mx.array(0.0)  # type: ignore
            for rollout, adv in zip(rows, advs):
                logp = llm.sequence_logprob(rollout.token_ids, prompt_len=rollout.prompt_len)
                if rollout.logprobs and rollout.weight_adapter and rollout.weight_adapter != train_adapter:
                    behavior_logp = llm.mx.array(sum(rollout.logprobs))  # type: ignore
                    ratio = llm.mx.exp(logp - behavior_logp)  # type: ignore
                    pg = -ratio * llm.mx.array(float(adv))  # type: ignore
                else:
                    pg = -llm.mx.array(float(adv)) * logp  # type: ignore
                if ref_llm is not None and cfg.rft.kl_coeff > 0:
                    ref_logp = ref_llm.sequence_logprob(rollout.token_ids, prompt_len=rollout.prompt_len)
                    pg = pg + llm.mx.array(cfg.rft.kl_coeff) * (logp - ref_logp)  # type: ignore
                loss = loss + pg
            return loss / llm.mx.array(float(len(rows)))  # type: ignore

        lval, grads = llm.value_and_grad(loss_fn)
        if grads is not None:
            llm.apply_grads(optimizer, grads)

        metrics_rows.append(
            {
                "ts": now_ts(),
                "task_id": task_id,
                "mean_reward": mean_r,
                "std_reward": std_r,
                "loss": float(lval.item()) if hasattr(lval, "item") else float(lval),
                "verifier_latency_ms": sum(r.verifier_latency_ms for r in rows) / max(1, len(rows)),
                "weight_adapter": rows[0].weight_adapter,
            }
        )

    return metrics_rows
